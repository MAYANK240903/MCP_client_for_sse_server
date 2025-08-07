import asyncio
import json
import os
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # Configure Gemini
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Initialize persistent chat session
        self.chat_session = None
        self.conversation_history = []
        self.available_tools = []
        
    @staticmethod
    def safe_json(obj):
        """Recursively convert protobuf objects to serializable Python types."""
        # Handle protobuf repeated fields (lists)
        if type(obj).__name__ == "RepeatedComposite" or isinstance(obj, list):
            return [MCPClient.safe_json(item) for item in obj]
        # Handle protobuf map fields (dicts)
        if type(obj).__name__ == "MapComposite" or isinstance(obj, dict):
            return {k: MCPClient.safe_json(v) for k, v in obj.items()}
        # Handle protobuf messages
        if hasattr(obj, "ListFields"):
            return {k.name: MCPClient.safe_json(v) for k, v in obj.ListFields()}
        # Handle basic types
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        # Fallback to string
        return str(obj)
        
    async def direct_tool_call(self, tool_name: str, args: dict = None) -> str:
        """Call MCP tool directly without using Gemini"""
        try:
            if args is None:
                args = {}
            
            result = await self.session.call_tool(tool_name, args)
            return f"Tool {tool_name} result:\n{result.content}"
        except Exception as e:
            return f"Error calling tool {tool_name}: {str(e)}"

    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""
        # Store the context managers so they stay alive
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        # Initialize
        await self.session.initialize()

        # List available tools to verify connection
        print("Initialized SSE client...")
        print("Listing tools...")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
        
        # Initialize tools and chat session
        await self._initialize_tools()
        self._initialize_chat_session()

    async def _initialize_tools(self):
        """Initialize and convert tools for Gemini"""
        response = await self.session.list_tools()
        self.available_tools = []
        
        # Convert MCP tools to Gemini function format
        for tool in response.tools:
            # Handle properties conversion
            properties = {}
            if hasattr(tool.inputSchema, 'properties') and tool.inputSchema.properties:
                for prop_name, prop_schema in tool.inputSchema.properties.items():
                    # Convert MCP property schema to Gemini format
                    gemini_prop = genai.protos.Schema(
                        type=self._convert_type(prop_schema.get('type', 'string')),
                        description=prop_schema.get('description', '')
                    )
                    properties[prop_name] = gemini_prop
            
            # Handle required fields
            required = []
            if hasattr(tool.inputSchema, 'required') and tool.inputSchema.required:
                required = tool.inputSchema.required
            
            gemini_tool = genai.protos.Tool(
                function_declarations=[
                    genai.protos.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description or "",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties=properties,
                            required=required
                        )
                    )
                ]
            )
            self.available_tools.append(gemini_tool)

    def _initialize_chat_session(self):
        """Initialize a persistent chat session with Gemini"""
        # Create system prompt that explains available tools
        tool_descriptions = []
        for tool in self.available_tools:
            for func_decl in tool.function_declarations:
                tool_descriptions.append(f"- {func_decl.name}: {func_decl.description}")
        
        system_prompt = f"""You are a helpful AI assistant with access to the following tools:

{chr(10).join(tool_descriptions)}

You can use these tools to help answer user questions. When you need to use a tool, I will execute it for you and provide the results. Please be conversational and remember our chat history.
"""
        
        # Start chat session with system message
        self.chat_session = self.model.start_chat(history=[])
        print(f"Chat session initialized with {len(self.available_tools)} tools available")

    def get_slack_system_prompt(self):
        """Get system prompt for Slack formatting and strict tool call requirements"""
        return """You are a helpful AI assistant integrated with Slack. Format all responses using Slack's formatting conventions only.

    - Only call a tool when you have every required field for that tool. Do **not** attempt any tool call before all required fields are present and validated.
    - **You must make only one tool call at a time. Never attempt to call multiple tools in a single response. Wait for the result of each tool call before considering another.**
    - If a tool call fails or returns an error, always try for calling tool again with corrected parameters.
    - Summarize each attempt and its result in Slack format.
    - Never provide any API URN, endpoint, or raw JSON data.
    - Use *bold* for emphasis, _italic_ for secondary emphasis, and `inline code` for technical terms.
    - Use ```code blocks``` for multi-line code, but do not include JSON or API URN.
    - Use • for bullet points and 1. 2. 3. for numbered lists.
    - Keep responses concise, readable, and conversational.
    - Always explain what you tried and the outcome.
    - If all attempts fail, summarize the errors and suggest next steps in Slack format.
    - Donot send any raw JSON or API URN data.
    Remember: Only use Slack formatting. Do not return any API URN, endpoint, or raw JSON format data in your responses.
    Wait for the complete result of all the apis called before responding.
    Do not respond until you have the final answer you asked for.
    """
    
    async def process_query(self, query: str) -> str:
        """Process a query using Gemini with persistent chat session and Slack formatting, calling only one tool per query and providing tool schema on error."""
        if self.chat_session is None:
            return "Error: Chat session not initialized. Please connect to server first."

        try:
            # Send Slack formatting instruction if not already sent
            if not hasattr(self, '_slack_formatting_sent'):
                self.chat_session.send_message(self.get_slack_system_prompt())
                self._slack_formatting_sent = True

            # Send user query to persistent chat session
            response = self.chat_session.send_message(
                query,
                tools=self.available_tools if self.available_tools else None
            )

            final_text = []
            tool_call_handled = False  # Only handle one tool call per cycle

            if response.parts:
                for part in response.parts:
                    if hasattr(part, 'text') and part.text:
                        formatted_text = self.format_for_slack(part.text)
                        final_text.append(formatted_text)
                    elif hasattr(part, 'function_call') and not tool_call_handled:
                        function_call = part.function_call
                        tool_name = function_call.name
                        tool_args = dict(function_call.args)

                        # tool_notification = f"🔧 *Calling tool:* `{tool_name}`\n```\n{json.dumps(tool_args, indent=2)}\n```"
                        tool_notification = f"🔧 *Calling tool:* `{tool_name}`\n```\n{json.dumps(self.safe_json(tool_args), indent=2)}\n```"
                        final_text.append(tool_notification)

                        max_retries = 2
                        attempt = 0
                        success = False
                        while attempt < max_retries and not success:
                            try:
                                # result = await self.session.call_tool(tool_name, tool_args)
                                result = await self.session.call_tool(tool_name, self.safe_json(tool_args))
                                function_response = genai.protos.Part(
                                    function_response=genai.protos.FunctionResponse(
                                        name=tool_name,
                                        response={"result": str(result.content)}
                                    )
                                )
                                follow_up_response = self.chat_session.send_message(function_response)
                                if follow_up_response.text:
                                    formatted_followup = self.format_for_slack(follow_up_response.text)
                                    final_text.append(formatted_followup)
                                success = True
                            except Exception as e:
                                attempt += 1
                                error_msg = f"❌ *Tool Error (attempt {attempt}):* `{tool_name}`\n> {str(e)}"
                                final_text.append(error_msg)
                                if attempt < max_retries:
                                    # Fetch tool schema and provide it to Gemini
                                    tool_schema = await self._get_tool_schema(tool_name)
                                    error_prompt = (
                                        f"The tool `{tool_name}` failed with error: {str(e)}.\n"
                                        f"Tool Schema:\n{tool_schema}\n"
                                        # f"Current arguments: ```\n{json.dumps(tool_args, indent=2)}\n```\n"
                                        f"Current arguments: ```\n{json.dumps(self.safe_json(tool_args), indent=2)}\n```\n"
                                        "Please suggest corrected parameters and try again, making sure all required fields are present."
                                    )
                                    error_response = self.chat_session.send_message(error_prompt)
                                    if error_response.parts:
                                        for error_part in error_response.parts:
                                            if hasattr(error_part, 'function_call'):
                                                # Update tool_args for next retry
                                                new_function_call = error_part.function_call
                                                if new_function_call.name == tool_name:
                                                    tool_args = dict(new_function_call.args)
                                                    # final_text.append(f"🔄 Retrying with updated args: ```\n{json.dumps(tool_args, indent=2)}\n```")
                                                    final_text.append(f"🔄 Retrying with updated args: ```\n{json.dumps(self.safe_json(tool_args), indent=2)}\n```")
                                                    break
                                            elif hasattr(error_part, 'text') and error_part.text:
                                                formatted_error_response = self.format_for_slack(error_part.text)
                                                final_text.append(formatted_error_response)
                                else:
                                    error_response = self.chat_session.send_message(
                                        f"The tool {tool_name} failed after {max_retries} attempts. Please suggest an alternative approach."
                                    )
                                    if error_response.text:
                                        formatted_error_response = self.format_for_slack(error_response.text)
                                        final_text.append(formatted_error_response)
                        tool_call_handled = True  # Only process the first tool call
                        break  # Stop after handling one tool call

            result = "\n\n".join(final_text) if final_text else "No response generated."

            self.conversation_history.append({
                "query": query,
                "response": result,
                "timestamp": asyncio.get_event_loop().time()
            })

            return result

        except Exception as e:
            error_msg = f"❌ *Error processing query:* {str(e)}"
            print(error_msg)
            return error_msg
    
    async def _get_tool_schema(self, tool_name: str) -> str:
        """Get detailed schema information for a specific tool"""
        try:
            response = await self.session.list_tools()
            for tool in response.tools:
                if tool.name == tool_name:
                    schema_info = f"""
Tool: {tool.name}
Description: {tool.description}
Input Schema: {json.dumps(self.safe_json(tool.inputSchema), indent=2)}
"""
                    return schema_info
            return f"Schema not found for tool: {tool_name}"
        except Exception as e:
            return f"Error retrieving schema for {tool_name}: {str(e)}"
    
    def _convert_type(self, mcp_type: str) -> genai.protos.Type:
        """Convert MCP type to Gemini type"""
        type_mapping = {
            'string': genai.protos.Type.STRING,
            'number': genai.protos.Type.NUMBER,
            'integer': genai.protos.Type.INTEGER,
            'boolean': genai.protos.Type.BOOLEAN,
            'array': genai.protos.Type.ARRAY,
            'object': genai.protos.Type.OBJECT
        }
        return type_mapping.get(mcp_type.lower(), genai.protos.Type.STRING)

    def clear_conversation(self):
        """Clear conversation history and start fresh chat session"""
        self.conversation_history = []
        if self.available_tools:
            self._initialize_chat_session()
            print("Conversation cleared. Starting fresh chat session.")
        else:
            print("Cannot clear conversation - tools not initialized.")

    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation history"""
        if not self.conversation_history:
            return "No conversation history."
        
        summary = f"Conversation History ({len(self.conversation_history)} exchanges):\n\n"
        for i, exchange in enumerate(self.conversation_history[-5:], 1):  # Show last 5
            summary += f"{i}. User: {exchange['query'][:100]}{'...' if len(exchange['query']) > 100 else ''}\n"
            summary += f"   Assistant: {exchange['response'][:100]}{'...' if len(exchange['response']) > 100 else ''}\n\n"
        
        return summary

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    async def chat_loop(self):
        """Run an interactive chat loop with persistent memory"""
        print("\nMCP Client Started!")
        print("Type your queries, 'clear' to reset conversation, 'history' to see summary, or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                elif query.lower() == 'clear':
                    self.clear_conversation()
                    continue
                elif query.lower() == 'history':
                    print("\n" + self.get_conversation_summary())
                    continue
                elif query.startswith('direct:'):
                    tool_name = query[7:].strip()
                    response = await self.direct_tool_call(tool_name)
                    print("\n" + response)
                    continue
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")


    def format_for_slack(self, text: str) -> str:
        """Apply additional Slack-specific formatting"""
        # Ensure proper spacing around code blocks
        text = text.replace("```", "\n```\n")
        
        # Clean up extra newlines
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        
        return text.strip()

    def format_tool_result_for_slack(self, tool_name: str, result) -> str:
        """Format tool results specifically for Slack"""
        formatted_result = f"✅ *{tool_name} Results:*\n"
        
        try:
            # Try to parse as JSON for better formatting
            if hasattr(result, 'content'):
                content = result.content
                if isinstance(content, (dict, list)):
                    formatted_result += f"```json\n{json.dumps(content, indent=2)}\n```"
                else:
                    formatted_result += f"```\n{str(content)}\n```"
            else:
                formatted_result += f"```\n{str(result)}\n```"
        except Exception:
            formatted_result += f"```\n{str(result)}\n```"
        
        return formatted_result

async def main():
    server_url = "http://localhost:30876/sse"
    client = MCPClient()
    try:
        await client.connect_to_sse_server(server_url=server_url)
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
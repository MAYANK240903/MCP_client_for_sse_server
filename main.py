import asyncio
import logging
import os
from dotenv import load_dotenv
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient
# Import your MCP client
from mcp_client import MCPClient
# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
load_dotenv()
class SlackMCPBot:
    """Simplified Slack bot that forwards messages to MCP client."""
    def __init__(self, slack_bot_token: str, slack_app_token: str):
        self.app = AsyncApp(token=slack_bot_token)
        self.socket_mode_handler = AsyncSocketModeHandler(self.app, slack_app_token)
        self.client = AsyncWebClient(token=slack_bot_token)
        # Initialize MCP client
        self.mcp_client = MCPClient()
        # Set up event handlers
        self.app.event("app_mention")(self.handle_mention)
        self.app.message()(self.handle_message)
    
    async def initialize(self):
        """Initialize the MCP client connection."""
        try:
            server_url = "http://localhost:30876/sse"
            await self.mcp_client.connect_to_sse_server(server_url)
            logging.info("MCP client connected successfully")
            # Get bot info
            auth_info = await self.client.auth_test()
            self.bot_id = auth_info["user_id"]
            logging.info(f"Bot initialized with ID: {self.bot_id}")
        except Exception as e:
            logging.error(f"Failed to initialize: {e}")
            raise
    
    async def handle_mention(self, event, say):
        """Handle mentions of the bot in channels."""
        await self._process_message(event, say)
    
    async def handle_message(self, message, say):
        """Handle direct messages to the bot."""
        # Only process direct messages
        if message.get("channel_type") == "im" and not message.get("subtype"):
            await self._process_message(message, say)
    
    async def _process_message(self, event, say):
        """Process incoming messages using MCP client."""
        user_id = event.get("user")
        # Skip messages from the bot itself
        if user_id == getattr(self, "bot_id", None):
            return
        # Get text and remove bot mention if present
        text = event.get("text", "")
        if hasattr(self, "bot_id") and self.bot_id:
            text = text.replace(f"<@{self.bot_id}>", "").strip()
        channel = event["channel"]
        thread_ts = event.get("thread_ts", event.get("ts"))
        try:
            # Send typing indicator
            await self.client.chat_postMessage(
                channel=channel,
                text="Thinking...",
                thread_ts=thread_ts
            )
            # Process query using MCP client
            response = await self.mcp_client.process_query(text)
            # Send the response
            await say(text=response, channel=channel, thread_ts=thread_ts)
        except Exception as e:
            error_message = f"Sorry, I encountered an error: {str(e)}"
            logging.error(f"Error processing message: {e}", exc_info=True)
            await say(text=error_message, channel=channel, thread_ts=thread_ts)
    
    async def start(self):
        """Start the Slack bot."""
        await self.initialize()
        logging.info("Starting Slack bot...")
        await self.socket_mode_handler.start_async()
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            await self.mcp_client.cleanup()
            await self.socket_mode_handler.close_async()
            logging.info("Bot cleaned up successfully")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
async def main():
    """Initialize and run the Slack bot."""
    slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
    slack_app_token = os.getenv("SLACK_APP_TOKEN")
    if not slack_bot_token or not slack_app_token:
        raise ValueError("SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set in environment variables")
    bot = SlackMCPBot(slack_bot_token, slack_app_token)
    try:
        await bot.start()
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        await bot.cleanup()
if __name__ == "__main__":
    asyncio.run(main())
from fastmcp import Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from utils.config import MCP_CONFIG_DIR_PATH
from utils.load import mcp_tools_config
from utils.logger import get_logger
from utils.helpers import apply_config, create_dir
from pathlib import Path

mcp_logger = get_logger(__name__)

def handle_mcp_config(mcp_tools: dict, action: str):
    """Handle the MCP config"""
    success = False
    message = "Invalid action"
    create_dir(MCP_CONFIG_DIR_PATH)

    if mcp_tools is not None:
        data = {"mcp_clients": mcp_tools}
    else:
        mcp_tools = []
        data = {"mcp_clients": []}
    
    config_file_path = Path(MCP_CONFIG_DIR_PATH) / mcp_tools_config

    if not config_file_path.exists():
        action = "save"

    if action in ["save", "load"]:
        if action == "save":
            success, current_config = apply_config(config_file_path, "r")
            if not success:
                current_config = {}

            config = {
                "mcp_clients": mcp_tools,
            }

            success, _ = apply_config(config_file_path, "w", config)
            message = "MCP configuration saved successfully" if success else "Failed to save MCP configuration"
        elif action == "load":
            success, data = apply_config(config_file_path, "r")
            message = "MCP configuration loaded successfully" if success else "Failed to load MCP configuration (check if the configuration file exists or apply for new configuration)"
    return {"success": success, "message": message, "data": data}
    
class MCPClient:
    def __init__(self, server_path: str, transport_type: str):
        self.path = server_path
        self.transport_type = transport_type
        self.status = False
        self.mcp = None
        self.tools = []

    async def connect(self) -> bool:
        """Connect to the MCP server"""
        try:
            if self.transport_type == "stdio":
                transport = StdioTransport(command="python", args=[self.path], keep_alive=False)
            else:
                transport = StreamableHttpTransport(url=self.path)

            self.mcp = Client(transport)
            await self.mcp.__aenter__()

            if self.transport_type != "stdio":
                await self.mcp.ping()

            self.status = True
            return True
        except Exception as e:
            mcp_logger.error(f"Failed to connect to MCP server: {e}")
            self.status = False
            return False

    async def disconnect(self) -> bool:
        """Disconnect from the MCP server"""
        try:
            if self.mcp:
                await self.mcp.__aexit__(None, None, None)
        except Exception as e:
            mcp_logger.error(f"Failed to disconnect from MCP server: {e}")
            return False
        finally:
            self.mcp = None
            self.status = False
        return True

    async def get_tools(self):
        """Get the tools from the MCP server"""
        try:
            if not self.status:
                return None
            tools = await self.mcp.list_tools()
            for tool in tools:
                llm_tool = {
                    "type": "function",
                    "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                    },
                }
                self.tools.append(llm_tool)
            return self.tools
        except Exception as e:
            mcp_logger.error(f"Failed to get tools from MCP server: {e}")
            return "No tools available"

    async def call_tool(self, tool_name: str, arguments: dict):
        """Call the tool from the MCP server"""
        try:
            if not self.status:
                return None
            result = await self.mcp.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            mcp_logger.error(f"Failed to call tool from MCP server: {e}")
            return None

async def mcp_client(server_path: str, transport_type: str) -> MCPClient:
    """Create a MCP client"""
    client = MCPClient(server_path=server_path, transport_type=transport_type)
    return client


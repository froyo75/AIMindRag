#!/usr/bin/env python3
"""
Simple MCP Server with Streamable HTTP Transport
"""

from fastmcp import FastMCP


# Create a basic stateless MCP server
mcp = FastMCP(name="SimpleStreamableHTTPMCPServer", stateless_http=True)


@mcp.tool()
def hello_world(name: str = "World") -> str:
    """Say hello to someone"""
    return f"Hello, {name}!"


@mcp.tool()
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together"""
    return a + b


@mcp.tool()
def random_number(min_val: int = 0, max_val: int = 100) -> int:
    """Generate a random integer between min_val and max_val (inclusive)"""
    import random

    if min_val > max_val:
        min_val, max_val = max_val, min_val

    return random.randint(min_val, max_val)


@mcp.resource("simple://info")
async def get_server_info() -> str:
    """Get information about this server"""
    return "This is a simple MCP server with streamable HTTP transport. It supports tools for greeting, adding numbers, and generating random numbers."

def main():
    """Main entry point for the MCP server"""
    print("Starting Simple MCP Server...")
    print("Available tools: hello_world, add_numbers, random_number")
    print("Available resources: simple://info")

    # Run with streamable HTTP transport
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8888,
        path="/mcp",
        log_level="debug",
    )

if __name__ == "__main__":
    main()

from modules.llm import LLMClient
from modules.rag import RAGClient
from utils.config import LLM_SYSTEM_PROMPT
from fastmcp.prompts.prompt import TextContent
import uuid
import json
from utils.logger import get_logger

chat_logger = get_logger(__name__)

def get_available_tools(mcp_clients: dict):
    """Get the available tools from the MCP clients"""
    chat_logger.info("Getting available tools from the MCP clients")
    llm_tools = []
    for _, info in mcp_clients.items():
        if info.get("status") == "Connected":
            client = info.get("mcp_client")
            if client and getattr(client, "tools", None):
                llm_tools.extend(client.tools)
    return llm_tools

async def handle_tools_call(llm_provider: str, tools_call: list, mcp_clients: dict):
    """Handle the tools call"""
    result = "No tool found"
    tool_id = None
    tool_name = None
    tools_response = []

    for tool_call in tools_call:
        tool_id = tool_call["id"]
        tool_name = tool_call["function"]["name"]
        tool_args_raw = tool_call["function"]["arguments"]
        tool_args = tool_args_raw if isinstance(tool_args_raw, dict) else json.loads(tool_args_raw)
        for _, info in mcp_clients.items():
            mcp_client = info.get("mcp_client")
            if mcp_client:
                for mcp_tool in mcp_client.tools:
                    if mcp_tool["function"]["name"] == tool_name:
                        try:
                            chat_logger.info(f"Calling tool: {tool_name}")
                            call_result = await mcp_client.call_tool(tool_name, tool_args)
                            result = call_result.content
                        except Exception as e:
                            chat_logger.error(f"Error calling tool: {e}")
                            result = f'Error calling tool: {e}'
                        break
        if isinstance(result, list) and isinstance(result[0], TextContent):
            content = result[0].text
        else:
            content = result
        tools_response.append({
            "role": "tool",
            "content": content,
            "tool_call_id": tool_id
        })

    return tools_response

async def handle_stream_responses(llm_provider: str, messages: list, mcp_clients: dict, use_tools: bool, use_rag: bool, embedding_provider: str, rag_max_nb_results: int):
    """Handle the stream responses"""
    tools = get_available_tools(mcp_clients)
    tools_to_use = tools if use_tools and len(tools) > 0 else None

    while True:
        stream_content = ""
        tools_call = []
        tools_response = []
        llm_client = LLMClient(llm_provider)

        if use_rag:
            query = messages[-1].get("content")
            rag_client = RAGClient(embedding_provider)
            rag_results = rag_client.search_documents(query, rag_max_nb_results)
            rag_documents = rag_results["documents"][0]
            if rag_documents and len(rag_documents) > 0:
                context = "\n".join(rag_documents)
            else:
                context = "No relevant information found in the database"
            messages.append({"role": "user", "content": f"Answer the question based on the context.\n\nContext:\n{context}\n\nQuestion:\n{query}"})

        stream_response = llm_client.generate_response(messages, tools=tools_to_use)

        for chunk in stream_response:
            call_tool = False
            if llm_provider == "gpt":
                choice = chunk.choices[0]
                finish_reason = chunk.choices[0].finish_reason
                delta = choice.delta
                content = delta.content
                if delta.tool_calls:
                    for tool_call_chunck in delta.tool_calls:
                        index = tool_call_chunck.index
                        if len(tools_call) <= index:
                            tools_call.append(
                                {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            )
                        tool_call = tools_call[index]
                        if tool_call_chunck.id:
                            tool_call["id"] += tool_call_chunck.id
                        if tool_call_chunck.function.name:
                            tool_call["function"]["name"] += tool_call_chunck.function.name
                        if tool_call_chunck.function.arguments:
                            tool_call["function"]["arguments"] += tool_call_chunck.function.arguments
            else:
                choice = chunk.message
                finish_reason = chunk.done_reason
                content = choice.content
                if choice.tool_calls:
                    call_tool = True
                    for tool_call in choice.tool_calls:
                        tool_call_id = str(uuid.uuid4())
                        tools_call.append(
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                        )
            
            if not choice:
                chat_logger.error("Sorry, there was an error. Please try again!")
                raise Exception("Sorry, there was an error. Please try again!")

            if content is not None and content != "":
                stream_content += content or ""
                yield stream_content

            if finish_reason == "tool_calls" or call_tool:
                messages.append({"role": "assistant", "tool_calls": tools_call})
                tools_response = await handle_tools_call(llm_provider, tools_call, mcp_clients)
                messages.extend(tools_response)
                break
            elif finish_reason is None:
                continue
            elif finish_reason == "stop":
                messages.append({"role": "assistant", "content": stream_content})
                return
            else:
                chat_logger.error(f"LLM API error (finish reason: {finish_reason})!")
                raise Exception(f"LLM API error (finish reason: {finish_reason})!")
        
async def chat(message: str, history: list, llm_provider: str, mcp_clients: dict, use_tools: bool, use_rag: bool, embedding_provider: str, rag_max_nb_results: int):
    """Handle the chat"""
    try:
        messages = []
        
        system_prompt = LLM_SYSTEM_PROMPT
        messages.append({"role": "system", "content": system_prompt})
        
        for msg in history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append(msg)

        messages.append({"role": "user", "content": message})

        async for response in handle_stream_responses(llm_provider, messages, mcp_clients, use_tools, use_rag, embedding_provider, rag_max_nb_results):
            yield response

    except Exception as e:
        chat_logger.error(f"Error in chat: {str(e)}")
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        yield error_msg
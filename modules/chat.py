from modules.llm import LLMClient
from modules.rag import RAGClient
from utils.config import LLM_SYSTEM_PROMPT
from fastmcp.prompts.prompt import TextContent
import uuid
import json
import asyncio
import re
from utils.logger import get_logger

_STREAM_END = object()

# Async to handle/cancel the stream response
async def _chunks_async(stream_response):
    stream_iter = iter(stream_response)
    while True:
        chunk = await asyncio.to_thread(next, stream_iter, _STREAM_END)
        if chunk is _STREAM_END:
            break
        yield chunk

chat_logger = get_logger(__name__)

THINK_OPEN = '<think>'
THINK_CLOSE = '</think>'
RAG_CONTEXT_MSG = "Answer the question based on the context.\n\nContext:\n{context}\n\nQuestion:\n{query}"

def _content_as_query_string(content) -> str:
    """Last user message content for RAG may be str or GPT structured list."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and len(content) > 0:
        first = content[0]
        if isinstance(first, dict) and first.get("text"):
            return first["text"]
    return str(content)


def _normalize_system_prompt(llm_provider: str) -> dict:
    """Normalize the system prompt"""
    normalized_system_prompt = None
    if llm_provider != "claude":
        normalized_system_prompt = {
            "role": "system",
            "content": LLM_SYSTEM_PROMPT
        }
    return normalized_system_prompt

def _normalize_chat_message(msg: dict, llm_provider: str) -> dict:
    """Normalize the chat message format for the LLM"""
    excluded_keys = ("metadata", "options")
    normalized_msg = {key: msg[key] for key in msg if key not in excluded_keys}
    if llm_provider == "gpt":
        role = normalized_msg["role"]
        content = normalized_msg["content"][0]["text"]
        content_type = "output_text" if role == "assistant" else "input_text"
        normalized_msg["content"] = [{"type": content_type, "text": content}]
    elif llm_provider == "ollama":
        role = normalized_msg["role"]
        content = normalized_msg["content"]
        normalized_msg["content"] = "".join(part["text"] for part in content if part["type"] == "text")
    return normalized_msg

def _normalize_tools(tools: dict, llm_provider: str) -> dict:
    """Normalize the tools format for the LLM"""
    normalized_tools = []
    if llm_provider == "gpt":
        for tool in tools:
            normalized_tool = {
                'type': 'function',
                'name': tool["function"]["name"],
                'description': tool["function"]["description"],
                'parameters': tool["function"]["parameters"]
            }
            normalized_tools.append(normalized_tool)
    elif llm_provider == "claude":
        for tool in tools:
            normalized_tool = {
                'name': tool["function"]["name"],
                'description': tool["function"]["description"],
                'input_schema': tool["function"]["parameters"]
            }
            normalized_tools.append(normalized_tool)
    else:
        return tools
    return normalized_tools

def _normalize_tool_outputs(tool_id: str, tool_output: str, llm_provider: str) -> dict:
    """Normalize the tool output format for the LLM"""
    normalized_tool_output = {}
    if llm_provider == "gpt":
        normalized_tool_output = {
            "type": "function_call_output",
            "call_id": tool_id,
            "output": tool_output
        }
    elif llm_provider == "claude":
        normalized_tool_output = {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": tool_output
        }
    else:
        normalized_tool_output = {
            "role": "tool",
            "tool_call_id": tool_id,
            "content": tool_output
        }
    return normalized_tool_output

def _format_thinking_display(stream_reasoning, stream_content, llm_use_thinking) -> str:
    """Format the thinking display"""
    if not llm_use_thinking:
        return stream_content
    reasoning = (stream_reasoning or '')
    reasoning = re.sub(r'</?think>', '', reasoning, flags=re.IGNORECASE).strip()
    if not reasoning:
        return stream_content
    thinking_display = f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}\n{stream_content}"
    return thinking_display

def get_available_tools(mcp_clients: dict) -> list:
    """Get the available tools from the MCP clients"""
    chat_logger.info("Getting available tools from the MCP clients")
    llm_tools = []
    
    for _, info in mcp_clients.items():
        if info.get("status") == "Connected":
            client = info.get("mcp_client")
            if client and getattr(client, "tools", None):
                llm_tools.extend(client.tools)
    return llm_tools

async def handle_tools_call(llm_provider: str, tool_calls: list, mcp_clients: dict) -> list:
    """Handle the tools call"""
    result = "No tool found"
    tool_id = None
    tool_name = None
    tool_outputs = []

    for tool_call in tool_calls:
        if llm_provider == "gpt":
            tool_id = tool_call["call_id"]
            tool_name = tool_call["name"]
            tool_args_raw = tool_call["arguments"]
        elif llm_provider == "claude":
            tool_id = tool_call["id"]
            tool_name = tool_call["name"]
            tool_args_raw = tool_call["input"]
        else:
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
        tool_outputs.append(_normalize_tool_outputs(tool_id, content, llm_provider))
    return tool_outputs

async def handle_stream_responses(llm_provider: str, messages: list, mcp_clients: dict, use_tools: bool, use_rag: bool, embedding_provider: str, rag_max_nb_results: int, llm_use_thinking: bool):
    """Handle the stream responses"""
    tools = get_available_tools(mcp_clients)
    normalized_tools = _normalize_tools(tools, llm_provider)
    tools_to_use = normalized_tools if use_tools and len(normalized_tools) > 0 else None

    while True:
        stream_content = ""
        stream_reasoning = ""
        tool_calls = []
        tool_calls_map = {}
        tool_outputs = []
        current_tool_id = None
        current_tool_name = None
        current_tool_input = None
        normalized_tool_outputs = []
        llm_client = LLMClient(llm_provider, llm_use_thinking)

        if use_rag:
            query = _content_as_query_string(messages[-1].get("content"))
            rag_client = RAGClient(embedding_provider)
            rag_results = rag_client.search_documents(query, rag_max_nb_results)
            rag_documents = rag_results["documents"][0]
            if rag_documents and len(rag_documents) > 0:
                context = "\n".join(rag_documents)
            else:
                context = "No relevant information found in the database"
            messages.append({"role": "user", "content": RAG_CONTEXT_MSG.format(context=context, query=query)})

        stream_response = llm_client.generate_response(messages, tools=tools_to_use)
        last_display = ""
        async for chunk in _chunks_async(stream_response):
            call_tool = False
            content = None
            finish_reason = None
            
            if llm_provider == "gpt":
                if chunk.type.startswith("response."):
                    if chunk.type == "response.output_text.delta":
                        content = chunk.delta
                    elif chunk.type == "response.output_item.done" and chunk.item.type == "function_call":
                        tool_calls.append(
                            {
                                "type": "function_call",
                                "name": chunk.item.name,
                                "call_id": chunk.item.call_id,
                                "arguments": chunk.item.arguments,
                            }
                        )
                    elif chunk.type == "response.reasoning_summary_text.delta":
                        stream_reasoning += chunk.delta
                    elif chunk.type == "response.completed":
                        finish_reason = "tool_calls" if tool_calls else "stop"
                    elif chunk.type == "response.failed":
                        finish_reason = "failed"
                    elif chunk.type == "response.incomplete":
                        finish_reason = "stop"
                elif chunk.type == "error":
                    chat_logger.error(f"OpenAI stream error: {chunk.message}")
                    raise Exception(chunk.message)
            elif llm_provider == "claude":
                if chunk.type == "content_block_start":
                    if chunk.content_block.type == "tool_use":
                        tool_calls_map[chunk.index] = {
                            "id": chunk.content_block.id,
                            "name": chunk.content_block.name,
                            "input": ""
                        }
                elif chunk.type == "content_block_delta":
                    if chunk.delta.type == "text_delta":
                        content = chunk.delta.text
                    elif chunk.delta.type == "input_json_delta":
                        tool_calls_map[chunk.index]["input"] += chunk.delta.partial_json
                    elif chunk.delta.type == "thinking_delta":
                        stream_reasoning += chunk.delta.thinking
                elif chunk.type == "content_block_stop":
                    if chunk.index in tool_calls_map:
                        tool_data = tool_calls_map[chunk.index]
                        try:
                            args = json.loads(tool_data["input"]) if tool_data["input"] else {}
                        except json.JSONDecodeError:
                            args = {}
                        tool_calls.append({
                            "type": "tool_use",
                            "id": tool_data["id"],
                            "name": tool_data["name"],
                            "input": args
                        })
                        del tool_calls_map[chunk.index]
                elif chunk.type == "message_delta":
                    finish_reason = chunk.delta.stop_reason
                    if finish_reason == "tool_use":
                        finish_reason = "tool_calls"
                    elif finish_reason == "pause_turn":
                        finish_reason = None
                    else:
                        finish_reason = "stop"
                elif chunk.type == "message_stop":
                    finish_reason = "stop"
                elif chunk.type == "error":
                    chat_logger.error(f"Claude stream error: {chunk.error}")
                    raise Exception(str(chunk.error))
            else:
                if chunk.message.thinking:
                    stream_reasoning += chunk.message.thinking
                elif chunk.message.content != "":
                    content = chunk.message.content
                elif chunk.message.tool_calls:
                    call_tool = True
                    tool_calls.clear()
                    for tool_call in chunk.message.tool_calls:
                        tool_call_id = str(uuid.uuid4())
                        tool_calls.append(
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                        )
                elif chunk.done:
                    finish_reason = "tool_calls" if tool_calls else "stop"
                elif chunk.done_reason == "error":
                    chat_logger.error(f"Ollama stream error")
                    raise Exception("Ollama stream error")

            if not chunk:
                chat_logger.error("Sorry, there was an error. Please try again!")
                raise Exception("Sorry, there was an error. Please try again!")

            if content is not None and content != "":
                stream_content += content or ""

            display = _format_thinking_display(
                stream_reasoning, stream_content, llm_use_thinking
            )
            if display and display != last_display:
                last_display = display
                yield display

            if finish_reason == "tool_calls" or call_tool:
                normalized_tool_outputs = await handle_tools_call(llm_provider, tool_calls, mcp_clients)
                if llm_provider == "claude":
                    messages.append({"role": "assistant", "content": tool_calls})     
                    messages.append({"role": "user", "content": normalized_tool_outputs})               
                elif llm_provider == "gpt":
                    messages.extend(tool_calls)
                    messages.extend(normalized_tool_outputs)
                else:
                    messages.append({"role": "assistant", "tool_calls": tool_calls})
                    messages.extend(normalized_tool_outputs) 
                break
            elif finish_reason is None:
                continue
            elif finish_reason == "stop":
                messages.append({"role": "assistant", "content": stream_content})
                return
            else:
                chat_logger.error(f"LLM API error (finish reason: {finish_reason})!")
                raise Exception(f"LLM API error (finish reason: {finish_reason})!")
        
async def chat(message: str, history: list, llm_provider: str, mcp_clients: dict, use_tools: bool, use_rag: bool, embedding_provider: str, rag_max_nb_results: int, llm_use_thinking: bool):
    """Handle the chat"""
    try:
        messages = []

        normalized_system_prompt = _normalize_system_prompt(llm_provider)
        if normalized_system_prompt is not None:
            messages.append(normalized_system_prompt)
        
        for msg in history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append(_normalize_chat_message(msg, llm_provider))

        messages.append({"role": "user", "content": message})

        async for response in handle_stream_responses(llm_provider, messages, mcp_clients, use_tools, use_rag, embedding_provider, rag_max_nb_results, llm_use_thinking):
            yield response

    except Exception as e:
        chat_logger.error(f"Error in chat: {str(e)}")
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        yield error_msg

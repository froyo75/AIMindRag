import os
import gradio as gr
from utils.helpers import str_to_bool, delete_dir, create_dir
from utils.config import GRADIO_DEBUG, GRADIO_BIND_ADDRESS, GRADIO_PORT, GRADIO_TEMP_FOLDER_PATH, GRADIO_SHARE, GRADIO_PWA, GRADIO_ANALYTICS_ENABLED, RAG_DB_PATH, RAG_MARKDOWN, RAG_DEFAULT_DB_NAME, \
    LLM_DEFAULT_PROVIDER, LLM_DEFAULT_MODEL, LLM_DEFAULT_URL, LLM_DEFAULT_EMBEDDING_PROVIDER, LLM_DEFAULT_EMBEDDING_MODEL, RAG_DEFAULT_MAX_NB_RESULTS, LLM_DEFAULT_REASONING_EFFORT
from utils.load import available_llm_providers, available_rag_providers, handle_app_config
from modules.chat import chat
from modules.mcp import MCPClient, handle_mcp_config
from modules.llm import LLMClient
from modules.rag import RAGClient, get_vector_store_list
from modules.llm import handle_llm_config
from modules.rag import handle_vector_db, handle_rag_config
from pathlib import Path
from utils.logger import get_logger

ui_logger = get_logger(__name__)

# Set the environment variables
os.environ['GRADIO_TEMP_DIR'] = GRADIO_TEMP_FOLDER_PATH
os.environ['GRADIO_DEBUG'] = GRADIO_DEBUG
os.environ['GRADIO_ANALYTICS_ENABLED'] = GRADIO_ANALYTICS_ENABLED

# CSS for centering the checkbox, reverse the layout and the token metrics
css = """
.center-checkbox .block {
    min-height: unset !important;
    padding: 0 !important;
}

.center-checkbox {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding-top: 24px !important;
}

.chat-with-metrics-below {
    display: flex;
    flex-direction: column-reverse;
    gap: 0.75rem;
}

.token-metrics-centered {
    width: 100%;
    max-width: 100%;
    box-sizing: border-box;
    align-self: stretch;
    text-align: right;
}

.main-layout-row {
    flex-direction: row-reverse !important;
}
"""

def run_ui():
    """Run the Gradio UI"""
    with gr.Blocks(title="MCP Client") as ui:
        loaded_llm_model = gr.State("")
        loaded_embedding_model = gr.State("")
        selected_mcp_client = gr.State("")
        llm_reasoning_effort = gr.State(False)
        llm_base_url = gr.State("")
        use_tools = gr.State(True)
        mcp_clients = gr.State({})
        mcp_tools = gr.State({})
        available_llm_models = gr.State({})
        available_embedding_models = gr.State({})
        settings_sidebar_hidden = gr.State(True)

        with gr.Row(elem_classes=["main-layout-row"]):
            with gr.Column(scale=2, min_width=400, visible=False) as settings_column:
                # LLM Configuration Block
                with gr.Group():
                    gr.Markdown("##  LLM Configuration")
                    with gr.Row():
                        llm_provider = gr.Dropdown(
                            label="LLM Provider",
                            choices=available_llm_providers,
                            value=LLM_DEFAULT_PROVIDER,
                            interactive=True,
                        )

                        llm_model = gr.Dropdown(
                            label="LLM Model",
                            value=LLM_DEFAULT_MODEL,
                            allow_custom_value=True,
                        )

                        llm_api_key = gr.Textbox(
                            label="LLM API KEY",
                            type="password",
                            value="",
                            interactive=True,
                            info="Leave empty if already set",
                        )

                        llm_base_url = gr.Textbox(
                            label="LLM Base URL (only for Ollama)",
                            value=LLM_DEFAULT_URL,
                            interactive=True,
                        )

                    with gr.Row():
                    
                        llm_reasoning_effort = gr.Dropdown(
                            label="Reasoning effort",
                            choices=["minimal", "low", "medium", "high", "xhigh", "max"],
                            value=LLM_DEFAULT_REASONING_EFFORT,
                            interactive=True,
                            info="Take effect depending on the LLM model.",
                        )

                        llm_use_thinking = gr.Checkbox(
                            label="Use Reasoning",
                            value=False,
                            interactive=True,
                            elem_classes=["center-checkbox"],
                        )
                        
                    llm_save_btn = gr.Button("Apply", variant="primary", size="sm")
                        
                # MCP Server Management Block
                with gr.Group():
                    gr.Markdown("##  MCP Server Management")

                    use_tools = gr.Checkbox(
                        label="Use MCP Tools",
                        value=False,
                        interactive=True,
                    )
                    
                    with gr.Row():
                        mcp_client_path = gr.Textbox(
                            label="MCP Client Path",
                            value="http://127.0.0.1:8888/mcp",
                            interactive=True,
                        )

                        mcp_client_transport_type = gr.Dropdown(
                            label="MCP Client Transport Type",
                            choices=["http(s)", "stdio"],
                            value="http(s)",
                            interactive=True,
                        )

                    mcp_connect_btn = gr.Button("Connect", variant="primary", size="sm")
                    mcp_disconnect_btn = gr.Button("Disconnect", variant="secondary", size="sm")
                
                    with gr.Row():
                        clients_table = gr.Dataframe(
                            headers=["MCP client name", "Status"],
                            interactive=False,
                            show_search=True,
                            datatype="array",
                            label="MCP Clients",
                            max_height=100,
                            row_count=5,
                            row_limits=None,
                            column_count=2,
                            column_limits=None,
                        )
                    
                    with gr.Row():
                        tools_table = gr.Dataframe(
                            headers=["Tool name", "Description"],
                            interactive=False,
                            show_search=True,
                            datatype="array",
                            label="MCP Tools",
                            max_height=200,
                            row_count=5,
                            row_limits=None,
                            column_count=2,
                            column_limits=None,
                        )

                # RAG Management Block
                with gr.Group():
                    gr.Markdown("## 📚 RAG Management")

                    use_rag = gr.Checkbox(
                        label="Use RAG",
                        value=False,
                        interactive=True,
                    )
                    
                    with gr.Tab("Documents"):
                        file_upload = gr.File(
                            label="Upload Documents",
                            file_count="multiple",
                            file_types=[".txt", ".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".md", ".csv", ".json", ".xml", "outlook.msg", ".zip"],
                        )
                        
                        with gr.Row():
                            process_docs_btn = gr.Button("Process Documents", variant="primary", size="sm")
                        
                        doc_status = gr.Textbox(
                            label="Processing Status",
                            value="No documents to process",
                            interactive=False,
                            max_lines=10,
                        )

                        with gr.Row():
                            rag_max_nb_results = gr.Number(
                                label="Number of Results to return",
                                value=int(RAG_DEFAULT_MAX_NB_RESULTS),
                                interactive=True,
                            )
                    
                    with gr.Tab("Vector Store"):
                        with gr.Row():
                            vector_db_name = gr.Textbox(
                                label="Create New Local Vector DB",
                                value="work_db",
                                interactive=True,
                            )

                        rag_create_btn = gr.Button("Create", variant="primary", size="sm")

                        with gr.Row():
                            list_vector_db_names = gr.Dropdown(
                                label="List of Local Vector DBs",
                                allow_custom_value=True,
                            )

                        rag_delete_btn = gr.Button("Delete", variant="secondary", size="sm")

                        with gr.Row():
                            embedding_provider = gr.Dropdown(
                                label="Embedding Provider (apply to set)",
                                choices=available_rag_providers,
                                value=LLM_DEFAULT_EMBEDDING_PROVIDER,
                                interactive=True,
                            )

                            embedding_api_key = gr.Textbox(
                                label="Embedding API KEY",
                                type="password",
                                value="",
                                interactive=True,
                            )

                        with gr.Row():
                            embedding_model = gr.Dropdown(
                                label="Embedding Model",
                                value=LLM_DEFAULT_EMBEDDING_MODEL,
                                allow_custom_value=True,
                            )

                            embedding_base_url = gr.Textbox(
                                label="Embedding Base URL (only for Ollama)",
                                value=LLM_DEFAULT_URL,
                                interactive=True,
                            )
                        
                        with gr.Row():
                            chunk_size = gr.Slider(
                                label="Chunk Size",
                                minimum=100,
                                maximum=2000,
                                value=200,
                                step=100
                            )
                            chunk_overlap = gr.Slider(
                                label="Chunk Overlap",
                                minimum=0,
                                maximum=500,
                                value=150,
                                step=50
                            )

                            batch_size = gr.Slider(
                                label="Batch Size",
                                minimum=100,
                                maximum=8192,
                                value=512,
                                step=1
                            )

                        rag_save_btn = gr.Button("Apply", variant="primary", size="sm")
                        
                        vector_store_info = gr.Textbox(
                            label="Vector Store Info",
                            value="No documents in vector store",
                            interactive=False,
                            max_lines=15,
                        )
            
            # Right column for chat interface
            with gr.Column(scale=3, elem_classes=["chat-with-metrics-below"]):
                stream_metrics = gr.Markdown(
                    value="**0.00**s · **0** tokens · **0** tok/s",
                    label="Token metrics",
                    show_label=False,
                    elem_classes=["token-metrics-centered"],
                )
                gr.ChatInterface(
                    fn=chat,
                    title="🤖 AIMindRag Assistant",
                    additional_inputs=[llm_provider, mcp_clients, use_tools, use_rag, embedding_provider, rag_max_nb_results, llm_use_thinking],
                    additional_outputs=[stream_metrics],
                    analytics_enabled=str_to_bool(GRADIO_ANALYTICS_ENABLED),
                    save_history=True,
                    api_visibility="private",
                    chatbot=gr.Chatbot(
                        min_height=850,
                        max_height=850,
                        autoscroll=True,
                        buttons=["copy"],
                        reasoning_tags=[('<think>', '</think>')],
                        allow_tags=["think"],
                    )
                )
                toggle_settings_btn = gr.Button(
                    "Show settings panel",
                    size="sm",
                    variant="secondary",
                )

        async def init_ui(mcp_clients, mcp_tools, llm_provider, embedding_provider, llm_reasoning_effort, llm_base_url, embedding_base_url):
            """Initialize the UI"""
            clean_caches()
            llm_provider, embedding_provider, use_tools, use_rag = handle_app_config_load()
            mcp_clients, mcp_tools, clients_table, tools_table = await handle_mcp_load(mcp_clients, mcp_tools, use_tools)
            vector_db_names = get_vector_store_list()
            available_llm_models, available_embedding_models = get_available_llm_models()
            llm_model, gr_llm_provider, gr_llm_models, gr_llm_api_key, gr_llm_reasoning_effort, gr_llm_base_url = handle_llm_load_config(
                llm_provider, None, available_llm_models, llm_reasoning_effort, llm_base_url, notify=False)
            embedding_model, gr_embedding_provider, gr_embedding_model, gr_chunk_size, gr_chunk_overlap, gr_batch_size, gr_embedding_api_key, vector_store_info, gr_embedding_base_url = handle_rag_load_config(
                embedding_provider, None, None, None, None, None, available_embedding_models, embedding_base_url, notify=False)
            gr_embedding_models = update_embedding_models_list(embedding_provider, embedding_model, available_embedding_models)
            return use_tools, use_rag, mcp_clients, mcp_tools, clients_table, tools_table, gr_llm_provider, gr_llm_models, gr_llm_api_key, gr_llm_reasoning_effort, gr_llm_base_url, gr_embedding_provider, gr_embedding_models, gr_chunk_size, gr_chunk_overlap, gr_batch_size, \
                gr_embedding_api_key, gr_embedding_base_url, update_vector_db_names(vector_db_names), vector_store_info, available_llm_models, available_embedding_models

        def get_available_llm_models():
            """Get the available LLM models"""
            available_llm_models = {}
            available_embedding_models = {}
            for llm_provider in available_llm_providers:
                try:
                    llm_models = LLMClient(llm_provider).get_available_models() or []
                    llm_models_without_embedding = [model for model in llm_models if "embed" not in model.lower()]

                    if llm_provider in available_rag_providers:
                        embedding_models = [m for m in llm_models if "embed" in m.lower()]
                        if not embedding_models:
                            gr.Warning(f"No embedding models found for provider: {llm_provider}")
                    else:
                        embedding_models = []

                    if not llm_models:
                        gr.Warning(f"No LLM models found for provider: {llm_provider}")

                    available_llm_models[llm_provider] = llm_models_without_embedding
                    available_embedding_models[llm_provider] = embedding_models
                except Exception as e:
                    gr.Warning(f"Error getting available models for provider: {llm_provider}: {str(e)}")
                    available_llm_models[llm_provider] = []
                    available_embedding_models[llm_provider] = []
            return available_llm_models, available_embedding_models

        def toggle_base_url(provider: str):
            """Toggle the base URL"""
            if provider == "ollama":
                return gr.update(visible=True)
            else:
                return gr.update(visible=False)

        def toggle_api_key(provider: str):
            """Toggle the API key"""
            if provider == "ollama":
                return gr.update(visible=False)
            else:
                return gr.update(visible=True)

        def update_llm_models_list(llm_provider: str, llm_model: str, available_llm_models: dict):
            """Update the LLM models list"""
            llm_models_list = available_llm_models.get(llm_provider, [])
            if len(llm_models_list) == 0:
                llm_model = "<NO LLM MODELS FOUND>"
            return gr.update(choices=llm_models_list, interactive=True, value=llm_model)
        
        def update_embedding_models_list(embedding_provider: str, embedding_model: str, available_embedding_models: dict):
            """Update the embedding models list"""
            embedding_models_list = available_embedding_models.get(embedding_provider, [])
            if len(embedding_models_list) == 0:
                embedding_model = "<NO EMBEDDING MODELS FOUND>"
            return gr.update(choices=embedding_models_list, interactive=True, value=embedding_model)

        def update_vector_store_info(embedding_provider: str):
            """Update the vector store info"""
            rag_client = RAGClient(embedding_provider)
            vector_db_names = get_vector_store_list()
            if rag_client.db_client_collection is None or len(vector_db_names) == 0:
                vector_store_info = "No vector store found"
                gr.Warning("No vector databases found")
            elif rag_client.config.db_name in vector_db_names:
                nb_records = rag_client.get_nb_records()
                if nb_records == 0:
                    vector_store_info = f"No documents in '{rag_client.config.db_name}' vector store"
                else:
                    vector_store_info = f"Number of records in '{rag_client.config.db_name}' vector store: {nb_records}"
            else:
                vector_store_info = f"Vector store '{rag_client.config.db_name}' not found"
            return vector_store_info
        
        def update_vector_db_names(vector_db_names: list):
            """Update the vector store names list"""
            if len(vector_db_names) == 0:
                default_value = RAG_DEFAULT_DB_NAME
                return gr.update(choices=vector_db_names, interactive=True, value=default_value)
            else:
                return gr.update(choices=vector_db_names, interactive=True)

        def update_clients_table(mcp_clients: dict):
            """Update the clients table"""
            return [[name, info.get("status", "Disconnected")] for name, info in mcp_clients.items()]

        def update_tools_table(mcp_tools: dict):
            """Update the tools table"""
            rows = []
            if len(mcp_tools) != 0:
                for _, tools_discovered in mcp_tools.items():
                    for tool in tools_discovered["tools"]:
                        tool_name = tool["function"]["name"] or "No name"
                        tool_description = tool["function"]["description"] or "No description"
                        rows.append([tool_name, tool_description])
            return rows

        def handle_app_config_load():
            """Handle the app config load"""
            llm_provider = LLM_DEFAULT_PROVIDER
            embedding_provider = LLM_DEFAULT_EMBEDDING_PROVIDER
            use_tools = False
            use_rag = False
            loaded_config = handle_app_config(llm_provider, embedding_provider, use_tools, use_rag, "load")
            if not loaded_config["success"]:
                gr.Warning(loaded_config["message"])
            else:
                llm_provider = loaded_config["data"]["llm_provider"]
                embedding_provider = loaded_config["data"]["embedding_provider"]
                use_tools = loaded_config["data"]["use_tools"]
                use_rag = loaded_config["data"]["use_rag"]
                gr.Info(loaded_config["message"])
            return llm_provider, embedding_provider, use_tools, use_rag

        def handle_app_config_save(llm_provider: str, embedding_provider: str, use_tools: bool, use_rag: bool):
            """Handle the app config save"""
            saved_config = handle_app_config(llm_provider, embedding_provider, use_tools, use_rag, "save")
            if not saved_config["success"]:
                gr.Warning(saved_config["message"])
            return llm_provider, embedding_provider, use_tools, use_rag

        def handle_mcp_save_config(mcp_clients: dict) -> None:
            """Handle the MCP save config"""
            rows = []
            for url, info in mcp_clients.items():
                client = info.get("mcp_client")
                transport = getattr(client, "transport_type", None) or "http"
                rows.append({"url": url, "type": transport})
            result = handle_mcp_config(rows, "save")
            if not result["success"]:
                gr.Warning(result["message"])

        async def handle_mcp_load(mcp_clients: dict, mcp_tools: dict, use_tools: bool):
            """Handle the MCP load on startup"""
            if not use_tools:
                return mcp_clients, mcp_tools, update_clients_table(mcp_clients), update_tools_table(mcp_tools)
            else:
                loaded_config = handle_mcp_config(None, "load")
                if not loaded_config["success"]:
                    gr.Warning(loaded_config["message"])
                else:
                    mcp_clients_list = loaded_config["data"]["mcp_clients"]
                    for mcp_client in mcp_clients_list:
                        await handle_connect(mcp_client["url"], mcp_client["type"], mcp_clients, mcp_tools)
                gr.Info(loaded_config["message"])
            return mcp_clients, mcp_tools, update_clients_table(mcp_clients), update_tools_table(mcp_tools)
        
        async def handle_connect(client_path: str, transport_type: str, mcp_clients: dict, mcp_tools: dict):
            """Handle the MCP connect"""
            ok = False
            if not client_path:
                gr.Warning("Please provide an MCP client path.")
            else:
                client = MCPClient(server_path=client_path, transport_type=transport_type)
                ok = await client.connect()
            if not ok:
                gr.Warning("Error connecting to MCP server")
            else:
                mcp_clients.update({client_path: {"mcp_client": client, "status": "Connected"}})
                discovered_tools = await client.get_tools() or []
                mcp_tools.update({client_path: {"tools": discovered_tools}})
                gr.Info(f"Connected to MCP server: {client_path}")
                handle_mcp_save_config(mcp_clients)
            return mcp_clients, mcp_tools, update_clients_table(mcp_clients), update_tools_table(mcp_tools)
        
        async def handle_disconnect(selected_client_name: str, mcp_clients: dict, mcp_tools: dict):
            """Handle the MCP disconnect"""
            if not selected_client_name:
                gr.Warning("Please select a client from the table first.")
            else:
                if selected_client_name in mcp_clients:
                    client = mcp_clients[selected_client_name]["mcp_client"]
                    await client.disconnect()
                    del mcp_clients[selected_client_name]
                    del mcp_tools[selected_client_name]
                    gr.Info(f"Disconnected from MCP server: {selected_client_name}")
                    handle_mcp_save_config(mcp_clients)
                else:
                    raise gr.Error("Client not found", print_exception=False)
            return update_clients_table(mcp_clients), update_tools_table(mcp_tools), None

        def handle_client_select(evt: gr.SelectData, table_data):
            """Handle the MCP client select"""
            if evt and evt.index is not None:
                selected_client = str(table_data.iloc[evt.index[0], 0])
            return selected_client
        
        def handle_llm_save_config(llm_provider: str, llm_model: str, llm_api_key: str, llm_reasoning_effort: str, llm_base_url: str):
            """Handle the LLM save config"""
            saved_config = handle_llm_config(llm_provider, llm_model, llm_api_key, "save", llm_reasoning_effort, llm_base_url)
            if not saved_config["success"]:
                gr.Warning(saved_config["message"])
            else:
                gr.Info(saved_config["message"])
                handle_app_config_save(llm_provider, None, None, None)
            return llm_provider, gr.update(value=llm_model)
        
        def handle_llm_load_config(llm_provider: str, llm_model: str, available_llm_models: dict, llm_reasoning_effort: str, llm_base_url: str, *, notify: bool = True):
            """Handle the LLM load config"""
            reasoning_effort = llm_reasoning_effort
            base_url = llm_base_url
            loaded_config = handle_llm_config(llm_provider, None, None, "load", llm_reasoning_effort, llm_base_url)
            if not loaded_config["success"]:
                gr.Warning(loaded_config["message"])
            else:
                llm_provider = loaded_config["data"]["provider"]
                llm_model = loaded_config["data"]["model"]
                reasoning_effort = loaded_config["data"]["reasoning_effort"]
                base_url = loaded_config["data"]["base_url"]
                if notify:
                    gr.Info(loaded_config["message"])
            gr_llm_model = update_llm_models_list(llm_provider, llm_model, available_llm_models)
            return (
                llm_model,
                gr.update(value=llm_provider),
                gr_llm_model,
                toggle_api_key(llm_provider),
                gr.update(value=reasoning_effort),
                gr.update(value=base_url, visible=(llm_provider == "ollama")),
            )
        
        def handle_rag_create(embedding_provider: str, vector_name: str):
            """Handle the RAG create"""
            created_vector = handle_vector_db(vector_name, "create")
            vector_db_names = get_vector_store_list()
            vector_store_info = update_vector_store_info(embedding_provider)
            if not created_vector["success"]:
                gr.Warning(created_vector["message"])
            else:
                gr.Info(created_vector["message"])
            return update_vector_db_names(vector_db_names), vector_store_info
        
        def handle_rag_delete(vector_name: str, embedding_provider: str):   
            """Handle the RAG delete"""
            if vector_name is None:
                vector_name = RAG_DEFAULT_DB_NAME
            deleted_vector = handle_vector_db(vector_name, "delete")
            vector_db_names = get_vector_store_list()
            vector_store_info = update_vector_store_info(embedding_provider)
            if not deleted_vector["success"]:
                gr.Warning(deleted_vector["message"])
            else:
                gr.Info(deleted_vector["message"])
            return update_vector_db_names(vector_db_names), vector_store_info

        def handle_rag_save_config(embedding_provider: str, embedding_model: str, embedding_api_key: str, vector_name: str, chunk_size: int, chunk_overlap: int, batch_size: int, embedding_base_url: str):
            """Handle the RAG save config"""
            saved_config = handle_rag_config(embedding_provider, embedding_model, embedding_api_key, vector_name, chunk_size, chunk_overlap, batch_size, "save", embedding_base_url)
            vector_store_info = update_vector_store_info(embedding_provider)
            if not saved_config["success"]:
                gr.Warning(saved_config["message"])
            else:
                gr.Info(saved_config["message"])
                handle_app_config_save(None, embedding_provider, None, None)
            return embedding_provider, gr.update(value=embedding_model), gr.update(value=chunk_size), gr.update(value=chunk_overlap), gr.update(value=int(batch_size)), toggle_api_key(embedding_provider), toggle_base_url(embedding_provider), vector_store_info
        
        def handle_rag_load_config(embedding_provider: str, embedding_model: str, vector_name: str, chunk_size: int, chunk_overlap: int, batch_size: int, available_embedding_models: dict, embedding_base_url: str, *, notify: bool = True):
            """Handle the RAG load config"""
            base_url = embedding_base_url
            loaded_config = handle_rag_config(embedding_provider, None, None, None, None, None, None, "load", embedding_base_url)
            vector_store_info = update_vector_store_info(embedding_provider)
            if not loaded_config["success"]:
                gr.Warning(loaded_config["message"])
            else:
                embedding_provider = loaded_config["data"]["provider"]
                embedding_model = loaded_config["data"]["embedding_model"]
                chunk_size = loaded_config["data"]["chunk_size"]
                chunk_overlap = loaded_config["data"]["chunk_overlap"]
                batch_size = int(loaded_config["data"]["batch_size"])
                base_url = loaded_config["data"]["base_url"]
                if notify:
                    gr.Info(loaded_config["message"])
            gr_embedding_model = update_embedding_models_list(embedding_provider, embedding_model, available_embedding_models)
            return embedding_model, gr.update(value=embedding_provider), gr_embedding_model, gr.update(value=chunk_size), gr.update(value=chunk_overlap), gr.update(value=batch_size), toggle_api_key(embedding_provider), vector_store_info, gr.update(value=base_url, visible=(embedding_provider == "ollama"))

        def handle_process_docs(embedding_provider: str, file_upload: list, ):
            """Handle the process docs for RAG"""
            file_upload_status = gr.update(value=None)
            vector_db_names = get_vector_store_list()
            if len(vector_db_names) == 0:
                gr.Warning("No vector store found")
                yield "No vector store found", None, file_upload_status, None
            elif not file_upload:
                gr.Warning("No documents to process")
                yield "No documents to process", None, file_upload_status, None
            else:
                rag_client = RAGClient(embedding_provider)
                if rag_client.db_client_collection is None:
                    yield "No vector store found", None, file_upload_status, None
                else:
                    for i, document in enumerate(file_upload, start=1):
                        document_name = Path(document).name.strip()
                        yield f"Currently processing: {document_name} ({i}/{len(file_upload)})", None, file_upload_status, None
                        rag_client.store_document(document, str_to_bool(RAG_MARKDOWN))
                vector_store_info = update_vector_store_info(embedding_provider)
                yield "Documents processed successfully", vector_store_info, file_upload_status, update_vector_db_names(vector_db_names)

        def toggle_settings_sidebar(hidden: bool):
            new_hidden = not hidden
            button_label = "Show settings panel" if new_hidden else "Hide settings panel"
            settings_column_visibility = gr.update(visible=not new_hidden)
            return (
                new_hidden,
                settings_column_visibility,
                button_label,
            )
        
        def clean_caches():
            """Clean the gradio caches"""
            success = delete_dir(GRADIO_TEMP_FOLDER_PATH, remove_root=False)
            if success:
                gr.Info("Caches cleaned successfully")
            else:
                gr.Warning("Failed to clean caches")

        mcp_connect_btn.click(
            fn=handle_connect,
            inputs=[mcp_client_path, mcp_client_transport_type, mcp_clients, mcp_tools],
            outputs=[mcp_clients, mcp_tools, clients_table, tools_table],
            api_visibility="private"
        )
        
        mcp_disconnect_btn.click(
            fn=handle_disconnect,
            inputs=[selected_mcp_client, mcp_clients, mcp_tools],
            outputs=[clients_table, tools_table, selected_mcp_client],
            api_visibility="private"
        )

        clients_table.select(
            fn=handle_client_select,
            inputs=[clients_table],
            outputs=[selected_mcp_client],
            api_visibility="private"
        )

        llm_save_btn.click(
            fn=handle_llm_save_config,
            inputs=[llm_provider, llm_model, llm_api_key, llm_reasoning_effort, llm_base_url],
            outputs=[llm_provider, llm_model],
            api_visibility="private"
        )

        llm_provider.change(
            fn=handle_llm_load_config,
            inputs=[llm_provider, llm_model, available_llm_models, llm_reasoning_effort, llm_base_url],
            outputs=[loaded_llm_model, llm_provider, llm_model, llm_api_key, llm_reasoning_effort, llm_base_url],
            api_visibility="private"
        )

        rag_create_btn.click(
            fn=handle_rag_create,
            inputs=[embedding_provider, vector_db_name],
            outputs=[list_vector_db_names, vector_store_info],
            api_visibility="private"
        )

        rag_delete_btn.click(
            fn=handle_rag_delete,
            inputs=[list_vector_db_names, embedding_provider],
            outputs=[list_vector_db_names, vector_store_info],
            api_visibility="private"
        )

        rag_save_btn.click(
            fn=handle_rag_save_config,
            inputs=[embedding_provider, embedding_model, embedding_api_key, list_vector_db_names, chunk_size, chunk_overlap, batch_size, embedding_base_url],
            outputs=[embedding_provider, embedding_model, chunk_size, chunk_overlap, batch_size, embedding_api_key, embedding_base_url, vector_store_info],
            api_visibility="private"
        )

        embedding_provider.change(
            fn=handle_rag_load_config,
            inputs=[embedding_provider, embedding_model, list_vector_db_names, chunk_size, chunk_overlap, batch_size, available_embedding_models, embedding_base_url],
            outputs=[loaded_embedding_model, embedding_provider, embedding_model, chunk_size, chunk_overlap, batch_size, embedding_api_key, vector_store_info, embedding_base_url],
            api_visibility="private"
        )

        process_docs_btn.click(
            fn=handle_process_docs,
            inputs=[embedding_provider, file_upload],
            outputs=[doc_status, vector_store_info, file_upload, list_vector_db_names],
            api_visibility="private"
        )

        toggle_settings_btn.click(
            fn=toggle_settings_sidebar,
            inputs=[settings_sidebar_hidden],
            outputs=[settings_sidebar_hidden, settings_column, toggle_settings_btn],
            api_visibility="private",
        )

        use_tools.change(
            fn=handle_app_config_save, 
            inputs=[llm_provider, embedding_provider, use_tools, use_rag], 
            outputs=[llm_provider, embedding_provider, use_tools, use_rag],
            api_visibility="private"
        )

        use_rag.change(
            fn=handle_app_config_save, 
            inputs=[llm_provider, embedding_provider, use_tools, use_rag], 
            outputs=[llm_provider, embedding_provider, use_tools, use_rag],
            api_visibility="private"
        )

        ui.load(
            fn=init_ui,
            inputs=[mcp_clients, mcp_tools, llm_provider, embedding_provider, llm_reasoning_effort, llm_base_url, embedding_base_url],
            outputs=[use_tools, use_rag, mcp_clients, mcp_tools, clients_table, tools_table, llm_provider, llm_model, llm_api_key, llm_reasoning_effort, llm_base_url, embedding_provider, embedding_model, chunk_size, chunk_overlap, batch_size, embedding_api_key, embedding_base_url, list_vector_db_names, vector_store_info, available_llm_models, available_embedding_models],
            api_visibility="private"
        )

    try:
        ui.queue()
        ui.launch(
            css=css,
            server_name=GRADIO_BIND_ADDRESS,
            server_port=int(GRADIO_PORT),
            share=str_to_bool(GRADIO_SHARE),
            pwa=str_to_bool(GRADIO_PWA),
            footer_links=["gradio", "settings"]
        )
        ui_logger.info("UI successfully started")
    except Exception as e:
        ui_logger.error(f"Error launching the Gradio UI: {e}")
        raise e

import os
import gradio as gr
from utils.helpers import str_to_bool, delete_dir, create_dir
from utils.config import GRADIO_DEBUG, GRADIO_BIND_ADDRESS, GRADIO_PORT, GRADIO_TEMP_FOLDER_PATH, GRADIO_SHARE, GRADIO_PWA, GRADIO_ANALYTICS_ENABLED, RAG_DB_PATH, RAG_MARKDOWN, RAG_DEFAULT_DB_NAME, \
    LLM_DEFAULT_PROVIDER, LLM_DEFAULT_MODEL, LLM_DEFAULT_EMBEDDING_PROVIDER, LLM_DEFAULT_EMBEDDING_MODEL, RAG_DEFAULT_MAX_NB_RESULTS
from utils.load import available_llm_providers
from modules.chat import chat
from modules.mcp import MCPClient
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

def run_ui():
    """Run the Gradio UI"""
    with gr.Blocks(title="MCP Client") as ui:
        loaded_llm_model = gr.State("")
        loaded_embedding_model = gr.State("")
        selected_mcp_client = gr.State("")
        use_tools = gr.State(True)
        mcp_clients = gr.State({})
        mcp_tools = gr.State({})
        available_llm_models = gr.State({})
        available_embedding_models = gr.State({})

        with gr.Row():
            with gr.Column(scale=2, min_width=400):
                gr.HTML("<div style='height: 5px;'></div>")

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
                            value="<IF ALREADY CONFIGURED LOADED FROM CONFIG FILE>",
                            interactive=True,
                            max_lines=1
                        )
                        
                    llm_save_btn = gr.Button("Apply", variant="primary", size="sm")
                        
                # MCP Server Management Block
                with gr.Group():
                    gr.Markdown("##  MCP Server Management")

                    use_tools = gr.Checkbox(
                        label="Use MCP Tools",
                        value=True,
                        interactive=True,
                    )
                    
                    with gr.Row():
                        mcp_client_path = gr.Textbox(
                            label="MCP Client Path",
                            value="http://127.0.0.1:8888/mcp",
                            interactive=True,
                            max_lines=1
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
                            max_height=100
                        )
                    
                    with gr.Row():
                        tools_table = gr.Dataframe(
                            headers=["Tool name", "Description"],
                            interactive=False,
                            show_search=True,
                            datatype="array",
                            label="MCP Tools",
                            max_height=200
                        )

                # RAG Management Block
                with gr.Group():
                    gr.Markdown("## 📚 RAG Management")

                    use_rag = gr.Checkbox(
                        label="Use RAG",
                        value=True,
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
                            max_lines=10
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
                                max_lines=1
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
                                label="Embedding Provider",
                                choices=available_llm_providers,
                                value=LLM_DEFAULT_EMBEDDING_PROVIDER,
                                interactive=True,
                            )

                            embedding_api_key = gr.Textbox(
                                label="Embedding API KEY",
                                type="password",
                                value="<IF ALREADY CONFIGURED LOADED FROM CONFIG FILE>",
                                interactive=True,
                                max_lines=1
                            )

                        with gr.Row():
                            embedding_model = gr.Dropdown(
                                label="Embedding Model",
                                value=LLM_DEFAULT_EMBEDDING_MODEL,
                                allow_custom_value=True,
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

                        rag_save_btn = gr.Button("Apply", variant="primary", size="sm")
                        
                        vector_store_info = gr.Textbox(
                            label="Vector Store Info",
                            value="No documents in vector store",
                            interactive=False,
                            max_lines=15
                        )
            
            # Right column for chat interface
            with gr.Column(scale=3):
                gr.ChatInterface(
                    type="messages", 
                    fn=chat,
                    title="🤖 AIMindRag Assistant",
                    additional_inputs=[llm_provider, mcp_clients, use_tools, use_rag, embedding_provider, rag_max_nb_results],
                    analytics_enabled=str_to_bool(GRADIO_ANALYTICS_ENABLED),
                    description="Chat with your AI assistant enhanced with MCP tools and RAG knowledge",
                    chatbot=gr.Chatbot(
                        type="messages",
                        min_height=850,
                        max_height=850,
                        autoscroll=True,
                        show_copy_button=True,
                    )
                )

        def init_ui(llm_provider, embedding_provider):
            """Initialize the UI"""
            clean_caches()
            vector_db_names = get_vector_store_list()
            available_llm_models, available_embedding_models = get_available_llm_models()
            llm_model, gr_llm_provider, gr_llm_model, gr_llm_api_key = handle_llm_load_config(llm_provider, None, available_llm_models)
            embedding_model, gr_embedding_provider, gr_embedding_model, gr_chunk_size, gr_chunk_overlap, gr_embedding_api_key, vector_store_info = handle_rag_load_config(embedding_provider, None, None, None, None, available_embedding_models)
            gr_llm_models = update_llm_models_list(llm_provider, llm_model, available_llm_models)
            gr_embedding_models = update_embedding_models_list(embedding_provider, embedding_model, available_embedding_models)
            return gr_llm_provider, gr_llm_models, gr_llm_api_key, gr_embedding_provider, gr_embedding_models, gr_chunk_size, gr_chunk_overlap, \
                gr_embedding_api_key, update_vector_db_names(vector_db_names), vector_store_info, available_llm_models, available_embedding_models

        def get_available_llm_models():
            """Get the available LLM models"""
            available_llm_models = {}
            available_embedding_models = {}
            for llm_provider in available_llm_providers:
                try:
                    llm_models = LLMClient(llm_provider).get_available_models() or []
                    embedding_models = [model for model in llm_models if "embed" in model.lower()]
                    llm_models_without_embedding = [model for model in llm_models if "embed" not in model.lower()]

                    if not llm_models:
                        gr.Warning(f"No LLM models found for provider: {llm_provider}")
                    if not embedding_models:
                        gr.Warning(f"No Embedding models found for provider: {llm_provider}")

                    available_llm_models[llm_provider] = llm_models_without_embedding
                    available_embedding_models[llm_provider] = embedding_models
                except Exception as e:
                    gr.Warning(f"Error getting available models for provider: {llm_provider}: {str(e)}")
                    available_llm_models[llm_provider] = []
                    available_embedding_models[llm_provider] = []
            return available_llm_models, available_embedding_models

        def toggle_api_key(provider: str):
            """Toggle the API key"""
            if provider == "ollama":
                return gr.update(interactive=False, value="")
            else:
                return gr.update(interactive=True, value="<IF ALREADY CONFIGURED LOADED FROM CONFIG FILE>")

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
                gr.Info("Connected to MCP server")
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
                    gr.Info("Disconnected from MCP server")
                else:
                    raise gr.Error("Client not found", print_exception=False)
            return update_clients_table(mcp_clients), update_tools_table(mcp_tools), None

        def handle_client_select(evt: gr.SelectData, table_data):
            """Handle the MCP client select"""
            if evt and evt.index is not None:
                selected_client = str(table_data.iloc[evt.index[0], 0])
            return selected_client
        
        def handle_llm_save_config(llm_provider: str, llm_model: str, llm_api_key: str):
            """Handle the LLM save config"""
            saved_config = handle_llm_config(llm_provider, llm_model, llm_api_key, "save")
            if not saved_config["success"]:
                gr.Warning(saved_config["message"])
            else:
                gr.Info(saved_config["message"])
            return llm_provider, gr.update(value=llm_model)
        
        def handle_llm_load_config(llm_provider: str, llm_model: str, available_llm_models: dict):
            """Handle the LLM load config"""
            loaded_config = handle_llm_config(llm_provider, None, None, "load")
            if not loaded_config["success"]:
                gr.Warning(loaded_config["message"])
            else:
                llm_provider = loaded_config["data"]["provider"]
                llm_model = loaded_config["data"]["model"]
                gr.Info(loaded_config["message"])
            gr_llm_model = update_llm_models_list(llm_provider, llm_model, available_llm_models)
            return llm_model, gr.update(value=llm_provider), gr_llm_model, toggle_api_key(llm_provider)
        
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

        def handle_rag_save_config(embedding_provider: str, embedding_model: str, embedding_api_key: str, vector_name: str, chunk_size: int, chunk_overlap: int):
            """Handle the RAG save config"""
            saved_config = handle_rag_config(embedding_provider, embedding_model, embedding_api_key, vector_name, chunk_size, chunk_overlap, "save")
            vector_store_info = update_vector_store_info(embedding_provider)
            if not saved_config["success"]:
                gr.Warning(saved_config["message"])
            else:
                gr.Info(saved_config["message"])
            return embedding_provider, gr.update(value=embedding_model), gr.update(value=chunk_size), gr.update(value=chunk_overlap), toggle_api_key(embedding_provider), vector_store_info
        
        def handle_rag_load_config(embedding_provider: str, embedding_model: str, vector_name: str, chunk_size: int, chunk_overlap: int, available_embedding_models: dict):
            """Handle the RAG load config"""
            loaded_config = handle_rag_config(embedding_provider, None, None, None, None, None, "load")
            vector_store_info = update_vector_store_info(embedding_provider)
            if not loaded_config["success"]:
                gr.Warning(loaded_config["message"])
            else:
                embedding_provider = loaded_config["data"]["provider"]
                embedding_model = loaded_config["data"]["embedding_model"]
                chunk_size = loaded_config["data"]["chunk_size"]
                chunk_overlap = loaded_config["data"]["chunk_overlap"]
                gr.Info(loaded_config["message"])
            gr_embedding_model = update_embedding_models_list(embedding_provider, embedding_model, available_embedding_models)
            return embedding_model, gr.update(value=embedding_provider), gr_embedding_model, gr.update(value=chunk_size), gr.update(value=chunk_overlap), toggle_api_key(embedding_provider), vector_store_info

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
        )
        
        mcp_disconnect_btn.click(
            fn=handle_disconnect,
            inputs=[selected_mcp_client, mcp_clients, mcp_tools],
            outputs=[clients_table, tools_table, selected_mcp_client],
        )

        clients_table.select(
            fn=handle_client_select,
            inputs=[clients_table],
            outputs=[selected_mcp_client],
        )

        llm_save_btn.click(
            fn=handle_llm_save_config,
            inputs=[llm_provider, llm_model, llm_api_key],
            outputs=[llm_provider, llm_model],
        )

        llm_provider.change(
            fn=handle_llm_load_config,
            inputs=[llm_provider, llm_model, available_llm_models],
            outputs=[loaded_llm_model, llm_provider, llm_model, llm_api_key],
        )

        rag_create_btn.click(
            fn=handle_rag_create,
            inputs=[embedding_provider, vector_db_name],
            outputs=[list_vector_db_names, vector_store_info],
        )

        rag_delete_btn.click(
            fn=handle_rag_delete,
            inputs=[list_vector_db_names, embedding_provider],
            outputs=[list_vector_db_names, vector_store_info],
        )

        rag_save_btn.click(
            fn=handle_rag_save_config,
            inputs=[embedding_provider, embedding_model, embedding_api_key, list_vector_db_names, chunk_size, chunk_overlap],
            outputs=[embedding_provider, embedding_model, chunk_size, chunk_overlap, embedding_api_key, vector_store_info],
        )

        embedding_provider.change(
            fn=handle_rag_load_config,
            inputs=[embedding_provider, embedding_model, list_vector_db_names, chunk_size, chunk_overlap, available_embedding_models],
            outputs=[loaded_embedding_model, embedding_provider, embedding_model, chunk_size, chunk_overlap, embedding_api_key, vector_store_info],
        )

        process_docs_btn.click(
            fn=handle_process_docs,
            inputs=[embedding_provider, file_upload],
            outputs=[doc_status, vector_store_info, file_upload, list_vector_db_names],
        )

        ui.load(
            fn=init_ui,
            inputs=[llm_provider, embedding_provider],
            outputs=[llm_provider, llm_model, llm_api_key, embedding_provider, embedding_model, chunk_size, chunk_overlap, embedding_api_key, list_vector_db_names, vector_store_info, available_llm_models, available_embedding_models],
        )

    try:
        ui.launch(
            server_name=GRADIO_BIND_ADDRESS,
            server_port=int(GRADIO_PORT),
            share=str_to_bool(GRADIO_SHARE),
            pwa=str_to_bool(GRADIO_PWA)
        )
        ui_logger.info("UI successfully started")
    except Exception as e:
        ui_logger.error(f"Error launching the Gradio UI: {e}")
        raise e
from utils.config import APP_CONFIG_DIR_PATH, GRADIO_TEMP_FOLDER_PATH, LLM_OLLAMA_DIR_PATH, RAG_DB_PATH, RAG_CONFIG_DIR_PATH, LLM_DEFAULT_PROVIDER, LLM_DEFAULT_EMBEDDING_PROVIDER
from utils.helpers import create_dir, apply_config
from pathlib import Path

"""Load the available LLM providers"""

app_config = "app.json"

available_llm_providers = ["gpt", "claude", "ollama", "openai_compat"]

available_llm_config_providers = {
    "gpt": "llm_gpt.json",
    "claude": "llm_claude.json",
    "ollama": "llm_ollama.json",
    "openai_compat": "llm_openai_compat.json",
}

available_rag_providers = ["gpt", "claude", "ollama", "openai_compat"]

available_rag_config_providers = {
    "gpt": "rag_gpt.json",
    "claude": "rag_claude.json",
    "ollama": "rag_ollama.json",
    "openai_compat": "rag_openai_compat.json"
}

mcp_tools_config = "mcp_tools.json"

piillmshield_config = "piillmshield.json"

available_voyage_embedding_models = [
    "voyage-4-large",
    "voyage-4",
    "voyage-4-lite",
    "voyage-code-3", 
    "voyage-finance-2",
    "voyage-law-2",
    "voyage-code-2",
]

def init_load():
    """Create the necessary directories"""
    create_dir(LLM_OLLAMA_DIR_PATH)
    create_dir(GRADIO_TEMP_FOLDER_PATH)
    create_dir(RAG_CONFIG_DIR_PATH)
    create_dir(RAG_DB_PATH)

def handle_app_config(llm_provider: str, embedding_provider: str, use_tools: bool, use_rag: bool, use_piillmshield: bool, action: str):
    """Handle the app config"""
    success = False
    message = "Invalid action"
    data = {"llm_provider": LLM_DEFAULT_PROVIDER, "embedding_provider": LLM_DEFAULT_EMBEDDING_PROVIDER,
            "use_tools": False, "use_rag": False, "use_piillmshield": False}
    create_dir(APP_CONFIG_DIR_PATH)
    
    config_file_path = Path(APP_CONFIG_DIR_PATH) / app_config

    if not config_file_path.exists():
        action = "save"
    
    if action in ["save", "load"]:
        if action == "save":
            success, current_config = apply_config(config_file_path, "r")
            if not success:
                current_config = {}

            if use_tools is None:
                use_tools = current_config.get("use_tools") or False
            if use_rag is None:
                use_rag = current_config.get("use_rag") or False
            if use_piillmshield is None:
                use_piillmshield = current_config.get("use_piillmshield") or False

            config = {
                "llm_provider": llm_provider or current_config.get("llm_provider") or LLM_DEFAULT_PROVIDER,
                "embedding_provider": embedding_provider or current_config.get("embedding_provider") or LLM_DEFAULT_EMBEDDING_PROVIDER,
                "use_tools": use_tools,
                "use_rag": use_rag,
                "use_piillmshield": use_piillmshield,
            }

            success, _ = apply_config(config_file_path, "w", config)
            message = "App configuration saved successfully" if success else "Failed to save App configuration"
        elif action == "load":
            success, data = apply_config(config_file_path, "r")
            message = "App configuration loaded successfully" if success else "Failed to load App configuration"
    return {"success": success, "message": message, "data": data}

def handle_piillmshield_config(url: str, action: str) -> dict:
    """Handle PIILLMShield configuration"""
    create_dir(APP_CONFIG_DIR_PATH)
    config_file_path = Path(APP_CONFIG_DIR_PATH) / piillmshield_config
    success = False
    message = "Invalid action"
    data = {"url": url}
    if action == "save":
        config = {"url": url}
        success, _ = apply_config(config_file_path, "w", config)
        message = "PIILLMShield configuration saved" if success else "Failed to save PIILLMShield configuration"
    elif action == "load":
        success, data = apply_config(config_file_path, "r")
        if not success:
            data = {"url": ""}
        message = "PIILLMShield configuration loaded" if success else "Failed to load PIILLMShield configuration"
    return {"success": success, "message": message, "data": data}
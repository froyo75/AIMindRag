from utils.config import GRADIO_TEMP_FOLDER_PATH, LLM_OLLAMA_DIR_PATH, RAG_DB_PATH, RAG_CONFIG_DIR_PATH
from utils.helpers import create_dir

"""Load the available LLM providers"""

available_llm_providers = ["gpt", "ollama"]

available_llm_config_providers = {
    "gpt": "llm_gpt.json",
    "ollama": "llm_ollama.json"
}

available_rag_providers = ["gpt", "ollama"]

available_rag_config_providers = {
    "gpt": "rag_gpt.json",
    "ollama": "rag_ollama.json"
}


def init_load():
    """Create the necessary directories"""
    create_dir(LLM_OLLAMA_DIR_PATH)
    create_dir(GRADIO_TEMP_FOLDER_PATH)
    create_dir(RAG_CONFIG_DIR_PATH)
    create_dir(RAG_DB_PATH)
    
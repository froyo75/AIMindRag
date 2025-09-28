from dotenv import load_dotenv, find_dotenv
import os

# Load environment variables
load_dotenv(find_dotenv())

# LLM Configuration
LLM_CONFIG_DIR_PATH = os.getenv('LLM_CONFIG_DIR_PATH') or './config'
LLM_DEFAULT_PROVIDER = os.getenv('LLM_DEFAULT_PROVIDER') or 'gpt'
LLM_DEFAULT_EMBEDDING_PROVIDER = os.getenv('LLM_DEFAULT_EMBEDDING_PROVIDER') or 'ollama'
LLM_DEFAULT_MODEL = os.getenv('LLM_DEFAULT_MODEL') or 'gpt-5-nano'
LLM_DEFAULT_EMBEDDING_MODEL = os.getenv('LLM_DEFAULT_EMBEDDING_MODEL') or 'mxbai-embed-large'
LLM_SYSTEM_PROMPT = os.getenv('LLM_SYSTEM_PROMPT') or 'You are a helpful assistant.'
LLM_DEFAULT_MAX_TOKENS = os.getenv('LLM_DEFAULT_MAX_TOKENS') or "1000"
LLM_DEFAULT_TEMPERATURE = os.getenv('LLM_DEFAULT_TEMPERATURE') or "0.5"
LLM_DEFAULT_OLLAMA_URL = os.getenv('LLM_DEFAULT_OLLAMA_URL') or 'http://ollama:11434'
LLM_OLLAMA_DIR_PATH = os.getenv('LLM_OLLAMA_DIR_PATH') or './ollama'

# RAG Configuration
RAG_CONFIG_DIR_PATH = os.getenv('RAG_CONFIG_DIR_PATH') or './config'
RAG_DB_PATH = os.getenv('RAG_DB_PATH') or './db'
RAG_DEFAULT_COLLECTION_NAME = os.getenv('RAG_DEFAULT_COLLECTION_NAME') or 'rag'
RAG_DEFAULT_DB_NAME = os.getenv('RAG_DEFAULT_DB_NAME') or 'work_db'
RAG_DEFAULT_CHUNK_SIZE = os.getenv('RAG_DEFAULT_CHUNK_SIZE') or "200"
RAG_DEFAULT_CHUNK_OVERLAP = os.getenv('RAG_DEFAULT_CHUNK_OVERLAP') or "150"
RAG_DEFAULT_MAX_NB_RESULTS = os.getenv('RAG_DEFAULT_MAX_NB_RESULTS') or "10"
RAG_MARKDOWN = os.getenv('RAG_MARKDOWN') or 'true'

# Frontend Configuration
GRADIO_BIND_ADDRESS = os.getenv('GRADIO_BIND_ADDRESS') or '127.0.0.1'
GRADIO_PORT = os.getenv('GRADIO_PORT') or '9998'
GRADIO_SHARE = os.getenv('GRADIO_SHARE') or 'false'
GRADIO_TEMP_FOLDER_PATH = os.getenv('GRADIO_TEMP_FOLDER_PATH') or './caches'
GRADIO_ANALYTICS_ENABLED = os.getenv('GRADIO_ANALYTICS_ENABLED') or 'false'
GRADIO_PWA = os.getenv('GRADIO_PWA') or 'false'
GRADIO_DEBUG = os.getenv('GRADIO_DEBUG') or "1"

# Data Configuration
DATA_DIR = os.getenv('DATA_DIR') or './data'
DATA_FOLDER_PATH = os.getenv('DATA_FOLDER_PATH') or './data'

# Logging Configuration
LOG_DIR_PATH = os.getenv('LOG_DIR_PATH') or './logs'
LOG_LEVEL = os.getenv('LOG_LEVEL') or 'INFO'
LOG_FILENAME = os.getenv('LOG_FILENAME') or 'app.log'
from utils.config import LLM_DEFAULT_URL, LLM_DEFAULT_EMBEDDING_PROVIDER, LLM_DEFAULT_EMBEDDING_MODEL, RAG_CONFIG_DIR_PATH, RAG_DB_PATH, RAG_DEFAULT_COLLECTION_NAME, RAG_DEFAULT_DB_NAME, RAG_DEFAULT_MAX_NB_RESULTS, \
    RAG_DEFAULT_BATCH_SIZE, RAG_DEFAULT_CHUNK_SIZE, RAG_DEFAULT_CHUNK_OVERLAP
from utils.load import available_rag_config_providers
from utils.helpers import apply_config, create_dir, delete_dir, safe_path, delete_file, list_dirs
from openai import OpenAI
from ollama import Client
from dataclasses import dataclass
import chromadb
import uuid
from pathlib import Path
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from modules.file import convert_to_markdown, get_file_content
from utils.logger import get_logger

rag_logger = get_logger(__name__)

def get_vector_store_list() -> list:
    """Get the list of vector stores"""
    vector_db_list = list_dirs(RAG_DB_PATH)
    return vector_db_list

def handle_rag_config(provider: str, embedding_model: str, api_key: str, db_name: str, chunk_size: int, chunk_overlap: int, batch_size: int, action: str, base_url: str) -> dict:
    """Handle RAG configuration creation and deletion"""
    config_file = available_rag_config_providers.get(provider)
    success = False
    message = "Invalid action"
    data = {"provider": provider, "embedding_model": embedding_model, "db_name": db_name, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "batch_size": batch_size}
    create_dir(RAG_CONFIG_DIR_PATH)

    if not config_file:
        rag_logger.error(f"Unsupported provider: {provider}")
        raise ValueError(f"Unsupported provider: {provider}")

    config_file_path = Path(RAG_CONFIG_DIR_PATH) / config_file

    if not config_file_path.exists():
        action = "save"

    if action in ["save", "load"]:
        if action == "save":
            success, current_config = apply_config(config_file_path, "r")
            if not success:
                current_config = {}

            if provider == "ollama":
                url = base_url or current_config.get("base_url") or LLM_DEFAULT_URL
            else:
                url = ""

            config = {
                "db_name": db_name or current_config.get("db_name") or RAG_DEFAULT_DB_NAME,
                "embedding_model": embedding_model or current_config.get("embedding_model") or LLM_DEFAULT_EMBEDDING_MODEL,
                "api_key": api_key or current_config.get("api_key") or "",
                "provider": provider or current_config.get("provider") or LLM_DEFAULT_EMBEDDING_PROVIDER,
                "chunk_size": chunk_size or current_config.get("chunk_size") or RAG_DEFAULT_CHUNK_SIZE,
                "chunk_overlap": chunk_overlap or current_config.get("chunk_overlap") or RAG_DEFAULT_CHUNK_OVERLAP,
                "batch_size": batch_size or current_config.get("batch_size") or RAG_DEFAULT_BATCH_SIZE,
                "base_url": url,
            }

            success, _ = apply_config(config_file_path, "w", config)
            message = "RAG configuration saved successfully" if success else "Failed to save RAG configuration"
        elif action == "load":
            success, data = apply_config(config_file_path, "r")
            message = "RAG configuration loaded successfully" if success else "Failed to load RAG configuration (check if the configuration file exists or apply for new configuration)"
    return {"success": success, "message": message, "data": data}

def handle_vector_db(vector_name: str, action: str) -> dict:
    """Handle vector database creation and deletion"""
    success = False
    message = "Invalid action"
    db_client = None
    create_dir(RAG_DB_PATH)
    valid_path, safe_vector_name = safe_path(RAG_DB_PATH, vector_name)
    if not valid_path:
        message = "Invalid vector DB name"
    else:
        vector_db_path = Path(RAG_DB_PATH) / Path(safe_vector_name).name
        if action in ["create", "load"]:
            try:
                if vector_db_path.exists():
                    message = f"Vector DB '{vector_name}' already exists" if action == "create" else f"Vector DB '{vector_name}' loaded successfully"
                    success = False
                else:
                    message = f"Vector DB '{vector_name}' created successfully"
                    success = True
                db_client = chromadb.PersistentClient(path=vector_db_path, settings=chromadb.Settings(anonymized_telemetry=False))
            except Exception as e:
                success = False
                message = f"Failed to create vector DB '{vector_name}'"
                rag_logger.error(f"Failed to create vector DB '{vector_name}': {e}")
        elif action == "delete":
            success = delete_dir(vector_db_path, remove_root=True)
            message = f"Vector DB '{vector_name}' deleted successfully" if success else f"Failed to delete vector DB '{vector_name}'"
    return {"success": success, "message": message, "db_client": db_client}

def handle_chunk_documents(document_content: str, chunk_size: int, chunk_overlap: int, markdown: bool = False) -> list:
    if markdown:
        semantic_text_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        semantic_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
    return semantic_text_splitter.split_text(document_content)

@dataclass
class RAGConfig:
    provider: str = ""
    embedding_model: str = ""
    api_key: str = ""
    db_name: str = ""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    batch_size: int = 512
    base_url: str = ""
    
class RAGClient:
    def __init__(self, provider: str):
        self.provider = provider
        self.db_client_collection = None
        self.embedding_client = self._initialize_client()

    def _initialize_client(self):
        """Initialize the RAG client"""
        loaded_config = handle_rag_config(self.provider, None, None, None, None, None, None, "load", None)
        if not loaded_config["success"]:
            rag_logger.error(loaded_config["message"])
            raise ValueError(loaded_config["message"])
        self.config = RAGConfig(**loaded_config["data"])
        
        self.db_client_collection = self._init_vector_db()
        
        provider_init_methods = {
            "gpt": self._init_openai,
            "ollama": self._init_ollama
        }
        
        init_method = provider_init_methods.get(self.config.provider)
        if init_method:
            return init_method()
        else:
            rag_logger.error(f"Unsupported provider: {self.config.provider}")
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _init_openai(self):
        """Initialize OpenAI client"""
        return OpenAI(api_key=self.config.api_key)
    
    def _init_ollama(self):
        """Initialize Ollama client"""
        url = self.config.base_url or LLM_DEFAULT_URL
        client = Client(host=url)
        return client

    def _init_vector_db(self):
        """Initialize Chroma client and collection"""
        try:
            db_client = handle_vector_db(self.config.db_name, "load")["db_client"]
            return db_client.get_or_create_collection(name=RAG_DEFAULT_COLLECTION_NAME)
        except Exception as e:
            rag_logger.error(f"Error initializing Chroma client: {str(e)}")
            raise Exception(f"Error initializing Chroma client: {str(e)}")

    def generate_embedding(self, texts: list) -> list:
        """Generate embedding based on the LLM type"""
        if self.config.provider == "gpt":
            return self._get_openai_embedding(texts)
        elif self.config.provider == "ollama":
            return self._get_ollama_embedding(texts)
        else:
            rag_logger.error(f"Unsupported provider: {self.provider}")
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _get_openai_embedding(self, texts: list) -> list:
        """Generate embedding using OpenAI API"""
        try:
            request_params = {
                "model": self.config.embedding_model,
                "input": texts,
                "encoding_format": "float",
            }
            
            response = self.embedding_client.embeddings.create(**request_params)    
            embeddings = [d.embedding for d in response.data]
            return embeddings

        except Exception as e:
            rag_logger.error(f"Error generating OpenAI response: {str(e)}")
            raise Exception(f"Error generating OpenAI response: {str(e)}")
    
    def _get_ollama_embedding(self, texts: list) -> list:
        """Generate embedding using Ollama API"""
        try:
            request_params = {
                "model": self.config.embedding_model,
                "input": texts,
            }

            response = self.embedding_client.embed(**request_params)
            return response["embeddings"]
                            
        except Exception as e:
            rag_logger.error(f"Error generating Ollama response: {str(e)}")
            raise Exception(f"Error generating Ollama response: {str(e)}")

    def get_nb_records(self) -> int:
        """Get the number of records in the database"""
        return self.db_client_collection.count()

    def store_document(self, document: str, markdown: bool = False):
        """Store document in the database"""
        try:
            rag_logger.info(f"Storing document: {document}")
            doc_content = convert_to_markdown(document) if markdown else get_file_content(document)
            if not doc_content:
                rag_logger.error(f"Error processing file content: {document}")
                raise Exception(f"Error processing file content: {document}")
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_content))
            chunks = handle_chunk_documents(doc_content, self.config.chunk_size, self.config.chunk_overlap, markdown)
            max_batch_size = int(self.config.batch_size or RAG_DEFAULT_BATCH_SIZE)
            for start in range(0, len(chunks), max_batch_size):
                batch = chunks[start : start + max_batch_size]
                embeddings = self.generate_embedding(batch)
                ids = [f"{doc_id}_chunk{start + i}" for i in range(len(batch))]
                self.db_client_collection.add(
                    embeddings=embeddings,
                    documents=batch,
                    ids=ids
                )
            rag_logger.info(f"Document stored successfully: {document}")
        except Exception as e:
            rag_logger.error(f"Error storing documents: {str(e)}")
            raise Exception(f"Error storing documents: {str(e)}")

    def search_documents(self, query: str, max_results: int) -> dict:
        """Search documents in the database"""
        try:
            embedding = self.generate_embedding([query])
            results = self.db_client_collection.query(
                query_embeddings=embedding,
                n_results=max_results
            )
            return results
        except Exception as e:
            rag_logger.error(f"Error searching documents: {str(e)}")
            raise Exception(f"Error searching documents: {str(e)}")
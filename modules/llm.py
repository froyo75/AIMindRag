from utils.config import LLM_DEFAULT_OLLAMA_URL, LLM_DEFAULT_MODEL, LLM_DEFAULT_PROVIDER, LLM_CONFIG_DIR_PATH, LLM_DEFAULT_MAX_TOKENS, LLM_DEFAULT_TEMPERATURE
from utils.load import available_llm_config_providers
from openai import OpenAI
from ollama import Client
from dataclasses import dataclass
from utils.helpers import apply_config, create_dir
from pathlib import Path
from utils.logger import get_logger

llm_logger = get_logger(__name__)

def handle_llm_config(provider: str, model: str, api_key: str, action: str):
    """Handle LLM configuration creation and deletion"""
    config_file = available_llm_config_providers.get(provider)
    success = False
    message = "Invalid action"
    data = {"provider": provider, "model": model}
    create_dir(LLM_CONFIG_DIR_PATH)

    if not config_file:
        llm_logger.error(f"Unsupported provider: {provider}")
        raise ValueError(f"Unsupported provider: {provider}")
    
    config_file_path = Path(LLM_CONFIG_DIR_PATH) / config_file

    if not config_file_path.exists():
        action = "save"
    
    if action in ["save", "load"]:
        if action == "save":
            success, current_config = apply_config(config_file_path, "r")
            if not success:
                current_config = {}

            config = {
                "model": model or current_config.get("model") or LLM_DEFAULT_MODEL,
                "api_key": api_key or current_config.get("api_key") or "",
                "provider": provider or current_config.get("provider") or LLM_DEFAULT_PROVIDER,
                "max_tokens": current_config.get("max_tokens") or LLM_DEFAULT_MAX_TOKENS,
                "temperature": current_config.get("temperature") or LLM_DEFAULT_TEMPERATURE,
            }

            success, _ = apply_config(config_file_path, "w", config)
            message = "LLM configuration saved successfully" if success else "Failed to save LLM configuration"
        elif action == "load":
            success, data = apply_config(config_file_path, "r")
            message = "LLM configuration loaded successfully" if success else "Failed to load LLM configuration (check if the configuration file exists or apply for new configuration)"
    return {"success": success, "message": message, "data": data}
    
@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = ""
    api_key: str = ""
    stream: bool = True
    max_tokens: int = 1000
    temperature: float = 0.5

class LLMClient:
    def __init__(self, provider: str):
        self.provider = provider
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client based on the provider"""
        loaded_config = handle_llm_config(self.provider, None, None, "load")
        if not loaded_config["success"]:
            llm_logger.error(loaded_config["message"])
            raise ValueError(loaded_config["message"])

        self.config = LLMConfig(**loaded_config["data"])
        
        provider_init_methods = {
            "gpt": self._init_openai,
            "ollama": self._init_ollama
        }
        
        init_method = provider_init_methods.get(self.config.provider)
        if init_method:
            return init_method()
        else:
            llm_logger.error(f"Unsupported provider: {self.config.provider}")
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        if not self.config.api_key:
            llm_logger.error("Api key must be set in the configuration file")
            raise ValueError("Api key must be set in the configuration file")
        return OpenAI(api_key=self.config.api_key)
    
    def _init_ollama(self):
        """Initialize Ollama client"""
        url = LLM_DEFAULT_OLLAMA_URL
        client = Client(host=url)
        return client
    
    def generate_response(self, messages, tools=None):
        """Generate response based on the LLM type with optional tool support"""
        if self.config.provider == "gpt":
            return self._generate_openai_response(messages, tools)
        elif self.config.provider == "ollama":
            return self._generate_ollama_response(messages, tools)
        else:
            llm_logger.error(f"Unsupported provider: {self.provider}")
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _generate_openai_response(self, messages, tools=None):
        """Generate response using OpenAI API with streaming and tool support"""
        try:
            request_params = {
                "model": self.config.model,
                "messages": messages,
                "max_completion_tokens": int(self.config.max_tokens),
                "temperature": float(self.config.temperature),
                "tools": tools,
                "stream": self.config.stream,
            }
            
            response = self.client.chat.completions.create(**request_params)        
            return response

        except Exception as e:
            llm_logger.error(f"Error generating OpenAI response: {str(e)}")
            raise Exception(f"Error generating OpenAI response: {str(e)}")
    
    def _generate_ollama_response(self, messages, tools=None):
        """Generate response using Ollama API with tool support"""
        try:
            request_params = {
                "model": self.config.model,
                "messages": messages,
                "tools": tools,
                "stream": self.config.stream,
                "options": {
                    "num_predict": int(self.config.max_tokens),
                    "temperature": float(self.config.temperature),
                }
            }

            response = self.client.chat(**request_params)
            return response
                            
        except Exception as e:
            llm_logger.error(f"Error generating Ollama response: {str(e)}")
            raise Exception(f"Error generating Ollama response: {str(e)}")

    def get_available_models(self):
        """Get available models based on the LLM type"""
        try:
            if self.config.provider == "gpt":
                available_models = self.client.models.list()
                models_list = [model.id for model in available_models]
                return models_list
            elif self.config.provider == "ollama":
                available_models = self.client.list()
                models_list = [m["model"].split(":", 1)[0] for m in available_models["models"]]
                return models_list
            else:
                llm_logger.error(f"Unsupported provider: {self.provider}")
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            llm_logger.error(f"Error getting available models: {str(e)}")
            raise Exception(f"Error getting available models: {str(e)}")    
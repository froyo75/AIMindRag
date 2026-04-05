import json
from utils.config import LLM_DEFAULT_URL, LLM_DEFAULT_MODEL, LLM_DEFAULT_PROVIDER, LLM_CONFIG_DIR_PATH, LLM_DEFAULT_MAX_TOKENS, LLM_DEFAULT_TEMPERATURE, \
    LLM_DEFAULT_REASONING_EFFORT, LLM_SYSTEM_PROMPT, OLLAMA_PREFIX_MODELS, OPENAI_PREFIX_MODELS, CLAUDE_PREFIX_MODELS
from utils.load import available_llm_config_providers
from openai import OpenAI
from ollama import Client
from anthropic import Anthropic
from dataclasses import dataclass
from utils.helpers import apply_config, create_dir
from pathlib import Path
from utils.logger import get_logger

llm_logger = get_logger(__name__)

ollama_prefix_models_supporting_think_level = tuple(OLLAMA_PREFIX_MODELS.split(","))
openai_prefix_models_supporting_think_level = tuple(OPENAI_PREFIX_MODELS.split(","))
claude_prefix_models_supporting_think_level = tuple(CLAUDE_PREFIX_MODELS.split(","))

def handle_llm_config(provider: str, model: str, api_key: str, action: str, reasoning_effort: str, base_url: str) -> dict:
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

            if provider == "ollama":
                url = base_url or current_config.get("base_url") or LLM_DEFAULT_URL
            else:
                url = ""

            config = {
                "model": model or current_config.get("model") or LLM_DEFAULT_MODEL,
                "api_key": api_key or current_config.get("api_key") or "",
                "provider": provider or current_config.get("provider") or LLM_DEFAULT_PROVIDER,
                "max_tokens": current_config.get("max_tokens") or LLM_DEFAULT_MAX_TOKENS,
                "temperature": current_config.get("temperature") or LLM_DEFAULT_TEMPERATURE,
                "reasoning_effort": reasoning_effort or current_config.get("reasoning_effort") or LLM_DEFAULT_REASONING_EFFORT,
                "base_url": url,
            }

            success, _ = apply_config(config_file_path, "w", config)
            message = "LLM configuration saved successfully" if success else "Failed to save LLM configuration"
        elif action == "load":
            success, data = apply_config(config_file_path, "r")
            message = "LLM configuration loaded successfully" if success else "Failed to load LLM configuration (check if the configuration file exists or apply for new configuration)"
    return {"success": success, "message": message, "data": data}
    
@dataclass
class LLMConfig:
    provider: str = ""
    model: str = ""
    api_key: str = ""
    stream: bool = True
    max_tokens: int = 0
    temperature: float = 0.7
    reasoning_effort: str = ""
    base_url: str = ""

class LLMClient:
    def __init__(self, provider: str, use_thinking: bool = False):
        self.provider = provider
        self.use_thinking = use_thinking
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client based on the provider"""
        loaded_config = handle_llm_config(self.provider, None, None, "load", None, None)
        if not loaded_config["success"]:
            llm_logger.error(loaded_config["message"])
            raise ValueError(loaded_config["message"])

        self.config = LLMConfig(**loaded_config["data"])
        
        provider_init_methods = {
            "gpt": self._init_openai,
            "claude": self._init_claude,
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

    def _init_claude(self):
        """Initialize Claude client"""
        if not self.config.api_key:
            llm_logger.error("Api key must be set in the configuration file")
            raise ValueError("Api key must be set in the configuration file")
        return Anthropic(api_key=self.config.api_key)
    
    def _init_ollama(self):
        """Initialize Ollama client"""
        url = self.config.base_url or LLM_DEFAULT_URL
        client = Client(host=url)
        return client
    
    def generate_response(self, messages, tools=None) -> dict:
        """Generate response based on the LLM type with optional tool support"""
        if self.config.provider == "gpt":
            return self._generate_openai_response(messages, tools)
        elif self.config.provider == "claude":
            return self._generate_claude_response(messages, tools)
        elif self.config.provider == "ollama":
            return self._generate_ollama_response(messages, tools)
        else:
            llm_logger.error(f"Unsupported provider: {self.provider}")
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _generate_openai_response(self, messages, tools=None) -> dict:
        """Generate response using OpenAI API with reasoning and tool support"""
        try:
            request_params = {
                "model": self.config.model,
                "input": messages,
                "stream": self.config.stream,
            }

            if tools:
                request_params["tools"] = tools

            if int(self.config.max_tokens) > 0:
                request_params["max_output_tokens"] = int(self.config.max_tokens)

            is_reasoning_model = self.config.model.startswith(openai_prefix_models_supporting_think_level)
            if is_reasoning_model:
                if self.use_thinking:
                    request_params["reasoning"] = {
                        "effort": self.config.reasoning_effort,
                        "summary": "auto"
                    }
            else:
                if float(self.config.temperature) > 0:
                    request_params["temperature"] = float(self.config.temperature)

            response = self.client.responses.create(**request_params)
            return response

        except Exception as e:
            llm_logger.error(f"Error generating OpenAI response: {str(e)}")
            raise Exception(f"Error generating OpenAI response: {str(e)}")
    
    def _generate_claude_response(self, messages, tools=None) -> dict:
        """Generate response using Claude API with reasoning and tool support"""
        try:
            request_params = {
                "system": LLM_SYSTEM_PROMPT,
                "model": self.config.model,
                "messages": messages,
                "stream": self.config.stream
            }

            if tools:
                request_params["tools"] = tools

            is_reasoning_model = self.config.model.startswith(claude_prefix_models_supporting_think_level)
            if is_reasoning_model:
                if self.use_thinking:
                    request_params["thinking"] = {"type": "adaptive"}
                    request_params["output_config"] = {"effort": self.config.reasoning_effort}
            else:
                if float(self.config.temperature) > 0:
                    request_params["temperature"] = float(self.config.temperature)

            if int(self.config.max_tokens) > 0:
                request_params["max_tokens"] = int(self.config.max_tokens)
            else:
                request_params["max_tokens"] = LLM_DEFAULT_MAX_TOKENS

            response = self.client.messages.create(**request_params)
            return response
                            
        except Exception as e:
            llm_logger.error(f"Error generating Claude response: {str(e)}")
            raise Exception(f"Error generating Claude response: {str(e)}")
    
    def _generate_ollama_response(self, messages, tools=None) -> dict:
        """Generate response using Ollama API with reasoning and tool support"""
        try:
            request_params = {
                "model": self.config.model,
                "messages": messages,
                "stream": self.config.stream,
                "options": {}
            }

            if tools:
                request_params["tools"] = tools

            think_level = False
            is_reasoning_model = self.config.model.startswith(ollama_prefix_models_supporting_think_level)
            if self.use_thinking:
                if self.config.model.startswith(ollama_prefix_models_supporting_think_level):
                    think_level = self.config.reasoning_effort
                else:
                    think_level = True
            request_params["think"] = think_level

            if int(self.config.max_tokens) > 0:
                request_params["options"]["num_predict"] = int(self.config.max_tokens)

            if float(self.config.temperature) > 0:
                request_params["options"]["temperature"] = float(self.config.temperature)

            response = self.client.chat(**request_params)
            return response
                            
        except Exception as e:
            llm_logger.error(f"Error generating Ollama response: {str(e)}")
            raise Exception(f"Error generating Ollama response: {str(e)}")

    def get_available_models(self) -> list:
        """Get available models based on the LLM type"""
        try:
            if self.config.provider == "gpt":
                available_models = self.client.models.list()
                models_list = [model.id for model in available_models]
                return models_list
            elif self.config.provider == "claude":
                available_models = self.client.models.list()
                models_list = [model.id for model in available_models]
                return models_list
            elif self.config.provider == "ollama":
                available_models = self.client.list()
                models_list = [m["model"] for m in available_models["models"]]
                return models_list
            else:
                llm_logger.error(f"Unsupported provider: {self.provider}")
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            llm_logger.error(f"Error getting available models: {str(e)}")
            raise Exception(f"Error getting available models: {str(e)}")    
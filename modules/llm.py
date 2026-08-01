import json
import requests
from utils.config import LLM_DEFAULT_URL, LLM_DEFAULT_MODEL, LLM_DEFAULT_PROVIDER, LLM_CONFIG_DIR_PATH, LLM_DEFAULT_MAX_TOKENS, LLM_DEFAULT_TEMPERATURE, \
    LLM_DEFAULT_REASONING_EFFORT, LLM_SYSTEM_PROMPT
from utils.load import available_llm_config_providers, available_voyage_embedding_models
from openai import OpenAI
from ollama import Client
from anthropic import Anthropic
from dataclasses import dataclass
from utils.helpers import apply_config, create_dir
from pathlib import Path
from utils.logger import get_logger

llm_logger = get_logger(__name__)

_OLLAMA_THINK_LEVELS = {
    "minimal": "low", "low": "low", "medium": "medium",
    "high": "high", "xhigh": "high", "max": "high",
}

_OPENAI_EFFORT_LEVELS = {
    "minimal": "minimal", "low": "low", "medium": "medium",
    "high": "high", "xhigh": "high", "max": "high",
}

_CLAUDE_EFFORT_LEVELS = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}

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

            url = base_url or current_config.get("base_url")
            if not url and provider in ("ollama", "openai_compat"):
                url = LLM_DEFAULT_URL

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
            if success and not data.get("base_url") and data.get("provider") in ("ollama", "openai_compat"):
                data["base_url"] = LLM_DEFAULT_URL
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
    def __init__(self, provider: str, use_thinking: bool = False, base_url_override: str = None):
        self.provider = provider
        self.use_thinking = use_thinking
        self.base_url_override = base_url_override
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client based on the provider"""
        loaded_config = handle_llm_config(self.provider, None, None, "load", None, None)
        if not loaded_config["success"]:
            llm_logger.error(loaded_config["message"])
            raise ValueError(loaded_config["message"])

        self.config = LLMConfig(**loaded_config["data"])
        if self.base_url_override:
            self.config.base_url = self.base_url_override

        provider_init_methods = {
            "gpt": self._init_openai,
            "claude": self._init_claude,
            "ollama": self._init_ollama,
            "openai_compat": self._init_openai_compat
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
        return OpenAI(api_key=self.config.api_key, base_url=self.config.base_url or None)

    def _init_openai_compat(self):
        """Initialize OpenAI-compatible client (OpenRouter, vLLM, etc.)"""
        if not self.config.base_url:
            llm_logger.error("base_url must be set in the configuration file for openai_compat")
            raise ValueError("base_url must be set in the configuration file for openai_compat")
        api_key = self.config.api_key or "EMPTY"
        return OpenAI(api_key=api_key, base_url=self.config.base_url or LLM_DEFAULT_URL)

    def _init_claude(self):
        """Initialize Claude client"""
        if not self.config.api_key:
            llm_logger.error("Api key must be set in the configuration file")
            raise ValueError("Api key must be set in the configuration file")            
        return Anthropic(api_key=self.config.api_key, base_url=self.config.base_url or None)
    
    def _init_ollama(self):
        """Initialize Ollama client"""
        client = Client(host=self.config.base_url or LLM_DEFAULT_URL)
        return client
    
    def generate_response(self, messages, tools=None) -> dict:
        """Generate response based on the LLM type with optional tool support"""
        if self.config.provider == "gpt":
            return self._generate_openai_response(messages, tools)
        elif self.config.provider == "openai_compat":
            return self._generate_openai_compat_response(messages, tools)
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

            if self.use_thinking and self.config.reasoning_effort:
                request_params["reasoning"] = {
                    "effort": _OPENAI_EFFORT_LEVELS.get(self.config.reasoning_effort, "low"),
                    "summary": "auto"
                }
            else:
                temperature = float(self.config.temperature)
                if 0.0 <= temperature <= 2.0:
                    request_params["temperature"] = temperature

            response = self.client.responses.create(**request_params)
            return response

        except Exception as e:
            llm_logger.error(f"Error generating OpenAI response: {str(e)}")
            raise Exception(f"Error generating OpenAI response: {str(e)}")

    def _generate_openai_compat_response(self, messages, tools=None) -> dict:
        """Generate response via the OpenAI-compatible Chat Completions API"""
        try:
            request_params = {
                "model": self.config.model,
                "messages": messages,
                "stream": self.config.stream,
            }

            if self.config.stream:
                request_params["stream_options"] = {"include_usage": True}

            if tools:
                request_params["tools"] = tools

            if int(self.config.max_tokens) > 0:
                request_params["max_tokens"] = int(self.config.max_tokens)

            if self.use_thinking and self.config.reasoning_effort:
                request_params["reasoning_effort"] = self.config.reasoning_effort
            else:
                temperature = float(self.config.temperature)
                if 0.0 <= temperature <= 2.0:
                    request_params["temperature"] = temperature

            return self.client.chat.completions.create(**request_params)

        except Exception as e:
            llm_logger.error(f"Error generating OpenAI-compatible response: {str(e)}")
            raise Exception(f"Error generating OpenAI-compatible response: {str(e)}")
    
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

            if self.use_thinking and self.config.reasoning_effort:
                request_params["thinking"] = {"type": "adaptive"}
                request_params["output_config"] = {
                    "effort": _CLAUDE_EFFORT_LEVELS.get(self.config.reasoning_effort, "low")
                }
            else:
                temperature = float(self.config.temperature)
                if 0.0 <= temperature <= 2.0:
                    request_params["temperature"] = temperature

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
            if self.use_thinking:
                if self.config.reasoning_effort:
                    think_level = _OLLAMA_THINK_LEVELS.get(self.config.reasoning_effort, "low")
                else:
                    think_level = True
            request_params["think"] = think_level

            if int(self.config.max_tokens) > 0:
                request_params["options"]["num_predict"] = int(self.config.max_tokens)

            temperature = float(self.config.temperature)
            if 0.0 <= temperature <= 2.0:
                request_params["options"]["temperature"] = temperature

            response = self.client.chat(**request_params)
            return response
                            
        except Exception as e:
            llm_logger.error(f"Error generating Ollama response: {str(e)}")
            raise Exception(f"Error generating Ollama response: {str(e)}")

    def _get_openai_compat_embedding_models(self) -> list:
            """Get available embedding models from OpenAI Compatible API """
            base_url = self.config.base_url
            if not base_url:
                return []
            try:
                response = requests.get(
                    f"{base_url}/embeddings/models",
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                    timeout=15,
                )
                if response.status_code != 200:
                    return []
                payload = response.json()
                entries = payload.get("data") or payload.get("models") or []
                embedding_models = [
                    entry if isinstance(entry, str) else (entry.get("id") or entry.get("name"))
                    for entry in entries
                ]
                return [model_id for model_id in embedding_models if model_id]
            except Exception as e:
                llm_logger.warning(f"Could not fetch embedding models from {base_url}/embeddings/models: {str(e)}")
                return []

    

    def get_available_models(self) -> list:
        """Get available models based on the LLM type"""
        try:
            if self.config.provider in ("gpt", "openai_compat"):
                raw_response = self.client.models.with_raw_response.list()
                payload = json.loads(raw_response.text)
                entries = payload.get("data") or payload.get("models") or []
                models_list = [
                    entry if isinstance(entry, str) else (entry.get("id") or entry.get("name"))
                    for entry in entries
                ]
                models_list = [model_id for model_id in models_list if model_id]
                if self.config.provider == "openai_compat":
                    embedding_models = self._get_openai_compat_embedding_models()
                    for model_id in embedding_models:
                        if model_id not in models_list:
                            models_list.append(model_id)
                return models_list
            elif self.config.provider == "claude":
                models_list = [model_id for model_id in available_voyage_embedding_models if model_id]
                return models_list
            elif self.config.provider == "ollama":
                available_models = self.client.list()
                models_list = [model["model"] for model in available_models["models"]]
                return models_list
            else:
                llm_logger.error(f"Unsupported provider: {self.config.provider}")
                raise ValueError(f"Unsupported provider: {self.config.provider}")
        except Exception as e:
            llm_logger.error(f"Error getting available models: {str(e)}")
            raise Exception(f"Error getting available models: {str(e)}")
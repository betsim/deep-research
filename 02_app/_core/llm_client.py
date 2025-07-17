from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Iterator
import os
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from dotenv import load_dotenv
from _core.config import config
from _core.logger import custom_logger


try:
    dotenv_path = config["api_keys"]["dotenv_path"]
    load_dotenv(dotenv_path)
    custom_logger.info_console(f"Loaded .env from: {dotenv_path}")
except Exception as e:
    custom_logger.info_console(f"Error loading .env file: {e}")


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def call(self, prompt: str, **kwargs) -> str:
        """Make a standard call to the LLM."""
        pass

    @abstractmethod
    def call_structured(
        self, prompt: str, json_schema: Dict[str, Any], **kwargs
    ) -> str:
        """Make a call to the LLM and retrieve a structured response."""
        pass

    @abstractmethod
    def call_streamed(self, prompt: str, **kwargs) -> Iterator:
        """Stream responses from the LLM."""
        pass


class OpenRouterClient(LLMClient):
    """OpenRouter API client implementation."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not found")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=self.api_key
        )

        # Retry configuration
        self.retry_decorator = retry(
            wait=wait_random_exponential(
                multiplier=config["llm"]["tenacity_wait_multiplier"],
                max=config["llm"]["tenacity_wait_max"],
            ),
            stop=stop_after_attempt(config["llm"]["tenacity_stop_attempts"]),
        )

    @property
    def _retry(self):
        return self.retry_decorator

    def call(
        self,
        prompt: str,
        model_id: str = None,
        temperature: float = None,
        max_tokens: int = None,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Make a standard call to OpenRouter."""

        @self._retry
        def _call():
            completion = self.client.chat.completions.create(
                model=model_id or config["models"]["performance_low"],
                temperature=temperature or config["temperature"]["low"],
                max_tokens=max_tokens or config["llm"]["max_tokens"],
                reasoning_effort=reasoning_effort,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return completion.choices[0].message.content.strip()

        return _call()

    def call_structured(
        self,
        prompt: str,
        json_schema: Dict[str, Any],
        model_id: str = None,
        temperature: float = None,
        max_tokens: int = None,
        system_message: str = None,
        **kwargs,
    ) -> str:
        """Make a structured call to OpenRouter."""

        @self._retry
        def _call():
            completion = self.client.chat.completions.create(
                model=model_id or config["models"]["performance_low"],
                temperature=temperature or config["temperature"]["low"],
                max_tokens=max_tokens or config["llm"]["max_tokens"],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "output",
                        "strict": True,
                        "schema": json_schema,
                    },
                },
                messages=[
                    {
                        "role": "developer",
                        "content": system_message or config["llm"]["system_message"],
                    },
                    {"role": "user", "content": prompt},
                ],
                **kwargs,
            )
            return completion.choices[0].message.content

        return _call()

    def call_streamed(
        self,
        prompt: str,
        model_id: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs,
    ) -> Iterator:
        """Stream responses from OpenRouter."""

        custom_logger.info_console(
            f"Streaming response for prompt: {prompt[:50]}... (model: {model_id or config['models']['performance_low']})"
        )

        @self._retry
        def _stream():
            return self.client.chat.completions.create(
                model=model_id or config["models"]["performance_low"],
                temperature=temperature or config["temperature"]["low"],
                max_tokens=max_tokens or config["llm"]["max_tokens"],
                stream=True,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

        return _stream()


class ClientManager:
    """Manages LLM client instances."""

    _instances: Dict[str, LLMClient] = {}

    @classmethod
    def get_client(cls, provider: str = "openrouter") -> LLMClient:
        """Get or create a client instance."""
        if provider not in cls._instances:
            if provider == "openrouter":
                cls._instances[provider] = OpenRouterClient()
            else:
                raise ValueError(f"Unknown provider: {provider}")

        return cls._instances[provider]

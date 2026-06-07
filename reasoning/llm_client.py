"""
LLM Client for the reasoning engine.

Supports Ollama (local) and OpenAI-compatible APIs.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM client supporting Ollama (local) and OpenAI-compatible endpoints.

    The client sends structured prompts and parses JSON responses.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.provider = config.provider.value
        self.model = config.model
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the appropriate LLM client."""
        if self.provider == "ollama":
            try:
                import ollama
                self._client = ollama.Client(host=self.config.base_url)
                logger.info(f"Ollama client initialized (model={self.model})")
            except ImportError:
                logger.warning("Ollama package not installed, LLM reasoning disabled")
            except Exception as e:
                logger.warning(f"Ollama client init failed: {e}")
        elif self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url if "localhost" not in self.config.base_url else None,
                )
                logger.info(f"OpenAI client initialized (model={self.model})")
            except ImportError:
                logger.warning("OpenAI package not installed")

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt for context.

        Returns:
            Raw text response from the LLM.
        """
        if self._client is None:
            logger.warning("LLM client not available, returning empty response")
            return "{}"

        try:
            if self.provider == "ollama":
                return self._generate_ollama(prompt, system_prompt)
            elif self.provider == "openai":
                return self._generate_openai(prompt, system_prompt)
            else:
                return "{}"
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "{}"

    def generate_json(self, prompt: str, system_prompt: str = "") -> dict:
        """Generate and parse a JSON response from the LLM.

        Returns:
            Parsed JSON dict. Returns empty dict on failure.
        """
        response = self.generate(prompt, system_prompt)

        # Try to extract JSON from response
        try:
            # Look for JSON in the response
            text = response.strip()

            # Try direct parse
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in markdown
        try:
            if "```json" in response:
                start = response.index("```json") + 7
                end = response.index("```", start)
                return json.loads(response[start:end].strip())
            elif "```" in response:
                start = response.index("```") + 3
                end = response.index("```", start)
                return json.loads(response[start:end].strip())
        except (ValueError, json.JSONDecodeError):
            pass

        # Try to find JSON object pattern
        try:
            start = response.index("{")
            end = response.rindex("}") + 1
            return json.loads(response[start:end])
        except (ValueError, json.JSONDecodeError):
            logger.warning(f"Failed to parse JSON from LLM response: {response[:200]}")
            return {}

    # -------------------------------------------------------------------
    # Provider-specific implementations
    # -------------------------------------------------------------------

    def _generate_ollama(self, prompt: str, system_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        )
        return response["message"]["content"]

    def _generate_openai(self, prompt: str, system_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content or ""

    @property
    def is_available(self) -> bool:
        return self._client is not None

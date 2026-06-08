"""
vLLM API client with a transformers pipeline-compatible interface.

Provides concurrent request handling via ThreadPoolExecutor for
high-throughput LLM inference through vLLM's OpenAI-compatible API.
"""

import json
import logging
from typing import List, Dict, Union, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VLLMClient")


class VLLMClient:
    """
    vLLM API client compatible with the HuggingFace transformers pipeline interface.

    Supports both chat-style (List[Dict] messages) and text-completion inputs
    with configurable concurrency for high-throughput scenarios.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000/v1",
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
        max_workers: int = 32,
    ):
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        self.max_workers = max_workers

        # Retry strategy for transient failures
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.max_workers,
            pool_maxsize=self.max_workers * 2
        )

        self.session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.model_name = model_name if model_name else self._get_model_name()
        logger.info(f"VLLMClient initialized: {self.api_url}, model: {self.model_name}, "
                     f"max_workers: {max_workers}, timeout: {timeout}")

    def _get_model_name(self) -> str:
        """Query the vLLM server for the loaded model name."""
        try:
            response = self.session.get(f"{self.api_url}/models", timeout=10)
            response.raise_for_status()
            models_data = response.json()
            if models_data.get("data"):
                return models_data["data"][0]["id"]
        except Exception as e:
            logger.warning(f"Could not fetch model name from server, using 'default': {e}")
        return "default"

    def _call_api(self, payload: Dict, use_chat_api: bool) -> Dict:
        """Low-level API call to vLLM."""
        endpoint = f"{self.api_url}/{'chat/completions' if use_chat_api else 'completions'}"
        response = self.session.post(endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _process_single_request(
        self, index: int, input_item: Any, gen_config: Dict
    ) -> tuple:
        """Process a single concurrent generation request."""
        try:
            # Auto-detect chat vs completion format
            is_chat = (
                isinstance(input_item, list)
                and len(input_item) > 0
                and isinstance(input_item[0], dict)
                and "role" in input_item[0]
            )

            payload = {"model": self.model_name, **gen_config}

            if is_chat:
                payload["messages"] = input_item
            elif isinstance(input_item, str):
                payload["messages"] = [{"role": "user", "content": input_item}]
                is_chat = True
            else:
                raise ValueError(f"Unsupported input format: {type(input_item)}")

            response = self._call_api(payload, is_chat)

            if is_chat:
                generated_text = response["choices"][0]["message"]["content"]
            else:
                generated_text = response["choices"][0]["text"]

            return (index, {"generated_text": generated_text})

        except Exception as e:
            logger.error(f"Request failed (index {index}): {e}")
            return (index, {"generated_text": f"Error: {str(e)}"})

    def __call__(
        self,
        inputs: Union[List[Any], Any],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = -1,
        stop: Optional[List[str]] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        Generate completions with concurrent batching.

        Compatible with the HuggingFace transformers pipeline interface.
        Returns a list of dicts, each containing 'generated_text'.
        """
        # Normalize to list
        single_input = False
        if not isinstance(inputs, list) or (
            len(inputs) > 0
            and isinstance(inputs[0], dict)
            and "role" in inputs[0]
        ):
            inputs = [inputs]
            single_input = True

        if not inputs:
            return []

        gen_config = {
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop": stop,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            **kwargs
        }

        results_map = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(inputs))) as executor:
            future_to_idx = {
                executor.submit(self._process_single_request, i, item, gen_config): i
                for i, item in enumerate(inputs)
            }
            for future in as_completed(future_to_idx):
                idx, res = future.result()
                results_map[idx] = res

        # Preserve original order
        ordered_results = [results_map[i] for i in range(len(inputs))]
        return ordered_results

    def generate_batch(self, prompts: List[Any], **kwargs) -> List[str]:
        """Convenience method: return list of generated text strings."""
        results = self(prompts, **kwargs)
        return [r["generated_text"] for r in results]

    def close(self):
        """Close the HTTP session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def build_vllm_client(
    api_url: str = "http://localhost:8000/v1",
    model_name: Optional[str] = None,
    max_workers: int = 32,
    timeout: Optional[int] = None,
) -> VLLMClient:
    """Factory function to build a VLLMClient."""
    return VLLMClient(
        api_url=api_url,
        model_name=model_name,
        max_workers=max_workers,
        timeout=timeout,
    )

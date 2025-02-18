from typing import Any, Optional

import httpx
from neo4j_graphrag.llm.openai_llm import BaseOpenAILLM
from utils import OPENROUTERAI_BASE_URL, OPENROUTERAI_API_KEY

class OpenRouterAILLM(BaseOpenAILLM):
    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """OpenRouterAI LLM

        Wrapper for the OpenRouterAI Python client LLM.

        Args:
            model_name (str):
            model_params (str): Parameters like temperature that will be passed to the model when text is sent to it. Defaults to None.
            kwargs: All other parameters will be passed to the openai.OpenAI init -- which also works for OpenRouterAI
        """
        super().__init__(model_name, model_params)
        if 'base_url' not in kwargs:
            kwargs['base_url'] = OPENROUTERAI_BASE_URL
        if 'api_key' not in kwargs:
            kwargs['api_key'] = OPENROUTERAI_API_KEY
        self.client = self.openai.OpenAI(**kwargs)
        self.async_client = self.openai.AsyncOpenAI(**kwargs)
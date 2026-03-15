# RAG_Core/models/llm_model.py  (FIXED – invoke_with_usage dùng cho Supervisor)
"""
Không thay đổi API, chỉ đảm bảo invoke() cũng track được token
khi cần (dùng invoke_with_usage thay thế).
"""

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from config.settings import settings
import traceback, json, logging
from typing import Iterator, AsyncIterator, Tuple, Optional
import httpx

logger = logging.getLogger(__name__)


class LLMModel:
    def __init__(self):
        logger.info(f"[LLMModel] model={settings.LLM_MODEL} base_url={settings.OLLAMA_URL}")
        self.llm = OllamaLLM(
            model=settings.LLM_MODEL,
            base_url=getattr(settings, "OLLAMA_URL", "http://ollama:11434"),
            temperature=0.1,
        )
        self.output_parser = StrOutputParser()
        self.ollama_url  = getattr(settings, "OLLAMA_URL", "http://ollama:11434")
        self.model_name  = settings.LLM_MODEL

    # ──────────────────────────────────────────────
    # NON-STREAMING
    # ──────────────────────────────────────────────

    def invoke(self, prompt: str, **kwargs) -> str:
        """Non-streaming – trả về text only (backward compat)."""
        text, _ = self.invoke_with_usage(prompt, **kwargs)
        return text

    def invoke_with_usage(self, prompt: str, **kwargs) -> Tuple[str, int]:
        """
        Non-streaming – trả về (text, total_tokens).
        Dùng cho: Supervisor, Generator (non-streaming), các agent khác.
        """
        try:
            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model":   self.model_name,
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": 0.1},
            }
            response = httpx.post(url, json=payload, timeout=120.0)
            response.raise_for_status()
            data   = response.json()
            text   = data.get("response", "")
            tokens = (
                (data.get("prompt_eval_count") or 0)
                + (data.get("eval_count") or 0)
            )
            return self.output_parser.parse(text), tokens
        except Exception as e:
            traceback.print_exc()
            return f"Lỗi xử lý: {str(e)}", 0

    # ──────────────────────────────────────────────
    # STREAMING
    # ──────────────────────────────────────────────

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        try:
            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model":   self.model_name,
                "prompt":  prompt,
                "stream":  True,
                "options": {"temperature": 0.1},
            }
            with httpx.stream("POST", url, json=payload, timeout=60.0) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        try:
                            d = json.loads(line)
                            chunk = d.get("response", "")
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\n\n[Lỗi streaming: {e}]"

    async def astream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        try:
            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model":   self.model_name,
                "prompt":  prompt,
                "stream":  True,
                "options": {"temperature": 0.1},
            }
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if line:
                            try:
                                d = json.loads(line)
                                chunk = d.get("response", "")
                                if chunk:
                                    yield chunk
                                if d.get("done"):
                                    break
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            logger.error(f"Async streaming error: {e}", exc_info=True)
            yield f"\n\n[Lỗi streaming: {e}]"

    async def astream_with_usage(self, prompt: str, **kwargs):
        """
        Async streaming trả về token usage.
        Yield str chunks → cuối cùng yield {"__token_usage__": N}.
        """
        total_tokens = 0
        try:
            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model":   self.model_name,
                "prompt":  prompt,
                "stream":  True,
                "options": {"temperature": 0.1},
            }
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if line:
                            try:
                                d = json.loads(line)
                                chunk = d.get("response", "")
                                if chunk:
                                    yield chunk
                                if d.get("done"):
                                    total_tokens = (
                                        (d.get("prompt_eval_count") or 0)
                                        + (d.get("eval_count") or 0)
                                    )
                                    break
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            logger.error(f"astream_with_usage error: {e}", exc_info=True)
            yield f"\n\n[Lỗi streaming: {e}]"

        yield {"__token_usage__": total_tokens}

    def create_chain(self, template: str):
        prompt = PromptTemplate.from_template(template)
        return prompt | self.llm | self.output_parser


llm_model = LLMModel()
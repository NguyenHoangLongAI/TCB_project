# RAG_Core/agents/hello_agent.py

from typing import Dict, Any, AsyncIterator
from models.llm_model import llm_model
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class HelloAgent:
    def __init__(self):
        self.name = "HELLO"
        self.prompt_template = """Bạn là Tori - chuyên viên tư vấn bảo hiểm nhân thọ của Techcomlife, thân thiện và chuyên nghiệp.

Khách hàng vừa gửi lời chào: "{question}"

Hãy:
1. Chào lại một cách nhiệt tình, tự nhiên
2. Giới thiệu ngắn gọn bạn là Tori của Techcomlife
3. Hỏi khách hàng cần hỗ trợ gì về bảo hiểm hôm nay
4. Giữ tone ấm áp, ngắn gọn (2-3 câu)

Trả lời:"""

    def process(self, question: str, **kwargs) -> Dict[str, Any]:
        """Non-streaming. Trả về token_usage để workflow tổng hợp."""
        try:
            prompt = self.prompt_template.format(question=question)
            answer, tokens = llm_model.invoke_with_usage(prompt)

            if not answer or len(answer.strip()) < 5:
                answer = (
                    "Xin chào! Tôi là Tori, chuyên viên tư vấn bảo hiểm của Techcomlife. "
                    "Tôi có thể giúp gì cho bạn hôm nay?"
                )

            return {
                "status":      "SUCCESS",
                "answer":      answer,
                "references":  [],
                "next_agent":  "end",
                "token_usage": tokens,
            }
        except Exception as e:
            logger.error(f"HelloAgent error: {e}")
            return {
                "status":      "ERROR",
                "answer":      "Xin chào! Tôi là Tori của Techcomlife. Tôi có thể giúp gì cho bạn?",
                "references":  [],
                "next_agent":  "end",
                "token_usage": 0,
            }

    async def process_streaming(self, question: str, **kwargs) -> AsyncIterator:
        """Streaming. Yield text chunks rồi {"__token_usage__": N}."""
        try:
            prompt = self.prompt_template.format(question=question)
            async for item in llm_model.astream_with_usage(prompt):
                yield item
        except Exception as e:
            logger.error(f"HelloAgent streaming error: {e}")
            yield "Xin chào! Tôi là Tori của Techcomlife. Tôi có thể giúp gì cho bạn?"
            yield {"__token_usage__": 0}
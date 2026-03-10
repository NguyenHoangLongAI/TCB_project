# RAG_Core/agents/base_streaming_agent.py
"""
Base class cho tất cả agents với streaming support
"""

from typing import Dict, Any, List, AsyncIterator
from models.llm_model import llm_model
import logging

logger = logging.getLogger(__name__)


class BaseStreamingAgent:
    """
    Base agent với streaming support cho tất cả agents
    """

    def __init__(self, name: str, prompt_template: str):
        self.name = name
        self.prompt_template = prompt_template

    def process(self, **kwargs) -> Dict[str, Any]:
        """
        Non-streaming process (compatibility)
        Subclass override nếu cần logic đặc biệt
        """
        raise NotImplementedError("Subclass must implement process()")

    async def process_streaming(self, **kwargs) -> AsyncIterator[str]:
        """
        Streaming process - DEFAULT IMPLEMENTATION

        Workflow:
        1. Format prompt từ kwargs
        2. Stream từ LLM
        3. Yield chunks
        """
        try:
            # Subclass phải implement method này để format prompt
            prompt = self._format_prompt(**kwargs)

            logger.info(f"🚀 {self.name}: Starting streaming")

            chunk_count = 0
            async for chunk in llm_model.astream(prompt):
                if chunk:
                    chunk_count += 1
                    logger.debug(f"{self.name} chunk #{chunk_count}: {chunk[:30]}...")
                    yield chunk

            logger.info(f"✅ {self.name}: Completed {chunk_count} chunks")

        except Exception as e:
            logger.error(f"❌ {self.name} streaming error: {e}", exc_info=True)
            yield f"\n\n[Lỗi {self.name}: {str(e)}]"

    def _format_prompt(self, **kwargs) -> str:
        """
        Format prompt từ template và kwargs
        Subclass PHẢI override method này
        """
        raise NotImplementedError("Subclass must implement _format_prompt()")

    def _get_fallback_answer(self, **kwargs) -> str:
        """
        Fallback answer nếu LLM fail
        Subclass có thể override
        """
        return "Xin lỗi, tôi không thể xử lý yêu cầu này lúc này."


# ============================================================================
# STREAMING-ENABLED AGENTS
# ============================================================================

class StreamingChatterAgent(BaseStreamingAgent):
    """ChatterAgent với streaming thật"""

    def __init__(self):
        prompt_template = """Bạn là một chuyên viên chăm sóc khách hàng của Techcomlife - công ty bảo hiểm nhân thọ uy tín, thân thiện và chuyên nghiệp, chuyên gia xử lý cảm xúc và an ủi khách hàng.

Nhiệm vụ: An ủi, làm dịu cảm xúc tiêu cực của khách hàng và cung cấp thông tin liên hệ hỗ trợ của Techcomlife.

Nội dung khách hàng: "{question}"
Lịch sử hội thoại: {history}
Số điện thoại hỗ trợ: {support_phone}

Hướng dẫn:
1. Thể hiện sự thông cảm và hiểu biết cảm xúc khách hàng
2. Xin lỗi một cách chân thành thay mặt Techcomlife
3. Đảm bảo sẽ cải thiện chất lượng dịch vụ bảo hiểm
4. Cung cấp số hotline Techcomlife để được hỗ trợ trực tiếp
5. Giữ thái độ ấm áp, chuyên nghiệp, đúng chuẩn mực của Techcomlife

Trả lời:"""

        super().__init__("CHATTER", prompt_template)
        self.support_phone = None  # Will be set from settings

    def _format_prompt(self, question: str, history: List = None, support_phone: str = "", **kwargs) -> str:
        history_text = "\n".join(history) if history else "Không có lịch sử"

        return self.prompt_template.format(
            question=question,
            history=history_text,
            support_phone=support_phone
        )

    def process(self, question: str, history: List = None, **kwargs) -> Dict[str, Any]:
        """Non-streaming process"""
        try:
            from config.settings import settings

            prompt = self._format_prompt(
                question=question,
                history=history,
                support_phone=settings.SUPPORT_PHONE
            )

            answer = llm_model.invoke(prompt)

            if not answer or len(answer.strip()) < 10:
                answer = self._get_fallback_answer()

            return {
                "status": "SUCCESS",
                "answer": answer,
                "references": [{"document_id": "support_contact", "type": "SUPPORT"}],
                "next_agent": "end"
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "answer": self._get_fallback_answer(),
                "references": [],
                "next_agent": "end"
            }


class StreamingOtherAgent(BaseStreamingAgent):
    """OtherAgent với streaming thật"""

    def __init__(self):
        prompt_template = """Bạn là một chuyên viên chăm sóc khách hàng của Techcomlife - công ty bảo hiểm nhân thọ uy tín, thân thiện và chuyên nghiệp, chuyên xử lý các yêu cầu ngoài phạm vi hỗ trợ.

Nhiệm vụ: Thông báo lịch sự khi yêu cầu nằm ngoài phạm vi tư vấn bảo hiểm và dịch vụ Techcomlife, đồng thời hướng dẫn khách hàng.

Yêu cầu của khách hàng: "{question}"
Số điện thoại hỗ trợ: {support_phone}

Hướng dẫn:
1. Giải thích rằng yêu cầu nằm ngoài phạm vi hỗ trợ bảo hiểm và dịch vụ Techcomlife hiện tại
2. Đề xuất liên hệ hotline Techcomlife để được tư vấn cụ thể hơn
3. Giữ thái độ lịch sự và chuyên nghiệp theo chuẩn mực Techcomlife
4. Không từ chối một cách thô lỗ

Trả lời:"""

        super().__init__("OTHER", prompt_template)

    def _format_prompt(self, question: str, support_phone: str = "", **kwargs) -> str:
        return self.prompt_template.format(
            question=question,
            support_phone=support_phone
        )

    def process(self, question: str, **kwargs) -> Dict[str, Any]:
        """Non-streaming process"""
        try:
            from config.settings import settings

            prompt = self._format_prompt(
                question=question,
                support_phone=settings.SUPPORT_PHONE
            )

            answer = llm_model.invoke(prompt)

            if not answer or len(answer.strip()) < 10:
                answer = self._get_fallback_answer()

            return {
                "status": "SUCCESS",
                "answer": answer,
                "references": [],
                "next_agent": "end"
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "answer": self._get_fallback_answer(),
                "references": [],
                "next_agent": "end"
            }


class StreamingNotEnoughInfoAgent(BaseStreamingAgent):
    """NotEnoughInfoAgent với streaming thật"""

    def __init__(self):
        prompt_template = """Bạn là một chuyên viên chăm sóc khách hàng của Techcomlife - công ty bảo hiểm nhân thọ uy tín, thân thiện và chuyên nghiệp.

Câu hỏi người dùng: "{question}"

YÊU CẦU BẮT BUỘC:
- Xin lỗi khách hàng vì chưa có thông tin để trả lời câu hỏi này
- KHÔNG giải thích thêm bất kỳ thông tin gì về chủ đề được hỏi
- KHÔNG suy đoán, KHÔNG cung cấp thông tin tham khảo
- Đề nghị khách hàng liên hệ hotline Techcomlife {support_phone} để được hỗ trợ chính xác
- Giữ thái độ lịch sự, chuyên nghiệp

Chỉ trả về nội dung trả lời, không giải thích gì thêm."""

        super().__init__("NOT_ENOUGH_INFO", prompt_template)

    def _format_prompt(self, question: str, support_phone: str = "", **kwargs) -> str:
        return self.prompt_template.format(
            question=question,
            support_phone=support_phone
        )

    def process(self, question: str, **kwargs) -> Dict[str, Any]:
        """Non-streaming process"""
        try:
            from config.settings import settings

            prompt = self._format_prompt(
                question=question,
                support_phone=settings.SUPPORT_PHONE
            )

            answer = llm_model.invoke(prompt)

            return {
                "status": "SUCCESS",
                "answer": answer,
                "references": [{"document_id": "llm_knowledge", "type": "GENERAL_KNOWLEDGE"}],
                "next_agent": "end"
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "answer": self._get_fallback_answer(),
                "references": [],
                "next_agent": "end"
            }
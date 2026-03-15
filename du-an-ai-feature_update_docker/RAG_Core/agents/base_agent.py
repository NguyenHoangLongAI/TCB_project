# RAG_Core/agents/base_agent.py  (FIXED – process() trả về token_usage)
"""
Fix: StreamingChatterAgent, StreamingOtherAgent, StreamingNotEnoughInfoAgent
     dùng invoke_with_usage() thay vì invoke() để trả về token count.
     Thêm "token_usage" vào kết quả process() của cả 3 agent.
"""

from typing import Dict, Any, List, AsyncIterator
from models.llm_model import llm_model
import logging

logger = logging.getLogger(__name__)


class BaseStreamingAgent:
    def __init__(self, name: str, prompt_template: str):
        self.name            = name
        self.prompt_template = prompt_template

    def process(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Subclass must implement process()")

    async def process_streaming(self, **kwargs) -> AsyncIterator[str]:
        """
        Default streaming implementation.
        Dùng astream_with_usage để yield text chunks rồi {"__token_usage__": N}.
        """
        try:
            prompt      = self._format_prompt(**kwargs)
            chunk_count = 0
            logger.info(f"🚀 {self.name}: Starting streaming")
            async for item in llm_model.astream_with_usage(prompt):
                if isinstance(item, dict) and "__token_usage__" in item:
                    logger.info(f"✅ {self.name}: {item['__token_usage__']} tokens, {chunk_count} chunks")
                    yield item   # forward sentinel lên workflow
                elif isinstance(item, str) and item:
                    chunk_count += 1
                    yield item
        except Exception as e:
            logger.error(f"❌ {self.name} streaming error: {e}", exc_info=True)
            yield f"\n\n[Lỗi {self.name}: {str(e)}]"
            yield {"__token_usage__": 0}

    def _format_prompt(self, **kwargs) -> str:
        raise NotImplementedError("Subclass must implement _format_prompt()")

    def _get_fallback_answer(self, **kwargs) -> str:
        return "Xin lỗi, tôi không thể xử lý yêu cầu này lúc này."


# ─────────────────────────────────────────────────────────────────────────────
# CHATTER
# ─────────────────────────────────────────────────────────────────────────────

class StreamingChatterAgent(BaseStreamingAgent):
    def __init__(self):
        prompt_template = """Bạn là Tori, một chuyên viên chăm sóc khách hàng của Techcomlife - công ty bảo hiểm nhân thọ uy tín, thân thiện và chuyên nghiệp, chuyên gia xử lý cảm xúc và an ủi khách hàng.

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

    def _format_prompt(self, question: str, history: List = None, support_phone: str = "", **kwargs) -> str:
        history_text = "\n".join(history) if history else "Không có lịch sử"
        return self.prompt_template.format(
            question=question,
            history=history_text,
            support_phone=support_phone,
        )

    def process(self, question: str, history: List = None, **kwargs) -> Dict[str, Any]:
        """Non-streaming – trả về token_usage."""
        try:
            from config.settings import settings
            prompt = self._format_prompt(
                question=question,
                history=history,
                support_phone=settings.SUPPORT_PHONE,
            )
            # ✅ dùng invoke_with_usage
            answer, tokens = llm_model.invoke_with_usage(prompt)
            if not answer or len(answer.strip()) < 10:
                answer = self._get_fallback_answer()
            return {
                "status":      "SUCCESS",
                "answer":      answer,
                "references":  [{"document_id": "support_contact", "type": "SUPPORT"}],
                "next_agent":  "end",
                "token_usage": tokens,  # ✅
            }
        except Exception as e:
            return {
                "status":      "ERROR",
                "answer":      self._get_fallback_answer(),
                "references":  [],
                "next_agent":  "end",
                "token_usage": 0,
            }


# ─────────────────────────────────────────────────────────────────────────────
# OTHER
# ─────────────────────────────────────────────────────────────────────────────

class StreamingOtherAgent(BaseStreamingAgent):
    def __init__(self):
        prompt_template = """Bạn là Tori, một chuyên viên chăm sóc khách hàng của Techcomlife - công ty bảo hiểm nhân thọ uy tín, thân thiện và chuyên nghiệp, chuyên xử lý các yêu cầu ngoài phạm vi hỗ trợ.

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
        return self.prompt_template.format(question=question, support_phone=support_phone)

    def process(self, question: str, **kwargs) -> Dict[str, Any]:
        try:
            from config.settings import settings
            prompt = self._format_prompt(question=question, support_phone=settings.SUPPORT_PHONE)
            answer, tokens = llm_model.invoke_with_usage(prompt)
            if not answer or len(answer.strip()) < 10:
                answer = self._get_fallback_answer()
            return {
                "status":      "SUCCESS",
                "answer":      answer,
                "references":  [],
                "next_agent":  "end",
                "token_usage": tokens,  # ✅
            }
        except Exception as e:
            return {
                "status":      "ERROR",
                "answer":      self._get_fallback_answer(),
                "references":  [],
                "next_agent":  "end",
                "token_usage": 0,
            }


# ─────────────────────────────────────────────────────────────────────────────
# NOT ENOUGH INFO
# ─────────────────────────────────────────────────────────────────────────────

class StreamingNotEnoughInfoAgent(BaseStreamingAgent):
    def __init__(self):
        prompt_template = """Bạn là Tori, một chuyên viên chăm sóc khách hàng của Techcomlife - công ty bảo hiểm nhân thọ uy tín, thân thiện và chuyên nghiệp.

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
        return self.prompt_template.format(question=question, support_phone=support_phone)

    def process(self, question: str, **kwargs) -> Dict[str, Any]:
        try:
            from config.settings import settings
            prompt = self._format_prompt(question=question, support_phone=settings.SUPPORT_PHONE)
            answer, tokens = llm_model.invoke_with_usage(prompt)
            return {
                "status":      "SUCCESS",
                "answer":      answer,
                "references":  [{"document_id": "llm_knowledge", "type": "GENERAL_KNOWLEDGE"}],
                "next_agent":  "end",
                "token_usage": tokens,  # ✅
            }
        except Exception as e:
            return {
                "status":      "ERROR",
                "answer":      self._get_fallback_answer(),
                "references":  [],
                "next_agent":  "end",
                "token_usage": 0,
            }
# RAG_Core/agents/supervisor.py  (FIXED – track token usage)
"""
Fix: Supervisor dùng invoke_with_usage() thay vì invoke()
     để token của bước classify được tính vào total.
     Trả về thêm field "token_usage" trong kết quả classify_request().
"""

from typing import Dict, Any, List
from models.llm_model import llm_model
import logging, json, re

logger = logging.getLogger(__name__)


class SupervisorAgent:
    def __init__(self):
        self.name = "SUPERVISOR"
        self.classification_prompt = """Bạn là Tori, chuyên viên chăm sóc khách hàng của Techcomlife - người điều phối chính của hệ thống chatbot tư vấn bảo hiểm và dịch vụ Techcomlife.

        Nhiệm vụ:
        1. Dựa vào lịch sử hội thoại và câu hỏi hiện tại, hãy xác định ngữ cảnh (context) mà người dùng đang đề cập đến.
        2. Làm rõ câu hỏi nếu cần thiết (thay thế đại từ, bổ sung thông tin từ context).
        3. Phân loại câu hỏi và chọn agent phù hợp để xử lý.

        Các agent có thể chọn:
        - HELLO: Khách hàng chào hỏi, cảm ơn, tạm biệt, tin nhắn xã giao ngắn không chứa câu hỏi thực sự.
        - FAQ:
            - Dùng cho chào hỏi thân thiện, câu hỏi thường gặp.
            - Các câu hỏi liên quan đến sản phẩm và gói bảo hiểm của Techcomlife.
            - Tư vấn quyền lợi bảo hiểm nhân thọ, bảo hiểm sức khỏe, bảo hiểm tai nạn.
            - Thông tin về phí bảo hiểm, điều kiện tham gia, thủ tục bồi thường.
            - Quy trình, quy định nội bộ tại Techcomlife.
            - Thông tin về hợp đồng bảo hiểm, điều khoản, phụ lục.
            - Các dịch vụ hỗ trợ khách hàng của Techcomlife.
        - OTHER: Câu hỏi hoặc yêu cầu nằm ngoài phạm vi tư vấn bảo hiểm và dịch vụ Techcomlife.
        - CHATTER: Người dùng có dấu hiệu không hài lòng, giận dữ, hoặc cần được an ủi, làm dịu.
        - REPORTER: Khi người dùng phản ánh lỗi, mất kết nối, hoặc vấn đề kỹ thuật của hệ thống.

        Đầu vào:
        Câu hỏi hiện tại: "{question}"
        Lịch sử hội thoại: {history}

        YÊU CẦU QUAN TRỌNG:
        - Phân tích xem câu hỏi có phải follow-up (tiếp theo cuộc trò chuyện trước) không
            - Truy vết lịch sử để xác định chính xác đối tượng được nhắc tới.
            - Đặc biệt chú ý các cụm:
             "thành phần thứ X", "phần này", "nó", "ý trên", "cái đó", "OK","có", "chi tiết","hãy hướng dẫn", "tiếp tục" ...
            - Nếu lịch sử có DANH SÁCH ĐÁNH SỐ → ánh xạ theo ĐÚNG THỨ TỰ.
            - Nếu có yêu cầu hành động không cụ thể ("OK","có", "chi tiết","hãy hướng dẫn", "tiếp tục"...) → dựa vào lịch sử hội thoại làm rõ yêu cầu
            - Viết lại câu hỏi (contextualized_question) bằng TIẾNG VIỆT ĐẦY ĐỦ – RÕ NGHĨA – CÓ NGỮ CẢNH.
            - Đảm bảo câu hỏi được làm rõ (contextualized_question) phải có:
                - ĐỐI TƯỢNG cụ thể là gì
                - HÀNH ĐỘNG cụ thể là gì
                - Trong NGỮ CẢNH cụ thể là gì
        - Nếu không phải follow-up: contextualized_question = câu hỏi gốc, context_summary = "Câu hỏi độc lập"

        Hãy trả lời đúng định dạng JSON:
        {{
          "is_followup": true hoặc false,
          "contextualized_question": "Câu hỏi đã được làm rõ rất cụ thể hoặc câu hỏi gốc",
          "context_summary": "Tóm tắt ngắn gọn ngữ cảnh BẰNG TIẾNG VIỆT",
          "agent": "FAQ" hoặc "HELLO" hoặc "CHATTER" hoặc "REPORTER" hoặc "OTHER"
        }}

        Chỉ trả về JSON, không thêm text nào khác."""

    def classify_request(
        self,
        question: str,
        history: List[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Phân loại yêu cầu.
        Trả về thêm field "token_usage" để rag_workflow cộng dồn đúng.
        """
        try:
            logger.info("-" * 50)
            logger.info("👨‍💼 SUPERVISOR CLASSIFICATION")
            logger.info("-" * 50)
            logger.info(f"📝 Question: '{question}'")
            logger.info(f"📚 History Length: {len(history) if history else 0} messages")

            history_text = self._format_history(history or [])
            prompt = self.classification_prompt.format(
                question=question,
                history=history_text,
            )

            # ✅ FIX: dùng invoke_with_usage để lấy token count
            logger.info("🤖 Calling LLM for classification + contextualization…")
            response, supervisor_tokens = llm_model.invoke_with_usage(prompt)
            logger.info(f"💰 Supervisor used {supervisor_tokens} tokens")

            classification = self._parse_classification_response(response)

            agent_choice            = classification.get("agent", "").upper()
            is_followup             = classification.get("is_followup", False)
            contextualized_question = classification.get("contextualized_question", question)
            context_summary         = classification.get("context_summary", "")

            valid_agents = ["FAQ","HELLO", "CHATTER", "REPORTER", "OTHER"]
            if agent_choice not in valid_agents:
                logger.warning(f"⚠️  Invalid agent '{agent_choice}' → default to FAQ")
                agent_choice = "FAQ"

            logger.info(f"🎯 CLASSIFICATION RESULT:")
            logger.info(f"   Agent: {agent_choice}")
            logger.info(f"   Is Follow-up: {is_followup}")
            logger.info(f"   Original Q: '{question[:60]}'")
            logger.info(f"   Context Q:  '{contextualized_question[:60]}'")
            logger.info("-" * 50 + "\n")

            return {
                "agent":                    agent_choice,
                "contextualized_question":  contextualized_question,
                "context_summary":          context_summary,
                "is_followup":              is_followup,
                "reasoning":                classification.get("reasoning", ""),
                # ✅ NEW: trả về token dùng để workflow cộng dồn
                "token_usage":              supervisor_tokens,
            }

        except Exception as e:
            logger.error(f"❌ Error in supervisor classification: {e}", exc_info=True)
            return {
                "agent":                   "FAQ",
                "contextualized_question": question,
                "context_summary":         "",
                "is_followup":             False,
                "reasoning":               "Error - default to FAQ",
                "token_usage":             0,
            }


    def _parse_classification_response(self, response: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                parsed.setdefault("is_followup",            False)
                parsed.setdefault("contextualized_question","")
                parsed.setdefault("context_summary",        "")
                parsed.setdefault("agent",                  "FAQ")
                return parsed
            return {
                "agent": "FAQ", "is_followup": False,
                "contextualized_question": "", "context_summary": "",
            }
        except Exception as e:
            logger.error(f"Error parsing classification response: {e}")
            return {
                "agent": "FAQ", "is_followup": False,
                "contextualized_question": "", "context_summary": "",
            }

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return "Không có lịch sử"
        recent = history[-6:] if len(history) > 6 else history
        lines  = []
        for msg in recent:
            if isinstance(msg, dict):
                role    = "Người dùng" if msg.get("role") == "user" else "Trợ lý"
                content = msg.get("content", "")[:200]
            else:
                role    = "Người dùng" if getattr(msg, "role", "") == "user" else "Trợ lý"
                content = getattr(msg, "content", "")[:200]
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "Không có lịch sử"
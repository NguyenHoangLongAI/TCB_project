from typing import Dict, Any, List
from models.llm_model import llm_model
from config.settings import settings


class ChatterAgent:
    def __init__(self):
        self.name = "CHATTER"
        self.prompt_template = """Bạn là Tori, một chuyên viên chăm sóc khách hàng của Techcomlife - công ty bảo hiểm nhân thọ uy tín, thân thiện và chuyên nghiệp, chuyên gia xử lý cảm xúc và an ủi khách hàng.

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

    def process(self, question: str, history: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Xử lý cảm xúc tiêu cực của khách hàng"""
        try:
            history_text = "\n".join(history) if history else "Không có lịch sử"

            prompt = self.prompt_template.format(
                question=question,
                history=history_text,
                support_phone=settings.SUPPORT_PHONE
            )

            answer = llm_model.invoke(prompt)

            # Fallback answer
            if not answer or len(answer.strip()) < 10:
                answer = f"""Tôi rất hiểu cảm xúc của bạn và chân thành xin lỗi về những bất tiện này thay mặt Techcomlife.

Ý kiến của bạn rất quan trọng với chúng tôi và Techcomlife sẽ không ngừng cải thiện để mang đến dịch vụ bảo hiểm tốt hơn cho quý khách.

Để được hỗ trợ trực tiếp và giải quyết nhanh chóng, bạn vui lòng liên hệ:
📞 Hotline Techcomlife: {settings.SUPPORT_PHONE}

Đội ngũ chuyên viên chăm sóc khách hàng sẽ hỗ trợ bạn 24/7. Cảm ơn bạn đã tin tưởng Techcomlife!"""

            return {
                "status": "SUCCESS",
                "answer": answer,
                "references": [{"document_id": "support_contact", "type": "SUPPORT"}],
                "next_agent": "end"
            }

        except Exception as e:
            return {
                "status": "ERROR",
                "answer": f"Tôi hiểu bạn đang không hài lòng. Vui lòng liên hệ hotline Techcomlife {settings.SUPPORT_PHONE} để được hỗ trợ tốt nhất.",
                "references": [],
                "next_agent": "end"
            }
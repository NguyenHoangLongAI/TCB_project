from typing import Dict, Any, List
from models.llm_model import llm_model
from config.settings import settings


class OtherAgent:
    def __init__(self):
        self.name = "OTHER"
        self.prompt_template = """Bạn là Tori, một chuyên viên chăm sóc khách hàng của Techcomlife - công ty bảo hiểm nhân thọ uy tín, thân thiện và chuyên nghiệp, chuyên xử lý các yêu cầu ngoài phạm vi hỗ trợ.

Nhiệm vụ: Thông báo lịch sự khi yêu cầu nằm ngoài phạm vi tư vấn bảo hiểm và dịch vụ Techcomlife, đồng thời hướng dẫn khách hàng.

Yêu cầu của khách hàng: "{question}"
Số điện thoại hỗ trợ: {support_phone}

Hướng dẫn:
1. Giải thích rằng yêu cầu nằm ngoài phạm vi tư vấn bảo hiểm và dịch vụ Techcomlife hiện tại
2. Đề xuất liên hệ hotline Techcomlife để được tư vấn cụ thể hơn về các sản phẩm và dịch vụ phù hợp
3. Giữ thái độ lịch sự và chuyên nghiệp theo chuẩn mực Techcomlife
4. Không từ chối một cách thô lỗ

Trả lời:"""

    def process(self, question: str, **kwargs) -> Dict[str, Any]:
        """Xử lý yêu cầu ngoài phạm vi hỗ trợ"""
        try:
            prompt = self.prompt_template.format(
                question=question,
                support_phone=settings.SUPPORT_PHONE
            )

            answer = llm_model.invoke(prompt)

            # Fallback answer
            if not answer or len(answer.strip()) < 15:
                answer = f"""Cảm ơn bạn đã liên hệ với Techcomlife!

Yêu cầu của bạn có vẻ nằm ngoài phạm vi tư vấn bảo hiểm và dịch vụ Techcomlife mà tôi có thể hỗ trợ hiện tại.

Để được tư vấn và hỗ trợ tốt nhất về các sản phẩm bảo hiểm và dịch vụ của Techcomlife, bạn vui lòng:
📞 Liên hệ hotline Techcomlife: {settings.SUPPORT_PHONE}
⏰ Thời gian: 24/7

Đội ngũ chuyên viên Techcomlife sẽ hỗ trợ bạn một cách chuyên nghiệp nhất!"""

            return {
                "status": "SUCCESS",
                "answer": answer,
                "references": [],
                "next_agent": "end"
            }

        except Exception as e:
            return {
                "status": "ERROR",
                "answer": f"Đây không phải là tác vụ tôi có thể hỗ trợ. Vui lòng liên hệ hotline Techcomlife {settings.SUPPORT_PHONE} để được tư vấn.",
                "references": [],
                "next_agent": "end"
            }
from typing import Dict, Any, List
from models.llm_model import llm_model
from config.settings import settings


class NotEnoughInfoAgent:
    def __init__(self):
        self.name = "NOT_ENOUGH_INFO"

        self.prompt_template = """Bạn là Tori, chuyên viên chăm sóc khách hàng của Techcomlife - công ty bảo hiểm nhân thọ uy tín.

        Câu hỏi: "{question}"

        YÊU CẦU BẮT BUỘC:
        - Xin lỗi khách hàng vì chưa có thông tin để trả lời câu hỏi này
        - KHÔNG giải thích thêm bất kỳ thông tin gì về chủ đề được hỏi
        - KHÔNG suy đoán, KHÔNG cung cấp thông tin tham khảo
        - Đề nghị khách hàng liên hệ hotline Techcomlife {support_phone} để được hỗ trợ chính xác
        - Giữ thái độ lịch sự, chuyên nghiệp

        Chỉ trả về nội dung trả lời, không giải thích gì thêm.
        """

    def process(self, question: str, **kwargs) -> Dict[str, Any]:
        """Xử lý trường hợp không đủ thông tin - xin lỗi và hướng dẫn liên hệ"""
        try:
            prompt = self.prompt_template.format(
                question=question,
                support_phone=settings.SUPPORT_PHONE
            )

            answer = llm_model.invoke(prompt)

            if not answer or len(answer.strip()) < 10:
                answer = f"Xin lỗi, hiện tại chúng tôi chưa có thông tin để trả lời câu hỏi này. Vui lòng liên hệ hotline Techcomlife {settings.SUPPORT_PHONE} để được hỗ trợ chính xác."

            return {
                "status": "SUCCESS",
                "answer": answer,
                "references": [],
                "next_agent": "end"
            }

        except Exception as e:
            return {
                "status": "ERROR",
                "answer": f"Xin lỗi, hiện tại chúng tôi chưa có thông tin để trả lời câu hỏi này. Vui lòng liên hệ hotline Techcomlife {settings.SUPPORT_PHONE} để được hỗ trợ.",
                "references": [],
                "next_agent": "end"
            }
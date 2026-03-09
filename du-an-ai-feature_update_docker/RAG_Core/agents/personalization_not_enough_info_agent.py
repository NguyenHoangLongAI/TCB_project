# RAG_Core/agents/personalization_not_enough_info_agent.py

from typing import Dict, Any, List, AsyncIterator
from models.llm_model import llm_model
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class PersonalizationNotEnoughInfoAgent:
    """
    NotEnoughInfoAgent với personalization - Cá nhân hóa câu trả lời khi không đủ thông tin

    Chức năng:
    - Trả lời dựa trên kiến thức chung khi không có đủ dữ liệu
    - Cá nhân hóa theo thông tin khách hàng (tên, chức danh, lĩnh vực)
    - Điều chỉnh tone phù hợp với vị trí khách hàng
    - Hỗ trợ streaming
    """

    def __init__(self):
        self.name = "PERSONALIZATION_NOT_ENOUGH_INFO"

        # Personalized prompt
        self.personalized_prompt = """Bạn là trợ lý ảo Onetouch - chuyên gia đào tạo kỹ năng số cho người dân và doanh nghiệp.

THÔNG TIN KHÁCH HÀNG:
- Tên: {customer_name}
- Giới thiệu: {customer_introduction}
- Phân tích: {customer_analysis}

TÌNH HUỐNG: Không có đủ dữ liệu trong hệ thống để trả lời chính xác câu hỏi này.

CÂU HỎI CỦA KHÁCH HÀNG: "{question}"

YÊU CẦU TRẢ LỜI:

1. **Xưng hô phù hợp**:
   - Nếu là Giám đốc/Tổng giám đốc/CEO: "Thưa Anh/Chị {customer_name}"
   - Nếu là Manager/Trưởng phòng: "Thưa Anh/Chị {customer_name}"
   - Nếu là nhân viên/cá nhân: "Bạn {customer_name}"
   - Nếu không rõ: "Thưa Anh/Chị {customer_name}"

2. **Cấu trúc câu trả lời** (BẮT BUỘC NGẮN GỌN):

    a) MỞ ĐẦU (1 câu và KẾT THÚC BẰNG XUỐNG DÒNG):
    "Thưa Anh/Chị {customer_name}, dựa trên tổng hợp từ các nguồn thông tin, bạn có thể tham khảo như sau:\n"

    b) NỘI DUNG CHÍNH:
    - Viết thành một đoạn riêng biệt.
    - KHÔNG nối liền với câu mở đầu.
    - KHÔNG gộp chung thành một dòng.
    - KHÔNG phân tích dài dòng.

    c) XUỐNG DÒNG và KẾT THÚC:
    "\nViệc lựa chọn giải pháp ứng dụng AI cần căn cứ vào thực tế hoạt động của từng Phòng/Ban, năng lực đội ngũ và yêu cầu bảo mật dữ liệu của đơn vị. Trợ lý học tập có thể hỗ trợ phân tích, giải thích chi tiết các nội dung còn vướng mắc trong quá trình học tập; đồng thời sẽ tư vấn chuyên sâu, sát thực tiễn hơn khi được cung cấp và nghiên cứu đầy đủ tài liệu, quy trình và dữ liệu liên quan của Phòng/Ban."

3. **Cá nhân hóa nội dung**:
   - Nếu biết lĩnh vực: Đưa ví dụ phù hợp (công nghệ, truyền thông, sản xuất...)
   - Điều chỉnh độ kỹ thuật theo vị trí:
     * Lãnh đạo → Tổng quan, chiến lược
     * Quản lý → Giải pháp thực tế
     * Nhân viên → Dễ hiểu, ứng dụng

4. **Tone phù hợp**:
   - Lãnh đạo cấp cao: Tôn trọng, chuyên nghiệp
   - Quản lý: Thân thiện, hỗ trợ
   - Nhân viên: Gần gũi, dễ tiếp cận

5. **QUY TẮC VỀ ĐƯỜNG LINK - BẮT BUỘC TUÂN THỦ**:
   - TUYỆT ĐỐI KHÔNG tự bịa đặt hoặc suy đoán bất kỳ đường link/URL nào.
   - CHỈ được cung cấp link nếu link đó được cung cấp rõ ràng trong dữ liệu hệ thống.
   - Nếu khách hàng hỏi về đường link mà hệ thống KHÔNG có: trả lời rằng hiện tại bạn không có thông tin về đường link này và xin lỗi vì chưa thể hỗ trợ được yêu cầu này.
   - Nếu có link hợp lệ từ hệ thống: trình bày dưới dạng Markdown để có thể click được, ví dụ: [Tên trang](https://example.com)

6. **YÊU CẦU ĐẶC BIỆT**:
   - NGẮN GỌN (tối đa 3-4 câu)
   - KHÔNG kể ví dụ dài
   - KHÔNG giải thích chi tiết
   - BẮT ĐẦU bằng lời xưng hô phù hợp

Hãy trả lời:"""

    def _analyze_customer_profile(
            self,
            customer_name: str,
            customer_introduction: str
    ) -> str:
        """
        Phân tích nhanh profile khách hàng
        """
        try:
            intro_lower = (customer_introduction or "").lower()

            # Detect title
            if any(x in intro_lower for x in ["tổng giám đốc", "tổng gd", "ceo"]):
                title = "Tổng giám đốc"
                seniority = "C-level"
                tone = "formal"
            elif any(x in intro_lower for x in ["giám đốc", "gd", "director"]):
                title = "Giám đốc"
                seniority = "C-level"
                tone = "formal"
            elif any(x in intro_lower for x in ["trưởng phòng", "tp", "manager"]):
                title = "Trưởng phòng"
                seniority = "Manager"
                tone = "professional"
            elif any(x in intro_lower for x in ["nhân viên", "nv", "staff"]):
                title = "Nhân viên"
                seniority = "Staff"
                tone = "friendly"
            else:
                title = "Quý khách"
                seniority = "Individual"
                tone = "professional"

            # Detect industry
            if any(x in intro_lower for x in ["công nghệ", "technology", "tech", "cntt"]):
                industry = "Công nghệ thông tin"
            elif any(x in intro_lower for x in ["truyền thông", "media", "marketing"]):
                industry = "Truyền thông & Marketing"
            elif any(x in intro_lower for x in ["sản xuất", "manufacturing"]):
                industry = "Sản xuất"
            else:
                industry = "Không xác định"

            return f"""
- Chức danh: {title}
- Cấp độ: {seniority}
- Lĩnh vực: {industry}
- Tone khuyến nghị: {tone}
"""
        except Exception as e:
            logger.error(f"Error analyzing profile: {e}")
            return "- Chức danh: Quý khách\n- Cấp độ: Individual\n- Tone: professional"

    def _build_prompt(
            self,
            question: str,
            customer_name: str,
            customer_introduction: str
    ) -> str:
        """Tạo prompt đã điền đầy đủ thông tin khách hàng"""
        customer_analysis = self._analyze_customer_profile(
            customer_name,
            customer_introduction
        )
        return self.personalized_prompt.format(
            customer_name=customer_name or "Quý khách",
            customer_introduction=customer_introduction or "Không có thông tin",
            customer_analysis=customer_analysis,
            question=question
        )

    def _fallback_message(self, customer_name: str) -> str:
        """
        Trả về fallback message chuẩn.
        - Không bịa link, không đề cập hotline.
        - Chỉ thêm link nếu settings.SUPPORT_URL tồn tại.
        """
        greeting = f"Thưa Anh/Chị {customer_name}" if customer_name else "Xin chào"

        support_url_part = ""
        if getattr(settings, "SUPPORT_URL", None):
            support_url_part = f"\n\nBạn có thể tham khảo thêm tại [Trang hỗ trợ]({settings.SUPPORT_URL})."

        return (
            f"{greeting},\n\n"
            f"Xin lỗi, hệ thống gặp lỗi khi xử lý câu hỏi của bạn."
            f"{support_url_part}\n\n"
            f"Cảm ơn bạn!"
        )

    def process(
            self,
            question: str,
            customer_name: str = "",
            customer_introduction: str = "",
            **kwargs
    ) -> Dict[str, Any]:
        """
        Non-streaming process với personalization

        Args:
            question: Câu hỏi
            customer_name: Tên khách hàng
            customer_introduction: Giới thiệu về khách hàng
        """
        try:
            logger.info("🎭 Personalized Not Enough Info (non-streaming)")
            logger.info(f"   Customer: {customer_name}")

            prompt = self._build_prompt(question, customer_name, customer_introduction)

            logger.info("🤖 Generating personalized answer (not enough info)...")

            answer = llm_model.invoke(
                prompt,
                temperature=0.2,
                top_p=0.7,
                max_tokens=150,
                frequency_penalty=0.5,
                presence_penalty=0.0
            )

            logger.info("✅ Personalized answer generated")

            return {
                "status": "SUCCESS",
                "answer": answer,
                "references": [
                    {
                        "document_id": "llm_knowledge",
                        "type": "GENERAL_KNOWLEDGE"
                    }
                ],
                "personalized": True,
                "customer_name": customer_name,
                "next_agent": "end"
            }

        except Exception as e:
            logger.error(f"❌ Personalized Not Enough Info error: {e}")

            return {
                "status": "ERROR",
                "answer": self._fallback_message(customer_name),
                "references": [],
                "personalized": bool(customer_name),
                "next_agent": "end"
            }

    async def process_streaming(
            self,
            question: str,
            customer_name: str = "",
            customer_introduction: str = "",
            **kwargs
    ) -> AsyncIterator[str]:
        """
        Streaming process với personalization

        Args:
            question: Câu hỏi
            customer_name: Tên khách hàng
            customer_introduction: Giới thiệu về khách hàng
        """
        try:
            logger.info("🎭 Personalized Not Enough Info streaming")
            logger.info(f"   Customer: {customer_name}")

            prompt = self._build_prompt(question, customer_name, customer_introduction)

            logger.info("🚀 Streaming personalized answer...")

            chunk_count = 0
            async for chunk in llm_model.astream(prompt):
                if chunk:
                    chunk_count += 1
                    logger.debug(f"Not Enough Info chunk #{chunk_count}: {chunk[:30]}...")
                    yield chunk

            logger.info(f"✅ Not Enough Info streaming completed: {chunk_count} chunks")

        except Exception as e:
            logger.error(f"❌ Streaming error: {e}", exc_info=True)
            yield self._fallback_message(customer_name)
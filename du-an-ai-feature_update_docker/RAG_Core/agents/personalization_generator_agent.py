# RAG_Core/agents/personalization_generator_agent.py

from typing import Dict, Any, List, AsyncIterator
from models.llm_model import llm_model
import logging

logger = logging.getLogger(__name__)


class PersonalizationGeneratorAgent:
    """
    Generator Agent với personalization tích hợp sẵn

    Chức năng:
    - Tạo câu trả lời từ documents
    - Tự động cá nhân hóa dựa trên thông tin khách hàng
    - Hỗ trợ streaming
    """

    def __init__(self):
        self.name = "PERSONALIZATION_GENERATOR"

        # Standard prompt với personalization
        self.personalized_standard_prompt = """Bạn là trợ lý ảo Onetouch - chuyên gia đào tạo kỹ năng số cho người dân và doanh nghiệp.

THÔNG TIN KHÁCH HÀNG:
- Tên: {customer_name}
- Giới thiệu: {customer_introduction}
- Phân tích: {customer_analysis}

CÂU HỎI CỦA KHÁCH HÀNG: "{question}"

THÔNG TIN TÀI LIỆU THAM KHẢO:
{documents}

LỊCH SỬ HỘI THOẠI GẦN NHẤT:
{history}

YÊU CẦU TRẢ LỜI:

1. **Xưng hô phù hợp**:
   - Nếu là Giám đốc/Tổng giám đốc/CEO: "Thưa Anh/Chị {customer_name}"
   - Nếu là Manager/Trưởng phòng: "Anh/Chị {customer_name}"
   - Nếu là nhân viên/cá nhân: "Bạn {customer_name}"
   - Nếu không rõ: "Anh/Chị {customer_name}"

2. **Cá nhân hóa nội dung**:
   - Liên kết câu trả lời với lĩnh vực/ngành nghề của khách hàng
   - Đưa ra ví dụ phù hợp với context công ty/vai trò
   - Điều chỉnh độ chuyên sâu dựa trên vị trí:
     * Lãnh đạo → Tổng quan chiến lược, tầm nhìn
     * Quản lý → Giải pháp cụ thể, thực tiễn
     * Nhân viên → Chi tiết thực hành, dễ hiểu

3. **Tone phù hợp**:
   - Lãnh đạo cấp cao: Tôn trọng, tư vấn chiến lược
   - Quản lý: Chuyên nghiệp, giải pháp cụ thể
   - Nhân viên: Thân thiện, hướng dẫn chi tiết

4. **Nội dung**:
   - Trả lời bằng giọng văn tự nhiên như người Việt Nam nói chuyện
   - Trả lời thẳng vào vấn đề, ngắn gọn súc tích
   - Dựa vào thông tin tài liệu nhưng diễn đạt theo cách hiểu của bạn
   - Kết thúc bằng câu hỏi ngắn để tiếp tục hỗ trợ nếu cần

5. **Định dạng**:
   - BẮT ĐẦU bằng lời xưng hô phù hợp
   - Nội dung chính với personalization
   - KẾT THÚC bằng câu hỏi mở

Hãy trả lời như đang nói chuyện trực tiếp với khách hàng:"""

        # Follow-up prompt với personalization
        self.personalized_followup_prompt = """Bạn là trợ lý ảo Onetouch - chuyên gia đào tạo kỹ năng số cho người dân và doanh nghiệp.

THÔNG TIN KHÁCH HÀNG:
- Tên: {customer_name}
- Giới thiệu: {customer_introduction}
- Phân tích: {customer_analysis}

NGỮ CẢNH CUỘC TRÒ CHUYỆN:
{context_summary}

LỊCH SỬ GẦN NHẤT:
{recent_history}

CÂU HỎI FOLLOW-UP: "{question}"

THÔNG TIN TÀI LIỆU LIÊN QUAN:
{documents}

YÊU CẦU ĐẶC BIỆT CHO FOLLOW-UP:

1. **Xưng hô nhất quán** với câu trả lời trước (Thưa Anh/Chị {customer_name})

2. **Tham chiếu tự nhiên**:
   - Nhận biết rằng khách hàng đang hỏi tiếp về chủ đề đã thảo luận
   - Tham chiếu đến thông tin đã cung cấp trước đó
   - Trả lời cụ thể vào phần mà khách hàng muốn biết thêm

3. **KHÔNG lặp lại** toàn bộ thông tin đã nói, chỉ tập trung vào phần được hỏi

4. **Cá nhân hóa tiếp tục**:
   - Duy trì tone phù hợp với vị trí khách hàng
   - Liên kết với ngành nghề/lĩnh vực của họ

5. **Nội dung**:
   - Trả lời ngắn gọn, đúng trọng tâm
   - Kết thúc bằng câu hỏi để tiếp tục hỗ trợ

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

    def _format_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents"""
        if not documents:
            return "Không có tài liệu tham khảo"

        doc_lines = []
        for i, doc in enumerate(documents[:5], 1):
            description = doc.get('description', '')
            score = doc.get('similarity_score', 0)
            doc_lines.append(f"[Tài liệu {i}] (Độ liên quan: {score:.2f})\n{description}")

        return "\n\n".join(doc_lines)

    def _format_history(self, history: List, max_turns: int = 2) -> str:
        """Format history"""
        if not history:
            return "Không có lịch sử"

        normalized_history = []
        for msg in history:
            if isinstance(msg, dict):
                normalized_history.append({
                    "role": msg.get("role", ""),
                    "content": msg.get("content", "")
                })
            else:
                normalized_history.append({
                    "role": getattr(msg, "role", ""),
                    "content": getattr(msg, "content", "")
                })

        recent = normalized_history[-(max_turns * 2):] if len(
            normalized_history) > max_turns * 2 else normalized_history

        history_lines = []
        for msg in recent:
            role = "👤 Khách hàng" if msg.get("role") == "user" else "🤖 Trợ lý"
            content = msg.get("content", "")
            if content:
                history_lines.append(f"{role}: {content}")

        return "\n".join(history_lines) if history_lines else "Không có lịch sử"

    def _extract_context_summary(self, history: List) -> str:
        """Extract context summary"""
        if not history or len(history) < 2:
            return "Đây là câu hỏi đầu tiên"

        normalized_history = []
        for msg in history:
            if isinstance(msg, dict):
                normalized_history.append(msg)
            else:
                normalized_history.append({
                    "role": getattr(msg, "role", ""),
                    "content": getattr(msg, "content", "")
                })

        for i in range(len(normalized_history) - 1, -1, -1):
            if normalized_history[i].get("role") == "user":
                prev_question = normalized_history[i].get("content", "")

                for j in range(i + 1, len(normalized_history)):
                    if normalized_history[j].get("role") == "assistant":
                        prev_answer = normalized_history[j].get("content", "")
                        return f"Chủ đề đang thảo luận: {prev_question}\nĐã trả lời: {prev_answer[:200]}..."

                return f"Chủ đề đang thảo luận: {prev_question}"

        return "Đang trong cuộc trò chuyện"

    def process(
            self,
            question: str,
            documents: List[Dict[str, Any]],
            customer_name: str = "",
            customer_introduction: str = "",
            references: List[Dict[str, Any]] = None,
            history: List[Dict[str, str]] = None,
            is_followup: bool = False,
            context_summary: str = "",
            **kwargs
    ) -> Dict[str, Any]:
        """
        Non-streaming generation với personalization
        """
        try:
            logger.info("🎭 Personalization Generator (non-streaming)")
            logger.info(f"   Customer: {customer_name}")
            logger.info(f"   Follow-up: {is_followup}")

            if not documents:
                return {
                    "status": "ERROR",
                    "answer": "Không có tài liệu để tạo câu trả lời",
                    "references": [],
                    "next_agent": "end"
                }

            # Analyze customer
            customer_analysis = self._analyze_customer_profile(
                customer_name,
                customer_introduction
            )

            # Format inputs
            doc_text = self._format_documents(documents)
            history_text = self._format_history(history or [], max_turns=2)

            # Choose prompt
            if is_followup:
                if not context_summary:
                    context_summary = self._extract_context_summary(history or [])

                prompt = self.personalized_followup_prompt.format(
                    customer_name=customer_name or "Quý khách",
                    customer_introduction=customer_introduction or "Không có thông tin",
                    customer_analysis=customer_analysis,
                    question=question,
                    context_summary=context_summary,
                    recent_history=history_text,
                    documents=doc_text
                )
            else:
                prompt = self.personalized_standard_prompt.format(
                    customer_name=customer_name or "Quý khách",
                    customer_introduction=customer_introduction or "Không có thông tin",
                    customer_analysis=customer_analysis,
                    question=question,
                    history=history_text,
                    documents=doc_text
                )

            # Generate answer
            answer = llm_model.invoke(prompt)

            if not answer or len(answer.strip()) < 10:
                answer = "Tôi đã tìm thấy thông tin liên quan nhưng gặp khó khăn trong việc tạo câu trả lời."

            logger.info("✅ Personalized answer generated")

            return {
                "status": "SUCCESS",
                "answer": answer,
                "references": references or [],
                "personalized": True,
                "customer_name": customer_name,
                "next_agent": "end"
            }

        except Exception as e:
            logger.error(f"❌ Generator error: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "answer": f"Lỗi tạo câu trả lời: {str(e)}",
                "references": [],
                "next_agent": "end"
            }

    async def process_streaming(
            self,
            question: str,
            documents: List[Dict[str, Any]],
            customer_name: str = "",
            customer_introduction: str = "",
            references: List[Dict[str, Any]] = None,
            history: List[Dict[str, str]] = None,
            is_followup: bool = False,
            context_summary: str = "",
            **kwargs
    ) -> AsyncIterator[str]:
        """
        Streaming generation với personalization
        """
        try:
            logger.info("🎭 Personalization Generator streaming")
            logger.info(f"   Customer: {customer_name}")
            logger.info(f"   Follow-up: {is_followup}")

            if not documents:
                yield "Không có tài liệu để tạo câu trả lời."
                return

            # Analyze customer
            customer_analysis = self._analyze_customer_profile(
                customer_name,
                customer_introduction
            )

            # Format inputs
            doc_text = self._format_documents(documents)
            history_text = self._format_history(history or [], max_turns=2)

            # Choose prompt
            if is_followup:
                if not context_summary:
                    context_summary = self._extract_context_summary(history or [])

                prompt = self.personalized_followup_prompt.format(
                    customer_name=customer_name or "Quý khách",
                    customer_introduction=customer_introduction or "Không có thông tin",
                    customer_analysis=customer_analysis,
                    question=question,
                    context_summary=context_summary,
                    recent_history=history_text,
                    documents=doc_text
                )
            else:
                prompt = self.personalized_standard_prompt.format(
                    customer_name=customer_name or "Quý khách",
                    customer_introduction=customer_introduction or "Không có thông tin",
                    customer_analysis=customer_analysis,
                    question=question,
                    history=history_text,
                    documents=doc_text
                )

            logger.info("🚀 Streaming personalized answer...")

            # Stream from LLM
            chunk_count = 0
            async for chunk in llm_model.astream(prompt):
                if chunk:
                    chunk_count += 1
                    logger.debug(f"Generator chunk #{chunk_count}: {chunk[:30]}...")
                    yield chunk

            logger.info(f"✅ Generator streaming completed: {chunk_count} chunks")

        except Exception as e:
            logger.error(f"❌ Generator streaming error: {e}", exc_info=True)
            yield f"\n\n[Lỗi: {str(e)}]"
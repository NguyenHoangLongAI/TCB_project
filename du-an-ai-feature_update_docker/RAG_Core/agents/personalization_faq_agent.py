# RAG_Core/agents/personalization_faq_agent.py - UPDATED TO USE PERSONALIZATION DB

from typing import Dict, Any, List, AsyncIterator
from models.llm_model import llm_model
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class PersonalizationFAQAgent:
    """
    FAQ Agent với personalization tích hợp sẵn
    UPDATED: Sử dụng personalization_milvus_client

    Chức năng:
    - Tìm kiếm FAQ phù hợp từ personalization database
    - Tự động cá nhân hóa câu trả lời dựa trên thông tin khách hàng
    - Hỗ trợ streaming
    """

    def __init__(self):
        self.name = "PERSONALIZATION_FAQ"

        # Thresholds
        self.vector_threshold = 0.5
        self.rerank_threshold = 0.6

        # Prompt với personalization tích hợp
        self.personalized_prompt = """Bạn là trợ lý ảo Onetouch - chuyên gia đào tạo kỹ năng số cho người dân và doanh nghiệp.

THÔNG TIN KHÁCH HÀNG:
- Tên: {customer_name}
- Giới thiệu: {customer_introduction}
- Phân tích: {customer_analysis}

CÂU HỎI CỦA KHÁCH HÀNG: "{question}"

KẾT QUẢ TÌM KIẾM FAQ (đã được rerank):
{faq_results}

NHIỆM VỤ CỦA BẠN:
1. **Xưng hô phù hợp**:
   - Nếu là Giám đốc/Tổng giám đốc/CEO: "Thưa Anh/Chị {customer_name}"
   - Nếu là Manager/Trưởng phòng: "Anh/Chị {customer_name}"
   - Nếu là nhân viên/cá nhân: "Bạn {customer_name}"
   - Nếu không rõ: "Anh/Chị {customer_name}"

2. **Trả lời dựa vào FAQ có rerank_score cao nhất**
   - Nếu không có FAQ phù hợp (tất cả score quá thấp), trả về "NOT_FOUND"
   - Nếu có FAQ phù hợp, dùng nội dung đó để trả lời

3. **Cá nhân hóa nội dung**:
   - Liên kết với lĩnh vực/ngành nghề của khách hàng
   - Đưa ra ví dụ phù hợp với context công ty/vai trò
   - Điều chỉnh tone phù hợp với vị trí (lãnh đạo → chiến lược, nhân viên → thực hành)

4. **Tone phù hợp**:
   - Lãnh đạo cấp cao: Tôn trọng, tư vấn chiến lược
   - Quản lý: Chuyên nghiệp, giải pháp cụ thể
   - Nhân viên: Thân thiện, hướng dẫn chi tiết

5. **Kết thúc**: Câu hỏi mở để tiếp tục hỗ trợ

YÊU CẦU QUAN TRỌNG:
- BẮT ĐẦU bằng lời xưng hô phù hợp
- Trả lời bằng tiếng Việt tự nhiên, thân thiện
- KHÔNG nói "Dựa vào FAQ..." - trả lời như bạn biết
- Giữ nguyên thông tin chính xác từ FAQ

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

    def process(
            self,
            question: str,
            customer_name: str = "",
            customer_introduction: str = "",
            is_followup: bool = False,
            context: str = "",
            **kwargs
    ) -> Dict[str, Any]:
        """
        Non-streaming process với personalization
        UPDATED: Sử dụng search_personalization_faq
        """
        try:
            logger.info("=" * 50)
            logger.info("🎭 PERSONALIZATION FAQ AGENT (Personalization DB)")
            logger.info("=" * 50)
            logger.info(f"📝 Question: '{question[:100]}'")
            logger.info(f"👤 Customer: {customer_name}")

            # ✅ IMPORT PERSONALIZATION TOOLS
            from tools.vector_search import search_personalization_faq, rerank_faq

            # Vector search trong PERSONALIZATION DB
            faq_results = search_personalization_faq.invoke({"query": question})

            if not faq_results or "error" in str(faq_results):
                logger.warning("❌ Personalization vector search failed")
                return self._route_to_retriever("Personalization vector search failed")

            # Filter by threshold
            filtered_faqs = [
                faq for faq in faq_results
                if faq.get("similarity_score", 0) >= self.vector_threshold
            ]

            if not filtered_faqs:
                logger.info(f"⚠️  No FAQ passed vector threshold {self.vector_threshold}")
                return self._route_to_retriever("No FAQ above threshold")

            logger.info(f"✅ Found {len(filtered_faqs)} FAQs above threshold (personalization DB)")

            # Rerank (using standard Cohere reranker)
            logger.info(f"🎯 Reranking with Cohere")
            reranked_faqs = rerank_faq.invoke({
                "query": question,
                "faq_results": filtered_faqs
            })

            if not reranked_faqs:
                logger.error("❌ Reranking returned empty results")
                return self._route_to_retriever("Rerank failed")

            best_faq = reranked_faqs[0]
            rerank_score = best_faq.get("rerank_score", 0)

            logger.info(f"📊 Best FAQ rerank score: {rerank_score:.3f}")

            # Check confidence
            if rerank_score < self.rerank_threshold:
                logger.info(f"⚠️  Not confident: {rerank_score:.3f} < {self.rerank_threshold}")
                return self._route_to_retriever(f"Rerank score too low: {rerank_score:.3f}")

            # Analyze customer profile
            customer_analysis = self._analyze_customer_profile(
                customer_name,
                customer_introduction
            )

            # Format FAQ results
            faq_text = self._format_reranked_faq(reranked_faqs[:3])

            # Create personalized prompt
            prompt = self.personalized_prompt.format(
                customer_name=customer_name or "Quý khách",
                customer_introduction=customer_introduction or "Không có thông tin",
                customer_analysis=customer_analysis,
                question=question,
                faq_results=faq_text
            )

            # Generate personalized answer
            logger.info(f"🤖 Generating personalized FAQ answer")
            response = llm_model.invoke(prompt)

            if "NOT_FOUND" in response.upper():
                logger.info("🔄 LLM determined FAQ not sufficient → RETRIEVER")
                return self._route_to_retriever("LLM rejected FAQ")

            if not response or len(response.strip()) < 10:
                logger.warning("⚠️  Generated answer too short → RETRIEVER")
                return self._route_to_retriever("Answer too short")

            logger.info(f"✅ Personalized FAQ answer generated (from personalization DB)")
            logger.info("=" * 50 + "\n")

            return {
                "status": "SUCCESS",
                "answer": response,
                "mode": "personalized_faq",
                "references": [
                    {
                        "document_id": best_faq.get("faq_id"),
                        "type": "FAQ",
                        "description": best_faq.get("question", ""),
                        "rerank_score": round(rerank_score, 4),
                        "source": "personalization_db"
                    }
                ],
                "personalized": True,
                "customer_name": customer_name,
                "next_agent": "end"
            }

        except Exception as e:
            logger.error(f"❌ Personalization FAQ error: {e}", exc_info=True)
            raise RuntimeError(f"Personalization FAQ failed: {e}") from e

    async def process_streaming(
            self,
            question: str,
            reranked_faqs: List[Dict[str, Any]],
            customer_name: str = "",
            customer_introduction: str = "",
            is_followup: bool = False,
            context: str = "",
            **kwargs
    ) -> AsyncIterator[str]:
        """
        Streaming với personalization
        NOTE: reranked_faqs should come from personalization DB
        """
        try:
            logger.info("🎭 Personalization FAQ streaming (personalization DB)")
            logger.info(f"   Customer: {customer_name}")

            if not reranked_faqs:
                yield "Không tìm thấy câu trả lời phù hợp."
                return

            # Analyze customer
            customer_analysis = self._analyze_customer_profile(
                customer_name,
                customer_introduction
            )

            # Format FAQs
            faq_text = self._format_reranked_faq(reranked_faqs[:3])

            # Create prompt
            prompt = self.personalized_prompt.format(
                customer_name=customer_name or "Quý khách",
                customer_introduction=customer_introduction or "Không có thông tin",
                customer_analysis=customer_analysis,
                question=question,
                faq_results=faq_text
            )

            logger.info("🚀 Streaming personalized FAQ answer...")

            # Stream from LLM
            chunk_count = 0
            async for chunk in llm_model.astream(prompt):
                if chunk:
                    chunk_count += 1
                    logger.debug(f"FAQ chunk #{chunk_count}: {chunk[:30]}...")
                    yield chunk

            logger.info(f"✅ FAQ streaming completed: {chunk_count} chunks")

        except Exception as e:
            logger.error(f"❌ FAQ streaming error: {e}", exc_info=True)
            yield f"\n\n[Lỗi FAQ: {str(e)}]"

    def _format_reranked_faq(self, faq_results: List[Dict[str, Any]]) -> str:
        """Format FAQ results"""
        if not faq_results:
            return "Không tìm thấy FAQ phù hợp"

        formatted_lines = []
        for i, faq in enumerate(faq_results, 1):
            question = faq.get('question', '')
            answer = faq.get('answer', '')
            rerank_score = faq.get('rerank_score', 0)

            formatted_lines.append(
                f"FAQ {i} (Rerank: {rerank_score:.3f}):\n"
                f"Q: {question}\n"
                f"A: {answer}\n"
            )

        return "\n".join(formatted_lines)

    def _route_to_retriever(self, reason: str) -> Dict[str, Any]:
        """Route to retriever"""
        logger.info(f"→ Routing to RETRIEVER: {reason}")
        return {
            "status": "NOT_FOUND",
            "answer": "",
            "references": [],
            "next_agent": "RETRIEVER"
        }
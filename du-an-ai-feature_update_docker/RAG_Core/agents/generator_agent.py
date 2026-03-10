# RAG_Core/agents/generator_agent.py - FIXED STREAMING VERSION

from typing import Dict, Any, List, AsyncIterator
from models.llm_model import llm_model
import logging

logger = logging.getLogger(__name__)


class GeneratorAgent:
    def __init__(self):
        self.name = "GENERATOR"

        self.standard_prompt = """Bạn là một chuyên viên chăm sóc khách hàng của Techcomlife - công ty bảo hiểm nhân thọ uy tín, thân thiện và chuyên nghiệp.

Câu hỏi của khách hàng: "{question}"

Thông tin tham khảo từ tài liệu Techcomlife:
{documents}

Lịch sử trò chuyện gần đây:
{history}

Yêu cầu trả lời:
- Trả lời bằng giọng văn tự nhiên như người Việt Nam nói chuyện
- Trả lời thẳng vào vấn đề, ngắn gọn súc tích
- Dựa vào thông tin tài liệu Techcomlife nhưng diễn đạt theo cách hiểu của bạn
- Đảm bảo thông tin về quyền lợi bảo hiểm, phí, thủ tục được trình bày rõ ràng, chính xác
- Kết thúc bằng câu hỏi ngắn để tiếp tục hỗ trợ nếu cần

Hãy trả lời như đang tư vấn trực tiếp với khách hàng:"""

        self.followup_prompt = """Bạn là một chuyên viên chăm sóc khách hàng của Techcomlife - công ty bảo hiểm nhân thọ uy tín, thân thiện và chuyên nghiệp.

🔍 NGỮ CẢNH CUỘC TRÒ CHUYỆN:
{context_summary}

📝 LỊCH SỬ GẦN NHẤT:
{recent_history}

❓ CÂU HỎI FOLLOW-UP CỦA KHÁCH HÀNG: "{question}"

📚 THÔNG TIN TÀI LIỆU TECHCOMLIFE LIÊN QUAN:
{documents}

⚠️ YÊU CẦU ĐẶC BIỆT cho follow-up question:
1. Nhận biết rằng khách hàng đang hỏi tiếp về chủ đề đã thảo luận
2. Tham chiếu đến thông tin đã cung cấp trước đó một cách tự nhiên
3. Trả lời cụ thể vào phần mà khách hàng muốn biết thêm
4. KHÔNG lặp lại toàn bộ thông tin đã nói, chỉ tập trung vào phần được hỏi

📋 YÊU CẦU CHUNG:
- Trả lời bằng giọng văn tự nhiên như người Việt Nam nói chuyện
- Ngắn gọn, súc tích, đúng trọng tâm
- Đảm bảo thông tin bảo hiểm chính xác theo tài liệu Techcomlife
- Kết thúc bằng câu hỏi để tiếp tục hỗ trợ nếu cần

Hãy trả lời:"""

    def _deduplicate_references(self, references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Loại bỏ references trùng lặp"""
        if not references:
            return []

        seen_doc_ids = set()
        unique_references = []

        for ref in references:
            doc_id = ref.get('document_id')
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                unique_references.append(ref)

        return unique_references

    def _format_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents thành text"""
        if not documents:
            return "Không có tài liệu tham khảo"

        doc_lines = []
        for i, doc in enumerate(documents[:5], 1):
            description = doc.get('description', '')
            score = doc.get('similarity_score', 0)
            doc_lines.append(f"[Tài liệu {i}] (Độ liên quan: {score:.2f})\n{description}")

        return "\n\n".join(doc_lines)

    def _format_history(self, history: List, max_turns: int = 2) -> str:
        """Format lịch sử hội thoại"""
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

        recent_history = normalized_history[-(max_turns * 2):] if len(
            normalized_history) > max_turns * 2 else normalized_history

        history_lines = []
        for msg in recent_history:
            role = "👤 Khách hàng" if msg.get("role") == "user" else "🤖 Trợ lý Techcomlife"
            content = msg.get("content", "")
            if content:
                history_lines.append(f"{role}: {content}")

        return "\n".join(history_lines) if history_lines else "Không có lịch sử"

    def _extract_context_summary(self, history: List) -> str:
        """Trích xuất context summary"""
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
            references: List[Dict[str, Any]] = None,
            history: List[Dict[str, str]] = None,
            is_followup: bool = False,
            context_summary: str = "",
            **kwargs
    ) -> Dict[str, Any]:
        """Non-streaming generation (original)"""
        try:
            if not documents:
                return {
                    "status": "ERROR",
                    "answer": "Không có tài liệu để tạo câu trả lời",
                    "references": [],
                    "next_agent": "end"
                }

            doc_text = self._format_documents(documents)
            history_text = self._format_history(history or [], max_turns=2)

            if is_followup:
                if not context_summary:
                    context_summary = self._extract_context_summary(history or [])

                prompt = self.followup_prompt.format(
                    question=question,
                    context_summary=context_summary,
                    recent_history=history_text,
                    documents=doc_text
                )
            else:
                prompt = self.standard_prompt.format(
                    question=question,
                    history=history_text,
                    documents=doc_text
                )

            answer = llm_model.invoke(prompt)

            if not answer or len(answer.strip()) < 10:
                answer = "Tôi đã tìm thấy thông tin liên quan nhưng gặp khó khăn trong việc tạo câu trả lời. Vui lòng liên hệ hotline Techcomlife để được hỗ trợ trực tiếp."

            unique_references = self._deduplicate_references(references or [])

            return {
                "status": "SUCCESS",
                "answer": answer,
                "references": unique_references,
                "next_agent": "end"
            }

        except Exception as e:
            logger.error(f"Error in generator agent: {e}", exc_info=True)
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
            references: List[Dict[str, Any]] = None,
            history: List[Dict[str, str]] = None,
            is_followup: bool = False,
            context_summary: str = "",
            **kwargs
    ) -> AsyncIterator[str]:
        """
        FIXED: Streaming generation with proper async/await
        """
        try:
            logger.info(f"🚀 Generator: Starting streaming for: {question[:50]}...")

            if not documents:
                yield "Không có tài liệu để tạo câu trả lời."
                return

            doc_text = self._format_documents(documents)
            history_text = self._format_history(history or [], max_turns=2)

            if is_followup:
                if not context_summary:
                    context_summary = self._extract_context_summary(history or [])

                prompt = self.followup_prompt.format(
                    question=question,
                    context_summary=context_summary,
                    recent_history=history_text,
                    documents=doc_text
                )
            else:
                prompt = self.standard_prompt.format(
                    question=question,
                    history=history_text,
                    documents=doc_text
                )

            logger.info(f"📝 Generator: Prompt prepared, length={len(prompt)}")

            chunk_count = 0
            async for chunk in llm_model.astream(prompt):
                if chunk:
                    chunk_count += 1
                    logger.debug(f"Generator yielding chunk #{chunk_count}: {chunk[:30]}...")
                    yield chunk

            logger.info(f"✅ Generator: Completed streaming {chunk_count} chunks")

        except Exception as e:
            logger.error(f"❌ Generator streaming error: {e}", exc_info=True)
            yield f"\n\n[Lỗi: {str(e)}]"
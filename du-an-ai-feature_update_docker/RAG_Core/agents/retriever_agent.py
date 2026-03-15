# RAG_Core/agents/retriever_agent.py  (OPTIMIZED – bỏ similarity filter trước rerank)
"""
Thay đổi:
  - Bỏ bước lọc SIMILARITY_THRESHOLD trước rerank.
  - Toàn bộ TOP_K=20 docs đi thẳng vào Grader → Cohere rerank quyết định.
  - Lý do: vietnamese-sbert COSINE score thấp (0.2-0.45) kể cả khi match tốt.
    Lọc 0.5 sẽ loại bỏ docs đúng trước khi Cohere kịp đánh giá.
  - Chỉ log để debug, không bỏ docs nào ở bước này.
"""

from typing import Dict, Any, Optional
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class RetrieverAgent:
    def __init__(self):
        self.name = "RETRIEVER"

    def process(
        self,
        question:                str,
        contextualized_question: str = "",
        is_followup:             bool = False,
        user_id:                 Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        try:
            search_query = (
                contextualized_question or question
                if (is_followup or contextualized_question)
                else question
            )

            if is_followup or contextualized_question:
                logger.info("🔍 Using CONTEXTUALIZED question for vector search")
            else:
                logger.info("🔍 Using ORIGINAL question for vector search")

            # ACL-aware search
            if user_id:
                from tools.vector_search import search_documents_for_user
                logger.info(f"🔒 ACL search: user_id={user_id}")
                search_results = search_documents_for_user.invoke({
                    "query":   search_query,
                    "user_id": user_id,
                })
            else:
                from tools.vector_search import search_documents
                logger.info("🔓 Open search (no user_id)")
                search_results = search_documents.invoke({"query": search_query})

            if not search_results or "error" in str(search_results):
                return {"status": "ERROR", "documents": [], "next_agent": "NOT_ENOUGH_INFO"}

            # Log phân phối score để monitor, KHÔNG lọc ở đây
            if search_results:
                scores = [d.get("similarity_score", 0) for d in search_results]
                logger.info(
                    f"📊 Vector scores — count={len(scores)} "
                    f"max={max(scores):.3f} min={min(scores):.3f} "
                    f"avg={sum(scores)/len(scores):.3f}"
                )

            # Toàn bộ docs đi vào Grader → Cohere rerank quyết định
            logger.info(f"✅ Passing {len(search_results)} docs to GRADER (no pre-filter)")
            return {
                "status":     "SUCCESS",
                "documents":  search_results,
                "next_agent": "GRADER",
            }

        except Exception as e:
            logger.error(f"❌ Retriever error: {e}", exc_info=True)
            return {"status": "ERROR", "documents": [], "next_agent": "REPORTER"}
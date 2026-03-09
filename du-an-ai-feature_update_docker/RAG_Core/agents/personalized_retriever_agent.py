# RAG_Core/agents/personalized_retriever_agent.py - NEW FILE

from typing import Dict, Any, List
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class PersonalizedRetrieverAgent:
    """
    Retriever Agent cho personalization database
    """

    def __init__(self):
        self.name = "PERSONALIZED_RETRIEVER"

    def process(
            self,
            question: str,
            contextualized_question: str = "",
            is_followup: bool = False,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Tìm kiếm tài liệu trong personalization database

        Args:
            question: Câu hỏi gốc (for logging)
            contextualized_question: Câu hỏi đã được làm rõ (dùng để search)
            is_followup: Có phải follow-up question không
        """
        try:
            # ✅ IMPORT PERSONALIZATION TOOL
            from tools.vector_search import search_personalization_documents

            # Quyết định query cho vector search
            if is_followup or contextualized_question:
                search_query = contextualized_question
                logger.info(f"🔍 Using CONTEXTUALIZED QUESTION for personalization search")
                logger.debug(f"Original: {question[:60]}")
                logger.debug(f"Contextualized: {contextualized_question[:100]}")
            else:
                search_query = question
                logger.info(f"🔍 Using ORIGINAL QUESTION for personalization search")

            logger.info(f"📚 Searching personalization documents with query: {search_query[:100]}...")

            # Tìm kiếm tài liệu trong personalization DB
            search_results = search_personalization_documents.invoke({"query": search_query})

            if not search_results or "error" in str(search_results):
                logger.warning("Personalization vector search failed or returned error")
                return {
                    "status": "ERROR",
                    "documents": [],
                    "next_agent": "NOT_ENOUGH_INFO"
                }

            # Lọc kết quả theo similarity threshold
            relevant_docs = [
                doc for doc in search_results
                if doc.get("similarity_score", 0) > settings.SIMILARITY_THRESHOLD
            ]

            if not relevant_docs:
                logger.info(
                    f"No personalization documents above threshold {settings.SIMILARITY_THRESHOLD}, "
                    f"returning all {len(search_results)} for grader"
                )
                return {
                    "status": "NOT_FOUND",
                    "documents": search_results,
                    "search_query_used": "contextualized" if (is_followup and contextualized_question) else "original",
                    "source": "personalization_db",
                    "next_agent": "GRADER"
                }

            logger.info(
                f"✅ Found {len(relevant_docs)} relevant personalization documents "
                f"(searched with {'contextualized question' if is_followup and contextualized_question else 'original question'})"
            )

            return {
                "status": "SUCCESS",
                "documents": relevant_docs,
                "search_query_used": "contextualized" if (is_followup and contextualized_question) else "original",
                "source": "personalization_db",
                "next_agent": "GRADER"
            }

        except Exception as e:
            logger.error(f"❌ Personalized Retriever error: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "documents": [],
                "next_agent": "REPORTER"
            }
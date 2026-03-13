# RAG_Core/agents/retriever_agent.py  (UPDATED – ACL-aware via user_id)

from typing import Dict, Any, List, Optional
from models.llm_model import llm_model
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class RetrieverAgent:
    def __init__(self):
        self.name  = "RETRIEVER"

    def process(
        self,
        question:               str,
        contextualized_question: str = "",
        is_followup:            bool = False,
        user_id:                Optional[str] = None,   # NEW
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Search for relevant documents.
        When user_id is provided, uses ACL-filtered search.
        """
        try:
            # Decide which query to use
            if is_followup or contextualized_question:
                search_query = contextualized_question or question
                logger.info("🔍 Using CONTEXTUALIZED question for vector search")
            else:
                search_query = question
                logger.info("🔍 Using ORIGINAL question for vector search")

            # ── ACL-aware search ────────────────────────────────────────────
            if user_id:
                from tools.vector_search import search_documents_for_user
                logger.info(f"🔒 ACL-filtered search for user_id={user_id}")
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

            relevant_docs = [
                doc for doc in search_results
                if doc.get("similarity_score", 0) > settings.SIMILARITY_THRESHOLD
            ]

            if not relevant_docs:
                logger.info(f"No docs above threshold {settings.SIMILARITY_THRESHOLD}, passing all to GRADER")
                return {"status": "NOT_FOUND", "documents": search_results, "next_agent": "GRADER"}

            logger.info(f"✅ Found {len(relevant_docs)} relevant documents")
            return {"status": "SUCCESS", "documents": relevant_docs, "next_agent": "GRADER"}

        except Exception as e:
            logger.error(f"❌ Retriever error: {e}", exc_info=True)
            return {"status": "ERROR", "documents": [], "next_agent": "REPORTER"}
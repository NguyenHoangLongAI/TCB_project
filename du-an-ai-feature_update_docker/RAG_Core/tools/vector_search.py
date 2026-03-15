# RAG_Core/tools/vector_search.py  (OPTIMIZED – truncate docs trước Cohere rerank)
"""
Thay đổi:
  - rerank_documents(): truncate description xuống 512 ký tự trước khi gửi Cohere.
    Cohere rerank giới hạn ~512 token/doc — gửi full text bị cắt ngẫu nhiên,
    tốt hơn là tự cắt có kiểm soát (giữ phần đầu chứa context_header).
  - rerank_faq(): giữ nguyên vì FAQ thường ngắn.
  - search_documents / search_documents_for_user: không đổi.
"""

from langchain_core.tools import tool
from typing import List, Dict, Any, Optional
import numpy as np, os, logging

from models.embedding_model import embedding_model
from database.milvus_client import milvus_client, _get_user_db_manager
from config.settings import settings

logger = logging.getLogger(__name__)

# ─── Cohere setup ─────────────────────────────────────────────────────────────

cohere_client       = None
COHERE_RERANK_MODEL = "rerank-multilingual-v3.0"

try:
    import cohere
    cohere_api_key = (
        getattr(settings, "COHERE_API_KEY", None)
        or os.getenv("COHERE_API_KEY")
        or "NoQ9Jjvz5r1JeRWZG8L9dnl8BxYljmnOdiUfTnfk"
    )
    if cohere_api_key and cohere_api_key != "your-api-key-here":
        cohere_client = cohere.Client(cohere_api_key)
        if hasattr(settings, "COHERE_RERANK_MODEL"):
            COHERE_RERANK_MODEL = settings.COHERE_RERANK_MODEL
        logger.info(f"✅ Cohere initialized: {COHERE_RERANK_MODEL}")
except Exception as e:
    logger.error(f"Cohere init error: {e}")

# Số ký tự tối đa gửi lên Cohere mỗi document
# ~512 token × 4 ký tự/token ≈ 2000 ký tự — an toàn cho tiếng Việt
_COHERE_MAX_CHARS = 2000

# ─── Vector utilities ─────────────────────────────────────────────────────────

def pad_vector_to_dimension(vector: np.ndarray, target_dim: int) -> np.ndarray:
    current_dim = vector.shape[0] if vector.ndim == 1 else vector.shape[1]
    if current_dim >= target_dim:
        return vector[:target_dim] if vector.ndim == 1 else vector[:, :target_dim]
    if vector.ndim == 1:
        return np.concatenate([vector, np.zeros(target_dim - current_dim, dtype=vector.dtype)])
    return np.concatenate(
        [vector, np.zeros((vector.shape[0], target_dim - current_dim), dtype=vector.dtype)],
        axis=1,
    )


def safe_encode_and_fix_dimension(query: str, target_collection: str, target_field: str) -> np.ndarray:
    query_vector = embedding_model.encode_single(query)
    expected_dim = milvus_client._get_collection_dimension(target_collection, target_field)
    if expected_dim > 0 and query_vector.shape[0] != expected_dim:
        query_vector = pad_vector_to_dimension(query_vector, expected_dim)
    return query_vector

# ─── Reranking ────────────────────────────────────────────────────────────────

@tool
def rerank_faq(query: str, faq_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rerank FAQ candidates using Cohere."""
    try:
        if not faq_results or cohere_client is None:
            return faq_results
        documents = [
            f"Câu hỏi: {f.get('question','')}\nTrả lời: {f.get('answer','')}"
            for f in faq_results
        ]
        resp = cohere_client.rerank(
            query=query, documents=documents, model=COHERE_RERANK_MODEL,
            top_n=len(documents), return_documents=False,
        )
        reranked = []
        for r in resp.results:
            faq = faq_results[r.index].copy()
            faq["rerank_score"] = float(r.relevance_score)
            reranked.append(faq)
        reranked.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        logger.info(
            f"✅ FAQ reranked {len(reranked)} — "
            f"best={reranked[0].get('rerank_score',0):.4f}"
        )
        return reranked
    except Exception as e:
        logger.error(f"rerank_faq error: {e}")
        return sorted(faq_results, key=lambda x: x.get("similarity_score", 0), reverse=True)


@tool
def rerank_documents(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rerank documents using Cohere.

    OPT: Truncate description xuống _COHERE_MAX_CHARS trước khi gửi.
    Cohere giới hạn ~512 token/doc. Gửi full text bị cắt ngẫu nhiên ở cuối —
    tự truncate có kiểm soát giữ được context_header (phần quan trọng nhất
    ở đầu mỗi chunk).
    """
    try:
        if not documents or cohere_client is None:
            return documents

        texts = [
            (d.get("description", "") or d.get("answer", ""))[:_COHERE_MAX_CHARS]
            for d in documents
        ]

        resp = cohere_client.rerank(
            query=query, documents=texts, model=COHERE_RERANK_MODEL,
            top_n=len(texts), return_documents=False,
        )
        reranked = []
        for r in resp.results:
            doc = documents[r.index].copy()
            doc["rerank_score"] = float(r.relevance_score)
            reranked.append(doc)
        reranked.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        # Log phân phối score để monitor threshold
        scores = [d.get("rerank_score", 0) for d in reranked]
        logger.info(
            f"✅ Reranked {len(reranked)} docs — "
            f"best={scores[0]:.4f} p50={scores[len(scores)//2]:.4f} worst={scores[-1]:.4f}"
        )
        return reranked
    except Exception as e:
        logger.error(f"rerank_documents error: {e}")
        return documents

# ─── Document search ──────────────────────────────────────────────────────────

@tool
def search_documents(query: str) -> List[Dict[str, Any]]:
    """Search document_embeddings không có ACL filter."""
    try:
        query_vector = safe_encode_and_fix_dimension(
            query, settings.DOCUMENT_COLLECTION, "description_vector"
        )
        return milvus_client.search_documents(query_vector, settings.TOP_K)
    except Exception as e:
        logger.error(f"search_documents error: {e}")
        return [{"error": str(e)}]


@tool
def search_documents_for_user(query: str, user_id: str) -> List[Dict[str, Any]]:
    """Search document_embeddings với ACL filter từ user_db."""
    try:
        query_vector = safe_encode_and_fix_dimension(
            query, settings.DOCUMENT_COLLECTION, "description_vector"
        )
        return milvus_client.search_documents_with_acl(
            query_vector, settings.TOP_K, user_id=user_id
        )
    except Exception as e:
        logger.error(f"search_documents_for_user error: {e}")
        return [{"error": str(e)}]

# ─── FAQ search ───────────────────────────────────────────────────────────────

@tool
def search_faq(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """Search faq_embeddings."""
    try:
        if top_k is None:
            top_k = getattr(settings, "FAQ_TOP_K", 10)
        query_vector = safe_encode_and_fix_dimension(
            query, settings.FAQ_COLLECTION, "question_vector"
        )
        return milvus_client.search_faq(query_vector, top_k)
    except Exception as e:
        logger.error(f"search_faq error: {e}")
        return [{"error": str(e)}]

# ─── Health check ─────────────────────────────────────────────────────────────

@tool
def check_database_connection() -> Dict[str, Any]:
    """Kiểm tra kết nối Milvus default db + user_db + Cohere."""
    try:
        default_ok   = milvus_client.check_connection()
        user_db_ok   = False
        user_db_info = {"available": False}
        try:
            mgr = _get_user_db_manager()
            if mgr:
                user_db_ok  = mgr.health_check()
                db_info     = mgr.get_database_info()
                user_db_info = {
                    "available":  user_db_ok,
                    "database":   db_info.get("database", "user_db"),
                    "user_count": db_info.get("user_count", "?"),
                }
        except Exception as ue:
            user_db_info = {"available": False, "error": str(ue)}

        return {
            "connected": default_ok,
            "message":   "OK" if default_ok else "default db disconnected",
            "default_db": {
                "connected":   default_ok,
                "collections": [settings.DOCUMENT_COLLECTION, settings.FAQ_COLLECTION],
            },
            "user_db": user_db_info,
            "cohere_reranker": {
                "available": cohere_client is not None,
                "model":     COHERE_RERANK_MODEL if cohere_client else None,
            },
        }
    except Exception as e:
        return {"connected": False, "message": str(e)}
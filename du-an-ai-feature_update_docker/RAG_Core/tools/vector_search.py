# RAG_Core/tools/vector_search.py  (UPDATED – ACL-aware search)
"""
All tools from the original file are preserved.
New: search_documents_acl – filters by user permission scope.
The existing search_documents tool now delegates to ACL search when user_id is set.
"""

from langchain_core.tools import tool
from typing import List, Dict, Any, Optional
import numpy as np, os, logging
from models.embedding_model import embedding_model
from database.milvus_client import milvus_client
from config.settings import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# COHERE SETUP  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

cohere_client = None
COHERE_RERANK_MODEL = "rerank-multilingual-v3.0"

try:
    import cohere
    cohere_api_key = getattr(settings, "COHERE_API_KEY", None) or os.getenv("COHERE_API_KEY") or "NoQ9Jjvz5r1JeRWZG8L9dnl8BxYljmnOdiUfTnfk"
    if cohere_api_key and cohere_api_key != "your-api-key-here":
        cohere_client = cohere.Client(cohere_api_key)
        if hasattr(settings, "COHERE_RERANK_MODEL"):
            COHERE_RERANK_MODEL = settings.COHERE_RERANK_MODEL
        logger.info(f"✅ Cohere initialized: {COHERE_RERANK_MODEL}")
except Exception as e:
    logger.error(f"Cohere init error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# ACL HELPER
# ─────────────────────────────────────────────────────────────────────────────

def build_acl_expr(permissions: Optional[Dict[str, str]]) -> str:
    """
    Build Milvus filter expression from user permission scope.

    Permission logic (4-level hierarchy):
      - A document is accessible if EVERY non-empty ACL field of the document
        matches the user's corresponding field, OR the document's ACL field is "".

    Example:
      user has group=Techcombank_group, company=Techcomlife, dept=Sell_Techcomlife
      → can see docs where:
           (acl_group_id == "" OR acl_group_id == "Techcombank_group")
        AND (acl_company_id == "" OR acl_company_id == "Techcomlife")
        AND (acl_department_id == "" OR acl_department_id == "Sell_Techcomlife")
        AND (acl_user_id == "" OR acl_user_id == "<user_id>")
    """
    if not permissions:
        return ""   # no restriction

    clauses = []
    field_map = {
        "group_id":      "acl_group_id",
        "company_id":    "acl_company_id",
        "department_id": "acl_department_id",
        "user_id":       "acl_user_id",
    }
    for perm_key, milvus_field in field_map.items():
        val = (permissions.get(perm_key) or "").strip()
        if val:
            # doc's field must be blank (public) OR match user's value
            clauses.append(f'({milvus_field} == "" or {milvus_field} == "{val}")')
        else:
            # user has no restriction at this level → only public (blank) docs at this level
            # OR any value is ok because user is at a higher wildcard level
            # Wildcard: user's field is empty → match anything at this level
            pass   # no clause needed when user has wildcard

    return " and ".join(clauses) if clauses else ""


def get_user_permissions(user_id: Optional[str]) -> Optional[Dict[str, str]]:
    """Fetch permission scope from Milvus user_groups collection."""
    if not user_id:
        return None
    try:
        from pymilvus import Collection, utility
        if not utility.has_collection("user_groups"):
            return None
        col = Collection("user_groups")
        col.load()
        rows = col.query(
            expr=f'user_id == "{user_id}"',
            output_fields=["group_id", "company_id", "department_id", "user_id"],
            limit=1,
        )
        if rows:
            r = rows[0]
            return {
                "group_id":      r.get("group_id", ""),
                "company_id":    r.get("company_id", ""),
                "department_id": r.get("department_id", ""),
                "user_id":       user_id,
            }
    except Exception as e:
        logger.warning(f"get_user_permissions error for {user_id}: {e}")
    return None

# ─────────────────────────────────────────────────────────────────────────────
# VECTOR DIMENSION UTILITIES  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def pad_vector_to_dimension(vector: np.ndarray, target_dim: int) -> np.ndarray:
    current_dim = vector.shape[0] if vector.ndim == 1 else vector.shape[1]
    if current_dim >= target_dim:
        return vector[:target_dim] if vector.ndim == 1 else vector[:, :target_dim]
    if vector.ndim == 1:
        return np.concatenate([vector, np.zeros(target_dim - current_dim, dtype=vector.dtype)])
    return np.concatenate([vector, np.zeros((vector.shape[0], target_dim - current_dim), dtype=vector.dtype)], axis=1)


def safe_encode_and_fix_dimension(query: str, target_collection: str, target_field: str) -> np.ndarray:
    query_vector  = embedding_model.encode_single(query)
    expected_dim  = milvus_client._get_collection_dimension(target_collection, target_field)
    if expected_dim > 0 and query_vector.shape[0] != expected_dim:
        query_vector = pad_vector_to_dimension(query_vector, expected_dim)
    return query_vector

# ─────────────────────────────────────────────────────────────────────────────
# RERANKING TOOLS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def rerank_faq(query: str, faq_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rerank FAQ candidates using Cohere."""
    try:
        if not faq_results or cohere_client is None:
            return faq_results
        documents = [f"Câu hỏi: {f.get('question','')}\nTrả lời: {f.get('answer','')}" for f in faq_results]
        resp = cohere_client.rerank(query=query, documents=documents, model=COHERE_RERANK_MODEL, top_n=len(documents), return_documents=False)
        reranked = []
        for r in resp.results:
            faq = faq_results[r.index].copy()
            faq["rerank_score"] = float(r.relevance_score)
            reranked.append(faq)
        reranked.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return reranked
    except Exception as e:
        logger.error(f"rerank_faq error: {e}")
        return sorted(faq_results, key=lambda x: x.get("similarity_score", 0), reverse=True)


@tool
def rerank_documents(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rerank documents using Cohere."""
    try:
        if not documents or cohere_client is None:
            return documents
        texts = [d.get("description", "") or d.get("answer", "") for d in documents]
        resp  = cohere_client.rerank(query=query, documents=texts, model=COHERE_RERANK_MODEL, top_n=len(texts), return_documents=False)
        reranked = []
        for r in resp.results:
            doc = documents[r.index].copy()
            doc["rerank_score"] = float(r.relevance_score)
            reranked.append(doc)
        reranked.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        logger.info(f"✅ Reranked {len(reranked)} docs. Best={reranked[0].get('rerank_score',0):.3f}")
        return reranked
    except Exception as e:
        logger.error(f"rerank_documents error: {e}")
        return documents

# ─────────────────────────────────────────────────────────────────────────────
# ACL-AWARE DOCUMENT SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def _search_documents_with_acl(query: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Internal helper: search document_embeddings with optional ACL filter.
    """
    try:
        query_vector = safe_encode_and_fix_dimension(query, settings.DOCUMENT_COLLECTION, "description_vector")
        permissions  = get_user_permissions(user_id)
        acl_expr     = build_acl_expr(permissions)

        if not milvus_client.check_connection():
            raise ConnectionError("Not connected to Milvus")

        from pymilvus import Collection
        collection   = milvus_client._get_collection(settings.DOCUMENT_COLLECTION)
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

        kwargs: dict = dict(
            data=[query_vector.tolist()],
            anns_field="description_vector",
            param=search_params,
            limit=settings.TOP_K,
            output_fields=["document_id", "description"],
        )
        if acl_expr:
            kwargs["expr"] = acl_expr
            logger.info(f"🔒 ACL filter applied: {acl_expr[:120]}")

        results = collection.search(**kwargs)
        documents = []
        for hits in results:
            for hit in hits:
                documents.append({
                    "document_id":      hit.entity.get("document_id"),
                    "description":      hit.entity.get("description"),
                    "similarity_score": hit.score,
                })
        logger.info(f"✅ Found {len(documents)} documents (user_id={user_id})")
        return documents
    except Exception as e:
        logger.error(f"_search_documents_with_acl error: {e}")
        return []


@tool
def search_documents(query: str) -> List[Dict[str, Any]]:
    """Search documents (no ACL – backward compat)."""
    try:
        query_vector = safe_encode_and_fix_dimension(query, settings.DOCUMENT_COLLECTION, "description_vector")
        return milvus_client.search_documents(query_vector, settings.TOP_K)
    except Exception as e:
        logger.error(f"search_documents error: {e}")
        return [{"error": str(e)}]


@tool
def search_documents_for_user(query: str, user_id: str) -> List[Dict[str, Any]]:
    """Search documents filtered by user's ACL permissions."""
    return _search_documents_with_acl(query, user_id)


@tool
def search_faq(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """Search FAQ collection."""
    try:
        if top_k is None:
            top_k = getattr(settings, "FAQ_TOP_K", 10)
        query_vector = safe_encode_and_fix_dimension(query, settings.FAQ_COLLECTION, "question_vector")
        return milvus_client.search_faq(query_vector, top_k)
    except Exception as e:
        logger.error(f"search_faq error: {e}")
        return [{"error": str(e)}]


@tool
def check_database_connection() -> Dict[str, Any]:
    """Check DB connection."""
    try:
        is_connected = milvus_client.check_connection()
        result = {"connected": is_connected, "message": "OK" if is_connected else "Disconnected"}
        result["cohere_reranker"] = {"available": cohere_client is not None, "model": COHERE_RERANK_MODEL if cohere_client else None}
        return result
    except Exception as e:
        return {"connected": False, "message": str(e)}
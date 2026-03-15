# RAG_Core/database/milvus_client.py  (UPDATED v2 – user_groups removed, user_db support)
"""
MilvusClient cho RAG_Core.

Thay đổi so với v1:
  - XÓA toàn bộ user_groups collection và các method liên quan:
      create_user_groups_collection(), get_user(), update_token_cost()
  - THÊM _get_user_db_manager() để proxy sang user_db (Milvus database "user_db")
    khi cần lấy thông tin user (VD: check_database_connection trả về user_db status)
  - Chỉ quản lý 2 collections trong default db:
      • document_embeddings
      • faq_embeddings
  - document_urls được quản lý bởi document_url_service (không thay đổi)
"""

from pymilvus import connections, Collection, utility, db
from typing import List, Dict, Any, Optional
import numpy as np
import logging
import time
import os

from config.settings import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# USER DB MANAGER  (lazy – proxy sang user_db)
# ─────────────────────────────────────────────────────────────────────────────

_user_db_manager = None


def _get_user_db_manager():
    """
    Lazy-init singleton UserDBManager.
    UserDBManager kết nối tới Milvus database "user_db" (tách biệt với default db).
    """
    global _user_db_manager
    if _user_db_manager is None:
        try:
            import sys
            # Thêm path tới Embedding_vectorDB để import user_db_manager
            embedding_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "Embedding_vectorDB"
            )
            abs_path = os.path.abspath(embedding_path)
            if abs_path not in sys.path:
                sys.path.insert(0, abs_path)

            from user_db_manager import get_user_db_manager
            _user_db_manager = get_user_db_manager(
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
            )
            logger.info("✅ UserDBManager ready (user_db)")
        except Exception as e:
            logger.warning(f"UserDBManager not available: {e}. User features disabled.")
    return _user_db_manager


class MilvusClient:
    """
    Quản lý kết nối và search trong Milvus default database.

    Collections được quản lý:
        default db
        ├── document_embeddings
        └── faq_embeddings

    KHÔNG quản lý:
        user_db.user_groups  → UserDBManager (user_db_manager.py)
        default.document_urls → DocumentURLService
    """

    def __init__(self):
        self.connected = False
        self.expected_dimension = None
        self.collections_cache: Dict[str, Collection] = {}
        self._connect()

    # ─────────────────────────────────────────────────────────────────────────
    # CONNECTION
    # ─────────────────────────────────────────────────────────────────────────

    def _connect(self):
        """Kết nối tới Milvus default database với retry."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                try:
                    connections.disconnect("default")
                except Exception:
                    pass

                connections.connect(
                    alias="default",
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT,
                    timeout=10,
                )
                logger.info(
                    f"✅ Connected to Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
                )

                # Đảm bảo dùng default database cho document/faq collections
                try:
                    db.using_database("default")
                    logger.info("✅ Using database: default")
                except Exception as db_err:
                    logger.warning(f"Could not switch database: {db_err}")

                self.connected = True

                try:
                    self._load_collections()
                except Exception as load_err:
                    logger.warning(f"Collection loading failed (non-fatal): {load_err}")

                return

            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    logger.error(f"Failed to connect to Milvus after {max_retries} attempts")
                    self.connected = False

    def _load_collections(self):
        """Load document và faq collections (không load user_groups)."""
        try:
            available = utility.list_collections()
            logger.info(f"📚 Available collections (default db): {available}")

            for col_name in [settings.DOCUMENT_COLLECTION, settings.FAQ_COLLECTION]:
                if col_name in available:
                    try:
                        col = Collection(col_name)
                        col.load()
                        self.collections_cache[col_name] = col
                        logger.info(f"✅ Loaded: {col_name} ({col.num_entities} entities)")
                    except Exception as e:
                        logger.warning(f"Could not load {col_name}: {e}")

        except Exception as e:
            logger.error(f"Error loading collections: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # CONNECTION CHECK
    # ─────────────────────────────────────────────────────────────────────────

    def check_connection(self) -> bool:
        """Kiểm tra kết nối Milvus default db."""
        if not self.connected:
            return False
        try:
            utility.list_collections(timeout=2)
            return True
        except Exception:
            logger.warning("Connection lost, reconnecting…")
            self._connect()
            return self.connected

    # ─────────────────────────────────────────────────────────────────────────
    # COLLECTION ACCESS
    # ─────────────────────────────────────────────────────────────────────────

    def _get_collection(self, collection_name: str) -> Collection:
        """Lấy collection với lazy loading và cache."""
        if collection_name in self.collections_cache:
            col = self.collections_cache[collection_name]
            try:
                _ = col.num_entities
                return col
            except Exception:
                logger.warning(f"Cached collection {collection_name} stale, reloading…")
                del self.collections_cache[collection_name]

        try:
            if not utility.has_collection(collection_name):
                raise ValueError(f"Collection '{collection_name}' does not exist")
            col = Collection(collection_name)
            col.load()
            self.collections_cache[collection_name] = col
            logger.info(f"✅ Loaded collection: {collection_name}")
            return col
        except Exception as e:
            logger.error(f"Failed to load collection {collection_name}: {e}")
            raise

    def _get_collection_dimension(
        self, collection_name: str, vector_field: str
    ) -> int:
        try:
            col    = self._get_collection(collection_name)
            schema = col.schema
            for field in schema.fields:
                if field.name == vector_field:
                    return field.params.get("dim", 0)
            logger.warning(f"Vector field {vector_field} not found in {collection_name}")
            return 0
        except Exception as e:
            logger.error(f"Error getting dimension: {e}")
            return 0

    def _validate_vector_dimension(
        self,
        vector: np.ndarray,
        collection_name: str,
        vector_field: str,
        auto_fix: bool = True,
    ) -> np.ndarray:
        expected_dim = self._get_collection_dimension(collection_name, vector_field)
        actual_dim   = vector.shape[0] if vector.ndim == 1 else vector.shape[1]
        if expected_dim == 0:
            return vector
        if actual_dim != expected_dim:
            if auto_fix:
                logger.warning(
                    f"Dimension mismatch: expected {expected_dim}, got {actual_dim}. Auto-fixing…"
                )
                return self._adjust_vector_dimension(vector, expected_dim)
            raise ValueError(f"Dimension mismatch: expected {expected_dim}, got {actual_dim}")
        return vector

    def _adjust_vector_dimension(self, vector: np.ndarray, target_dim: int) -> np.ndarray:
        if vector.ndim > 1:
            cur = vector.shape[1]
            if cur < target_dim:
                return np.concatenate(
                    [vector, np.zeros((vector.shape[0], target_dim - cur), dtype=vector.dtype)],
                    axis=1,
                )
            return vector[:, :target_dim]
        cur = vector.shape[0]
        if cur < target_dim:
            return np.concatenate([vector, np.zeros(target_dim - cur, dtype=vector.dtype)])
        return vector[:target_dim]

    # ─────────────────────────────────────────────────────────────────────────
    # SEARCH – DOCUMENT EMBEDDINGS (default db)
    # ─────────────────────────────────────────────────────────────────────────

    def search_documents(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search document_embeddings (không có ACL filter)."""
        return self._search_documents_internal(query_vector, top_k, acl_expr=None)

    def search_documents_with_acl(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search document_embeddings với ACL filter.
        ACL permissions lấy từ user_db.user_groups qua UserDBManager.
        """
        acl_expr = None
        if user_id:
            mgr = _get_user_db_manager()
            if mgr:
                acl_expr = mgr.build_acl_expression(user_id)
                if acl_expr:
                    logger.info(f"🔒 ACL filter (user_db): {acl_expr[:100]}")
                else:
                    logger.info(f"🔓 No ACL restrictions for user_id={user_id}")
            else:
                logger.warning("UserDBManager unavailable, running open search")

        return self._search_documents_internal(query_vector, top_k, acl_expr=acl_expr)

    def _search_documents_internal(
        self,
        query_vector: np.ndarray,
        top_k: int,
        acl_expr: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Thực hiện search với retry logic."""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if not self.check_connection():
                    raise ConnectionError("Not connected to Milvus")

                col = self._get_collection(settings.DOCUMENT_COLLECTION)
                query_vector = self._validate_vector_dimension(
                    query_vector, settings.DOCUMENT_COLLECTION, "description_vector"
                )

                search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
                kwargs = dict(
                    data=[query_vector.tolist()],
                    anns_field="description_vector",
                    param=search_params,
                    limit=top_k,
                    output_fields=["document_id", "description"],
                )
                if acl_expr:
                    kwargs["expr"] = acl_expr

                results   = col.search(**kwargs)
                documents = []
                for hits in results:
                    for hit in hits:
                        documents.append({
                            "document_id":      hit.entity.get("document_id"),
                            "description":      hit.entity.get("description"),
                            "similarity_score": hit.score,
                        })

                logger.info(f"✅ Found {len(documents)} documents")
                return documents

            except Exception as e:
                logger.error(f"Search attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    self._connect()
                    time.sleep(1)
                else:
                    raise

    # ─────────────────────────────────────────────────────────────────────────
    # SEARCH – FAQ (default db)
    # ─────────────────────────────────────────────────────────────────────────

    def search_faq(
        self, query_vector: np.ndarray, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Search faq_embeddings."""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if not self.check_connection():
                    raise ConnectionError("Not connected to Milvus")

                col = self._get_collection(settings.FAQ_COLLECTION)
                query_vector = self._validate_vector_dimension(
                    query_vector, settings.FAQ_COLLECTION, "question_vector"
                )

                search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
                results = col.search(
                    data=[query_vector.tolist()],
                    anns_field="question_vector",
                    param=search_params,
                    limit=top_k,
                    output_fields=["faq_id", "question", "answer"],
                )

                faqs = []
                for hits in results:
                    for hit in hits:
                        faqs.append({
                            "faq_id":           hit.entity.get("faq_id"),
                            "question":         hit.entity.get("question"),
                            "answer":           hit.entity.get("answer"),
                            "similarity_score": hit.score,
                        })

                logger.info(f"✅ Found {len(faqs)} FAQs")
                return faqs

            except Exception as e:
                logger.error(f"FAQ search attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    self._connect()
                    time.sleep(1)
                else:
                    raise

    # ─────────────────────────────────────────────────────────────────────────
    # COLLECTION INFO
    # ─────────────────────────────────────────────────────────────────────────

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Thông tin collection trong default db."""
        try:
            if not utility.has_collection(collection_name):
                return {"error": f"Collection {collection_name} does not exist"}
            col    = Collection(collection_name)
            schema = col.schema
            return {
                "collection_name": collection_name,
                "database":        "default",
                "fields":          [
                    {"name": f.name, "dtype": str(f.dtype), "params": f.params}
                    for f in schema.fields
                ],
                "description":  schema.description,
                "num_entities": col.num_entities,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_all_stats(self) -> Dict[str, Any]:
        """
        Stats cho tất cả databases.
        default db: document_embeddings, faq_embeddings
        user_db:    user_groups (via UserDBManager)
        """
        stats: Dict[str, Any] = {
            "default_db": {},
            "user_db":    {},
        }

        # default db stats
        for col_name in [settings.DOCUMENT_COLLECTION, settings.FAQ_COLLECTION]:
            try:
                if utility.has_collection(col_name):
                    col = Collection(col_name)
                    stats["default_db"][col_name] = {"count": col.num_entities}
                else:
                    stats["default_db"][col_name] = {"count": 0, "exists": False}
            except Exception as e:
                stats["default_db"][col_name] = {"error": str(e)}

        # user_db stats (via UserDBManager)
        try:
            mgr = _get_user_db_manager()
            if mgr:
                info = mgr.get_database_info()
                stats["user_db"]["user_groups"] = {
                    "count":    info.get("user_count", "?"),
                    "database": "user_db",
                }
            else:
                stats["user_db"]["user_groups"] = {"error": "UserDBManager unavailable"}
        except Exception as e:
            stats["user_db"]["user_groups"] = {"error": str(e)}

        return stats


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

milvus_client = MilvusClient()
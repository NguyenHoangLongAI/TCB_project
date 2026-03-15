# Embedding_vectorDB/milvus_client.py  (UPDATED v5 – user_groups tách sang user_db)
"""
Unified Milvus Manager cho document/FAQ/URL collections.

Thay đổi so với v4:
  - user_groups collection ĐÃ ĐƯỢC TÁCH ra user_db_manager.py / UserDBManager
  - MilvusManager chỉ quản lý:
      • document_embeddings  (default db)
      • faq_embeddings       (default db)
      • document_urls        (default db)
  - Mọi thao tác user (create_user, get_user, authenticate, update_token_cost)
    đều delegate sang user_db_manager singleton

Architecture:
    Milvus
    ├── default (db)
    │   ├── document_embeddings
    │   ├── faq_embeddings
    │   └── document_urls
    └── user_db (db)          ← quản lý bởi UserDBManager
        └── user_groups
"""

from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, DataType, utility, db
)
from typing import List, Dict, Any, Optional
import asyncio
import logging
import os

# Import UserDBManager thay vì tự quản lý user collection
from user_db_manager import get_user_db_manager, UserDBManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusManager:

    def __init__(self, host: str = "localhost", port: str = "19530", embedding_dim: int = 768):
        self.host = host
        self.port = port
        self.embedding_dim = embedding_dim

        # Collections trong default database
        self.doc_collection_name  = "document_embeddings"
        self.faq_collection_name  = "faq_embeddings"
        self.url_collection_name  = "document_urls"

        self.doc_collection = None
        self.faq_collection = None
        self.url_collection = None

        self.is_initialized = False

        self.max_id_length          = 190
        self.max_document_id_length = 90
        self.max_description_length = 60000
        self.max_question_length    = 60000
        self.max_answer_length      = 60000

        self._embedding_model = None

        # user_db_manager – lazy init (tránh double-connect lúc startup)
        self._user_db_manager: Optional[UserDBManager] = None

    # ──────────────────────────────────────────────
    # USER DB  (delegate sang UserDBManager)
    # ──────────────────────────────────────────────

    @property
    def user_mgr(self) -> UserDBManager:
        """Lazy-init UserDBManager singleton (kết nối tới user_db)."""
        if self._user_db_manager is None:
            self._user_db_manager = get_user_db_manager(
                host=self.host, port=self.port
            )
        return self._user_db_manager

    # ──────────────────────────────────────────────
    # CONNECTION  (chỉ kết nối cho default db)
    # ──────────────────────────────────────────────

    async def initialize(self, max_retries: int = 5, retry_delay: int = 2):
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Connecting to Milvus {self.host}:{self.port} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                # Resolve hostname inside Docker
                if self.host == "milvus":
                    import socket
                    try:
                        socket.gethostbyname("milvus")
                    except socket.gaierror:
                        self.host = "localhost"
                        logger.warning("Running outside Docker → localhost")

                try:
                    connections.disconnect("default")
                except Exception:
                    pass

                connections.connect("default", host=self.host, port=self.port)

                # Đảm bảo ở đúng database (default cho document/faq/url)
                try:
                    db.using_database("default")
                except Exception:
                    pass

                logger.info(f"✅ Connected to Milvus (default db) {self.host}:{self.port}")

                # Tạo collections trong default db
                await self.create_document_collection()
                await self.create_faq_collection()
                await self.create_url_collection()

                # Khởi tạo user_db_manager (tạo/load user_db.user_groups)
                logger.info("🔄 Initializing UserDBManager (user_db) …")
                _ = self.user_mgr
                logger.info("✅ UserDBManager ready (database: user_db)")

                self.is_initialized = True
                logger.info("✅ MilvusManager initialization complete")
                return True

            except Exception as e:
                logger.error(f"❌ Init error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    self.is_initialized = False
                    raise e

    def _check_initialized(self):
        if not self.is_initialized:
            raise Exception("Milvus not initialized.")

    # ──────────────────────────────────────────────
    # EMBEDDING MODEL (lazy)
    # ──────────────────────────────────────────────

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading Vietnamese SBERT (CPU) …")
            self._embedding_model = SentenceTransformer(
                "keepitreal/vietnamese-sbert", device="cpu"
            )
            logger.info("✅ Embedding model ready")
        return self._embedding_model

    def embed_text(self, text: str) -> List[float]:
        try:
            if not text or not text.strip():
                return [0.0] * self.embedding_dim
            return self.embedding_model.encode(
                text.strip(), normalize_embeddings=True
            ).tolist()
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return [0.0] * self.embedding_dim

    # ──────────────────────────────────────────────
    # COLLECTION CREATION  (default db only)
    # ──────────────────────────────────────────────

    async def create_document_collection(self):
        """document_embeddings – ACL fields."""
        try:
            if utility.has_collection(self.doc_collection_name):
                logger.info(f"Collection {self.doc_collection_name} exists – loading.")
                self.doc_collection = Collection(self.doc_collection_name)
                await self._optimize_collection_index(self.doc_collection, "description_vector")
                self.doc_collection.load()
                return

            logger.info(f"Creating {self.doc_collection_name} …")
            fields = [
                FieldSchema(name="id",                    dtype=DataType.VARCHAR, max_length=200, is_primary=True),
                FieldSchema(name="document_id",           dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="description",           dtype=DataType.VARCHAR, max_length=65000),
                FieldSchema(name="description_vector",    dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
                # ACL fields (tham chiếu tới user_db.user_groups)
                FieldSchema(name="acl_group_id",          dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="acl_company_id",        dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="acl_department_id",     dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="acl_user_id",           dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="uploaded_by",           dtype=DataType.VARCHAR, max_length=100),
            ]
            schema = CollectionSchema(
                fields,
                description="Document embeddings (768D) + ACL (refs user_db.user_groups)",
            )
            self.doc_collection = Collection(
                self.doc_collection_name, schema=schema, using="default"
            )
            self.doc_collection.create_index(
                field_name="description_vector",
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
            self.doc_collection.load()
            logger.info(f"✅ {self.doc_collection_name} created with ACL fields.")
        except Exception as e:
            logger.error(f"❌ create_document_collection: {e}")
            raise

    async def create_faq_collection(self):
        try:
            if utility.has_collection(self.faq_collection_name):
                logger.info(f"Collection {self.faq_collection_name} exists – loading.")
                self.faq_collection = Collection(self.faq_collection_name)
                await self._optimize_collection_index(self.faq_collection, "question_vector")
                self.faq_collection.load()
                return

            fields = [
                FieldSchema(name="faq_id",         dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="question",        dtype=DataType.VARCHAR, max_length=65000),
                FieldSchema(name="answer",          dtype=DataType.VARCHAR, max_length=65000),
                FieldSchema(name="question_vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            ]
            schema = CollectionSchema(fields, description="FAQ embeddings (768D)")
            self.faq_collection = Collection(
                self.faq_collection_name, schema=schema, using="default"
            )
            self.faq_collection.create_index(
                field_name="question_vector",
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
            self.faq_collection.load()
            logger.info(f"✅ {self.faq_collection_name} created.")
        except Exception as e:
            logger.error(f"❌ create_faq_collection: {e}")
            raise

    async def create_url_collection(self):
        try:
            if utility.has_collection(self.url_collection_name):
                logger.info(f"Collection {self.url_collection_name} exists – loading.")
                self.url_collection = Collection(self.url_collection_name)
                self.url_collection.load()
                return

            fields = [
                FieldSchema(name="document_id",     dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="url",             dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="filename",        dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="file_type",       dtype=DataType.VARCHAR, max_length=20),
                FieldSchema(name="filename_vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            ]
            schema = CollectionSchema(fields, description="Document URLs + filename embeddings")
            self.url_collection = Collection(
                self.url_collection_name, schema=schema, using="default"
            )
            self.url_collection.create_index(
                field_name="filename_vector",
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128},
                },
            )
            self.url_collection.load()
            logger.info(f"✅ {self.url_collection_name} created.")
        except Exception as e:
            logger.error(f"❌ create_url_collection: {e}")
            raise

    async def _optimize_collection_index(self, collection: Collection, vector_field: str):
        try:
            indexes = collection.indexes
            if not indexes:
                collection.release()
                collection.create_index(
                    field_name=vector_field,
                    index_params={
                        "metric_type": "COSINE",
                        "index_type": "HNSW",
                        "params": {"M": 16, "efConstruction": 200},
                    },
                )
                collection.load()
                return
            for idx in indexes:
                if idx.params.get("index_type") == "IVF_FLAT":
                    collection.release()
                    collection.drop_index()
                    collection.create_index(
                        field_name=vector_field,
                        index_params={
                            "metric_type": "COSINE",
                            "index_type": "HNSW",
                            "params": {"M": 16, "efConstruction": 200},
                        },
                    )
                    collection.load()
                    return
        except Exception as e:
            logger.error(f"_optimize_collection_index error: {e}")

    # ──────────────────────────────────────────────
    # DOCUMENT EMBEDDINGS  (ACL-aware insert)
    # ──────────────────────────────────────────────

    async def insert_embeddings(
        self,
        embeddings_data: List[Dict],
        acl: Optional[Dict[str, str]] = None,
        uploaded_by: str = "",
    ) -> int:
        """Insert embeddings với ACL. ACL values lấy từ user_db.user_groups."""
        self._check_initialized()
        if not self.doc_collection or not embeddings_data:
            return 0

        acl = acl or {}
        acl_group      = (acl.get("group_id",      "") or "")[:100]
        acl_company    = (acl.get("company_id",    "") or "")[:100]
        acl_department = (acl.get("department_id", "") or "")[:100]
        acl_user       = (acl.get("user_id",       "") or "")[:100]

        field_limits = {
            "id":          self.max_id_length,
            "document_id": self.max_document_id_length,
            "description": self.max_description_length,
        }
        validated = [
            self._validate_and_truncate(item, field_limits)
            for item in embeddings_data
            if all(k in item for k in ["id", "document_id", "description", "description_vector"])
            and len(item["description_vector"]) == self.embedding_dim
        ]
        if not validated:
            return 0

        batch_size     = 100
        total_inserted = 0
        for i in range(0, len(validated), batch_size):
            batch = validated[i: i + batch_size]
            entities = [
                [r["id"]          for r in batch],
                [r["document_id"] for r in batch],
                [r["description"] for r in batch],
                [r["description_vector"] for r in batch],
                [acl_group]      * len(batch),
                [acl_company]    * len(batch),
                [acl_department] * len(batch),
                [acl_user]       * len(batch),
                [uploaded_by]    * len(batch),
            ]
            try:
                self.doc_collection.insert(entities)
                total_inserted += len(batch)
            except Exception as e:
                logger.error(f"Batch insert error: {e}")

        self.doc_collection.flush()
        logger.info(f"✅ Inserted {total_inserted} embeddings (ACL: {acl})")
        return total_inserted

    async def delete_document(self, document_id: str) -> bool:
        self._check_initialized()
        try:
            self.doc_collection.delete(f'document_id == "{document_id}"')
            return True
        except Exception as e:
            logger.error(f"delete_document error: {e}")
            return False

    # ──────────────────────────────────────────────
    # FAQ
    # ──────────────────────────────────────────────

    async def insert_faq(self, faq_id, question, answer, question_vector) -> bool:
        self._check_initialized()
        try:
            if len(faq_id) > 90:
                faq_id = faq_id[:90]
            if len(question) > self.max_question_length:
                question = question[: self.max_question_length - 3] + "..."
            if len(answer) > self.max_answer_length:
                answer = answer[: self.max_answer_length - 3] + "..."
            if len(question_vector) != self.embedding_dim:
                return False
            self.faq_collection.insert([[faq_id], [question], [answer], [question_vector]])
            self.faq_collection.flush()
            return True
        except Exception as e:
            logger.error(f"insert_faq error: {e}")
            return False

    async def delete_faq(self, faq_id: str) -> bool:
        self._check_initialized()
        try:
            self.faq_collection.delete(f'faq_id == "{faq_id}"')
            return True
        except Exception as e:
            logger.error(f"delete_faq error: {e}")
            return False

    # ──────────────────────────────────────────────
    # DOCUMENT URLS
    # ──────────────────────────────────────────────

    def insert_url(self, document_id, url, filename="", file_type="") -> bool:
        try:
            if not self.url_collection:
                return False
            document_id = document_id[:100]
            url         = url[:500]
            filename    = filename[:200]
            file_type   = file_type[:20]
            vec = self.embed_text(filename)
            try:
                existing = self.url_collection.query(
                    expr=f'document_id == "{document_id}"',
                    output_fields=["document_id"],
                    limit=1,
                )
                if existing:
                    self.url_collection.delete(f'document_id in ["{document_id}"]')
            except Exception:
                pass
            self.url_collection.insert(
                [[document_id], [url], [filename], [file_type], [vec]]
            )
            self.url_collection.flush()
            return True
        except Exception as e:
            logger.error(f"insert_url error: {e}")
            return False

    def delete_url(self, document_id: str) -> bool:
        try:
            self.url_collection.delete(f'document_id in ["{document_id}"]')
            self.url_collection.flush()
            return True
        except Exception as e:
            logger.error(f"delete_url error: {e}")
            return False

    def get_url(self, document_id: str) -> Optional[Dict]:
        try:
            results = self.url_collection.query(
                expr=f'document_id == "{document_id}"',
                output_fields=["url", "filename", "file_type"],
                limit=1,
            )
            if results:
                return {"document_id": document_id, **results[0]}
            return None
        except Exception as e:
            logger.error(f"get_url error: {e}")
            return None

    # ──────────────────────────────────────────────
    # USER GROUPS  (delegate → UserDBManager / user_db)
    # ──────────────────────────────────────────────

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Proxy tới user_db_manager.get_user()."""
        return self.user_mgr.get_user(user_id)

    def update_token_cost(self, user_id: str, tokens_used: int) -> bool:
        """Proxy tới user_db_manager.update_token_cost()."""
        return self.user_mgr.update_token_cost(user_id, tokens_used)

    def get_user_permissions(self, user_id: str) -> Optional[Dict[str, str]]:
        """Proxy tới user_db_manager.get_user_permissions()."""
        return self.user_mgr.get_user_permissions(user_id)

    # ──────────────────────────────────────────────
    # UTILITIES
    # ──────────────────────────────────────────────

    def _validate_and_truncate(self, data: Dict, field_limits: Dict) -> Dict:
        validated = data.copy()
        for field, max_len in field_limits.items():
            if field in validated and isinstance(validated[field], str):
                if len(validated[field]) > max_len:
                    validated[field] = validated[field][: max_len - 3] + "..."
        return validated

    async def health_check(self) -> bool:
        try:
            if not self.is_initialized:
                return False
            connections.get_connection_addr("default")
            return True
        except Exception:
            return False

    async def get_collection_stats(self) -> Dict:
        stats = {"initialized": self.is_initialized}
        for name, col in [
            ("document_embeddings", self.doc_collection),
            ("faq_embeddings",      self.faq_collection),
            ("document_urls",       self.url_collection),
        ]:
            if col:
                try:
                    stats[name] = {"count": col.num_entities, "database": "default"}
                except Exception:
                    stats[name] = {"count": "?"}

        # user_groups stats từ user_db
        try:
            stats["user_groups"] = self.user_mgr.get_database_info()
        except Exception:
            stats["user_groups"] = {"database": "user_db", "count": "?"}

        return stats


async def main():
    import json
    manager = MilvusManager(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530"),
    )
    await manager.initialize()
    stats = await manager.get_collection_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
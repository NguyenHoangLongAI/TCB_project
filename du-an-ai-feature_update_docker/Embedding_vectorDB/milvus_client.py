# Embedding_vectorDB/milvus_client.py  (UPDATED – user_groups + ACL)
"""
Unified Milvus Manager – adds user_groups collection and
permission-aware document_embeddings (group/company/dept/user_id fields).
"""

from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, DataType, utility
)
from typing import List, Dict, Any, Optional
import asyncio
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusManager:

    def __init__(self, host: str = "localhost", port: str = "19530", embedding_dim: int = 768):
        self.host = host
        self.port = port
        self.embedding_dim = embedding_dim

        self.doc_collection_name  = "document_embeddings"
        self.faq_collection_name  = "faq_embeddings"
        self.url_collection_name  = "document_urls"
        self.ug_collection_name   = "user_groups"

        self.doc_collection = None
        self.faq_collection = None
        self.url_collection = None
        self.ug_collection  = None

        self.is_initialized = False

        self.max_id_length          = 190
        self.max_document_id_length = 90
        self.max_description_length = 60000
        self.max_question_length    = 60000
        self.max_answer_length      = 60000

        self._embedding_model = None

    # ──────────────────────────────────────────────
    # CONNECTION
    # ──────────────────────────────────────────────

    async def initialize(self, max_retries: int = 5, retry_delay: int = 2):
        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to Milvus {self.host}:{self.port} (attempt {attempt+1}/{max_retries})")
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
                logger.info(f"✅ Connected to Milvus {self.host}:{self.port}")

                await self.create_document_collection()
                await self.create_faq_collection()
                await self.create_url_collection()
                await self.create_user_groups_collection()

                self.is_initialized = True
                logger.info("✅ Milvus initialization complete")
                return True
            except Exception as e:
                logger.error(f"❌ Init error (attempt {attempt+1}): {e}")
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
            self._embedding_model = SentenceTransformer("keepitreal/vietnamese-sbert", device="cpu")
            logger.info("✅ Embedding model ready")
        return self._embedding_model

    def embed_text(self, text: str) -> List[float]:
        try:
            if not text or not text.strip():
                return [0.0] * self.embedding_dim
            return self.embedding_model.encode(text.strip(), normalize_embeddings=True).tolist()
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return [0.0] * self.embedding_dim

    # ──────────────────────────────────────────────
    # COLLECTION CREATION
    # ──────────────────────────────────────────────

    async def create_document_collection(self):
        """document_embeddings – now includes ACL fields."""
        try:
            if utility.has_collection(self.doc_collection_name):
                logger.info(f"Collection {self.doc_collection_name} exists – loading.")
                self.doc_collection = Collection(self.doc_collection_name)
                await self._optimize_collection_index(self.doc_collection, "description_vector")
                self.doc_collection.load()
                return

            logger.info(f"Creating {self.doc_collection_name} …")
            fields = [
                FieldSchema(name="id",              dtype=DataType.VARCHAR, max_length=200, is_primary=True),
                FieldSchema(name="document_id",     dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="description",     dtype=DataType.VARCHAR, max_length=65000),
                FieldSchema(name="description_vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
                # ── ACL fields ──────────────────────────────────────────────
                FieldSchema(name="acl_group_id",      dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="acl_company_id",    dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="acl_department_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="acl_user_id",       dtype=DataType.VARCHAR, max_length=100),
                # ── upload owner ────────────────────────────────────────────
                FieldSchema(name="uploaded_by",       dtype=DataType.VARCHAR, max_length=100),
            ]
            schema = CollectionSchema(fields, description="Document embeddings (768D) + ACL")
            self.doc_collection = Collection(self.doc_collection_name, schema=schema, using="default")
            self.doc_collection.create_index(
                field_name="description_vector",
                index_params={"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}},
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
            self.faq_collection = Collection(self.faq_collection_name, schema=schema, using="default")
            self.faq_collection.create_index(
                field_name="question_vector",
                index_params={"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}},
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
            self.url_collection = Collection(self.url_collection_name, schema=schema, using="default")
            self.url_collection.create_index(
                field_name="filename_vector",
                index_params={"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
            )
            self.url_collection.load()
            logger.info(f"✅ {self.url_collection_name} created.")
        except Exception as e:
            logger.error(f"❌ create_url_collection: {e}")
            raise

    async def create_user_groups_collection(self):
        """user_groups collection (4-level ACL + token cost)."""
        try:
            if utility.has_collection(self.ug_collection_name):
                logger.info(f"Collection {self.ug_collection_name} exists – loading.")
                self.ug_collection = Collection(self.ug_collection_name)
                self.ug_collection.load()
                return

            fields = [
                FieldSchema(name="user_id",        dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="group_id",        dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="company_id",      dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="department_id",   dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="username",        dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="password_hash",   dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="cost_llm_tokens", dtype=DataType.INT64),
                FieldSchema(name="dummy_vector",    dtype=DataType.FLOAT_VECTOR, dim=2),
            ]
            schema = CollectionSchema(fields, description="User groups + access control (4-level)")
            self.ug_collection = Collection(self.ug_collection_name, schema=schema, using="default")
            self.ug_collection.create_index(
                field_name="dummy_vector",
                index_params={"metric_type": "L2", "index_type": "FLAT", "params": {}},
            )
            self.ug_collection.load()
            logger.info(f"✅ {self.ug_collection_name} created.")
        except Exception as e:
            logger.error(f"❌ create_user_groups_collection: {e}")
            raise

    async def _optimize_collection_index(self, collection: Collection, vector_field: str):
        try:
            indexes = collection.indexes
            if not indexes:
                collection.release()
                collection.create_index(
                    field_name=vector_field,
                    index_params={"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}},
                )
                collection.load()
                return
            for idx in indexes:
                if idx.params.get("index_type") == "IVF_FLAT":
                    collection.release()
                    collection.drop_index()
                    collection.create_index(
                        field_name=vector_field,
                        index_params={"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}},
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
        """
        Insert embeddings with optional ACL.
        acl = { group_id, company_id, department_id, user_id }
        Empty string = wildcard.
        """
        self._check_initialized()
        if not self.doc_collection or not embeddings_data:
            return 0

        acl = acl or {}
        acl_group      = (acl.get("group_id",      "") or "")[:100]
        acl_company    = (acl.get("company_id",    "") or "")[:100]
        acl_department = (acl.get("department_id", "") or "")[:100]
        acl_user       = (acl.get("user_id",       "") or "")[:100]

        field_limits = {"id": self.max_id_length, "document_id": self.max_document_id_length, "description": self.max_description_length}
        validated = [
            self._validate_and_truncate(item, field_limits)
            for item in embeddings_data
            if all(k in item for k in ["id", "document_id", "description", "description_vector"])
            and len(item["description_vector"]) == self.embedding_dim
        ]
        if not validated:
            return 0

        batch_size    = 100
        total_inserted = 0
        for i in range(0, len(validated), batch_size):
            batch = validated[i : i + batch_size]
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
            if len(faq_id) > 90:           faq_id   = faq_id[:90]
            if len(question) > self.max_question_length: question = question[:self.max_question_length-3] + "..."
            if len(answer)   > self.max_answer_length:   answer   = answer[:self.max_answer_length-3]   + "..."
            if len(question_vector) != self.embedding_dim:
                return False
            self.faq_collection.insert([[faq_id],[question],[answer],[question_vector]])
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
            document_id = document_id[:100]; url = url[:500]; filename = filename[:200]; file_type = file_type[:20]
            vec = self.embed_text(filename)
            try:
                existing = self.url_collection.query(expr=f'document_id == "{document_id}"', output_fields=["document_id"], limit=1)
                if existing:
                    self.url_collection.delete(f'document_id in ["{document_id}"]')
            except Exception:
                pass
            self.url_collection.insert([[document_id],[url],[filename],[file_type],[vec]])
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
                output_fields=["url","filename","file_type"], limit=1,
            )
            if results:
                return {"document_id": document_id, **results[0]}
            return None
        except Exception as e:
            logger.error(f"get_url error: {e}")
            return None

    # ──────────────────────────────────────────────
    # USER GROUPS
    # ──────────────────────────────────────────────

    def get_user(self, user_id: str) -> Optional[Dict]:
        try:
            results = self.ug_collection.query(
                expr=f'user_id == "{user_id}"',
                output_fields=["user_id","group_id","company_id","department_id","username","cost_llm_tokens"],
                limit=1,
            )
            return results[0] if results else None
        except Exception as e:
            logger.error(f"get_user error: {e}")
            return None

    def update_token_cost(self, user_id: str, tokens_used: int) -> bool:
        """Read-modify-write (Milvus does not support in-place update)."""
        try:
            full = self.ug_collection.query(
                expr=f'user_id == "{user_id}"',
                output_fields=["user_id","group_id","company_id","department_id",
                                "username","password_hash","cost_llm_tokens"],
                limit=1,
            )
            if not full:
                return False
            r = full[0]
            new_total = r.get("cost_llm_tokens", 0) + tokens_used
            self.ug_collection.delete(f'user_id in ["{user_id}"]')
            self.ug_collection.insert([
                [r["user_id"]], [r["group_id"]], [r["company_id"]], [r["department_id"]],
                [r["username"]], [r["password_hash"]], [new_total], [[0.0, 0.0]],
            ])
            self.ug_collection.flush()
            logger.info(f"✅ Token cost updated: {user_id} → {new_total}")
            return True
        except Exception as e:
            logger.error(f"update_token_cost error: {e}")
            return False

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
            ("user_groups",         self.ug_collection),
        ]:
            if col:
                try:
                    stats[name] = {"count": col.num_entities}
                except Exception:
                    stats[name] = {"count": "?"}
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
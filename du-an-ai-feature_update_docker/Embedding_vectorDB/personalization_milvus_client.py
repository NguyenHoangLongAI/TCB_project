# Embedding_vectorDB/personalization_milvus_client.py
"""
Milvus Manager cho Personalization Collections:
- personalization_default: Document embeddings cho personalization
- personalization_faq_embeddings: FAQ embeddings cho personalization
- personalization_document_urls: Document URLs cho personalization

Sử dụng database riêng: personalization_db
"""

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
    db
)

from typing import List, Dict, Any
import asyncio
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonalizationMilvusManager:
    """Milvus manager cho personalization collections trong database riêng"""

    def __init__(
            self,
            host: str = "localhost",
            port: str = "19530",
            embedding_dim: int = 768,
            database_name: str = "personalization_db"
    ):
        self.host = host
        self.port = port
        self.embedding_dim = embedding_dim
        self.database_name = database_name

        # Collection names - PERSONALIZATION specific
        self.doc_collection_name = "personalization_default"
        self.faq_collection_name = "personalization_faq_embeddings"
        self.url_collection_name = "personalization_document_urls"

        # Collections
        self.doc_collection = None
        self.faq_collection = None
        self.url_collection = None

        # State
        self.is_initialized = False

        # Field limits
        self.max_id_length = 190
        self.max_document_id_length = 90
        self.max_description_length = 60000
        self.max_question_length = 60000
        self.max_answer_length = 60000

        # Embedding model (lazy loading)
        self._embedding_model = None

    # ==================== CONNECTION ====================

    async def initialize(self, max_retries: int = 5, retry_delay: int = 2):
        """Initialize Milvus connection and create personalization collections in personalization_db"""
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Attempting to connect to Milvus at {self.host}:{self.port} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )

                # Auto-detect Docker environment
                if self.host == "milvus":
                    import socket
                    try:
                        socket.gethostbyname("milvus")
                    except socket.gaierror:
                        self.host = "localhost"
                        logger.warning("⚠️ Running outside Docker, using localhost")

                # Disconnect existing connection
                try:
                    connections.disconnect("default")
                except:
                    pass

                # Connect
                connections.connect("default", host=self.host, port=self.port)
                logger.info(f"✅ Connected to Milvus at {self.host}:{self.port}")

                # Switch to personalization database
                # ===== Ensure database exists =====
                try:
                    databases = db.list_database()
                    if self.database_name not in databases:
                        logger.info(f"📁 Database '{self.database_name}' not found. Creating...")
                        db.create_database(self.database_name)
                        logger.info(f"✅ Database '{self.database_name}' created")

                    db.using_database(self.database_name)
                    logger.info(f"✅ Using database: {self.database_name}")

                except Exception as e:
                    logger.error(f"❌ Failed to setup database '{self.database_name}': {e}")
                    raise

                # Create all PERSONALIZATION collections
                await self.create_document_collection()
                await self.create_faq_collection()
                await self.create_url_collection()

                self.is_initialized = True
                logger.info("✅ Personalization Milvus initialization completed")
                return True

            except Exception as e:
                logger.error(
                    f"❌ Milvus initialization error "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"❌ Failed to initialize after {max_retries} attempts")
                    self.is_initialized = False
                    raise e

    def _check_initialized(self):
        """Check if Milvus is initialized"""
        if not self.is_initialized:
            raise Exception(
                "Milvus is not initialized. Please check connection."
            )

    # ==================== EMBEDDING MODEL ====================

    @property
    def embedding_model(self):
        """Lazy load embedding model (FORCE CPU)"""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("🔄 Loading Vietnamese SBERT model (CPU mode)...")
                device = 'cpu'

                self._embedding_model = SentenceTransformer(
                    'keepitreal/vietnamese-sbert',
                    device=device
                )

                logger.info(f"✅ Embedding model loaded on {device}")

            except ImportError:
                logger.error("❌ sentence-transformers not installed")
                raise ImportError("Run: pip install sentence-transformers")
            except Exception as e:
                logger.error(f"❌ Failed to load embedding model: {e}")
                raise

        return self._embedding_model

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            if not text or not text.strip():
                return [0.0] * self.embedding_dim

            embedding = self.embedding_model.encode(
                text.strip(),
                normalize_embeddings=True
            )

            return embedding.tolist()

        except Exception as e:
            logger.error(f"❌ Embedding error: {e}")
            return [0.0] * self.embedding_dim

    # ==================== COLLECTION CREATION ====================

    async def create_document_collection(self):
        """Create personalization_default collection"""
        try:
            if utility.has_collection(self.doc_collection_name):
                logger.info(f"📦 Collection {self.doc_collection_name} already exists")
                self.doc_collection = Collection(self.doc_collection_name)
                self.doc_collection.load()
                logger.info(f"✅ Loaded existing collection {self.doc_collection_name}")
                return

            logger.info(f"🔨 Creating NEW collection: {self.doc_collection_name}")

            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    max_length=200,
                    is_primary=True
                ),
                FieldSchema(
                    name="document_id",
                    dtype=DataType.VARCHAR,
                    max_length=100
                ),
                FieldSchema(
                    name="description",
                    dtype=DataType.VARCHAR,
                    max_length=65000
                ),
                FieldSchema(
                    name="description_vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.embedding_dim
                )
            ]

            schema = CollectionSchema(
                fields=fields,
                description="Personalization document embeddings (768D)"
            )

            self.doc_collection = Collection(
                name=self.doc_collection_name,
                schema=schema,
                using='default'
            )

            # HNSW index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {
                    "M": 16,
                    "efConstruction": 200
                }
            }

            self.doc_collection.create_index(
                field_name="description_vector",
                index_params=index_params
            )

            self.doc_collection.load()
            logger.info(f"✅ Collection {self.doc_collection_name} created with HNSW index")

        except Exception as e:
            logger.error(f"❌ Document collection creation error: {e}")
            raise e

    async def create_faq_collection(self):
        """Create personalization_faq_embeddings collection"""
        try:
            if utility.has_collection(self.faq_collection_name):
                logger.info(f"📦 Collection {self.faq_collection_name} already exists")
                self.faq_collection = Collection(self.faq_collection_name)
                self.faq_collection.load()
                logger.info(f"✅ Loaded existing collection {self.faq_collection_name}")
                return

            logger.info(f"🔨 Creating NEW collection: {self.faq_collection_name}")

            fields = [
                FieldSchema(
                    name="faq_id",
                    dtype=DataType.VARCHAR,
                    max_length=100,
                    is_primary=True
                ),
                FieldSchema(
                    name="question",
                    dtype=DataType.VARCHAR,
                    max_length=65000
                ),
                FieldSchema(
                    name="answer",
                    dtype=DataType.VARCHAR,
                    max_length=65000
                ),
                FieldSchema(
                    name="question_vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.embedding_dim
                )
            ]

            schema = CollectionSchema(
                fields=fields,
                description="Personalization FAQ embeddings (768D)"
            )

            self.faq_collection = Collection(
                name=self.faq_collection_name,
                schema=schema,
                using='default'
            )

            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {
                    "M": 16,
                    "efConstruction": 200
                }
            }

            self.faq_collection.create_index(
                field_name="question_vector",
                index_params=index_params
            )

            self.faq_collection.load()
            logger.info(f"✅ Collection {self.faq_collection_name} created with HNSW index")

        except Exception as e:
            logger.error(f"❌ FAQ collection creation error: {e}")
            raise e

    async def create_url_collection(self):
        """Create personalization_document_urls collection"""
        try:
            if utility.has_collection(self.url_collection_name):
                logger.info(f"📦 Collection {self.url_collection_name} already exists")
                self.url_collection = Collection(self.url_collection_name)
                self.url_collection.load()
                logger.info(f"✅ Loaded existing collection {self.url_collection_name}")
                return

            logger.info(f"🔨 Creating NEW collection: {self.url_collection_name}")

            fields = [
                FieldSchema(
                    name="document_id",
                    dtype=DataType.VARCHAR,
                    max_length=100,
                    is_primary=True,
                    description="Unique document identifier"
                ),
                FieldSchema(
                    name="url",
                    dtype=DataType.VARCHAR,
                    max_length=500,
                    description="Public URL to the document"
                ),
                FieldSchema(
                    name="filename",
                    dtype=DataType.VARCHAR,
                    max_length=200,
                    description="Original filename"
                ),
                FieldSchema(
                    name="file_type",
                    dtype=DataType.VARCHAR,
                    max_length=20,
                    description="File extension"
                ),
                FieldSchema(
                    name="filename_vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.embedding_dim,
                    description="Filename embedding"
                )
            ]

            schema = CollectionSchema(
                fields=fields,
                description="Personalization document URLs with filename embeddings"
            )

            self.url_collection = Collection(
                name=self.url_collection_name,
                schema=schema,
                using='default'
            )

            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }

            self.url_collection.create_index(
                field_name="filename_vector",
                index_params=index_params
            )

            self.url_collection.load()
            logger.info(f"✅ Collection {self.url_collection_name} created successfully")

        except Exception as e:
            logger.error(f"❌ URL collection creation error: {e}")
            raise e

    # ==================== DOCUMENT EMBEDDINGS ====================

    async def insert_embeddings(self, embeddings_data: List[Dict]) -> int:
        """Insert document embeddings"""
        try:
            self._check_initialized()

            if not self.doc_collection:
                raise Exception("Document collection not initialized")

            try:
                self.doc_collection.load()
            except:
                pass

            if not embeddings_data:
                return 0

            field_limits = {
                "id": self.max_id_length,
                "document_id": self.max_document_id_length,
                "description": self.max_description_length
            }

            validated_data = []
            for item in embeddings_data:
                if not all(key in item for key in
                           ["id", "document_id", "description", "description_vector"]):
                    continue

                validated_item = self._validate_and_truncate(item, field_limits)

                if len(validated_item["description_vector"]) != self.embedding_dim:
                    continue

                validated_data.append(validated_item)

            if not validated_data:
                return 0

            ids = [item["id"] for item in validated_data]
            document_ids = [item["document_id"] for item in validated_data]
            descriptions = [item["description"] for item in validated_data]
            vectors = [item["description_vector"] for item in validated_data]

            batch_size = 100
            total_inserted = 0

            for i in range(0, len(validated_data), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_document_ids = document_ids[i:i + batch_size]
                batch_descriptions = descriptions[i:i + batch_size]
                batch_vectors = vectors[i:i + batch_size]

                entities = [
                    batch_ids,
                    batch_document_ids,
                    batch_descriptions,
                    batch_vectors
                ]

                try:
                    self.doc_collection.insert(entities)
                    total_inserted += len(batch_ids)
                except Exception as batch_error:
                    logger.error(f"Error inserting batch: {batch_error}")
                    continue

            self.doc_collection.flush()
            logger.info(f"✅ Total inserted to PERSONALIZATION: {total_inserted} embeddings")
            return total_inserted

        except Exception as e:
            logger.error(f"❌ Insert error: {e}")
            raise e

    async def delete_document(self, document_id: str) -> bool:
        """Delete document embeddings"""
        try:
            self._check_initialized()

            if not self.doc_collection:
                raise Exception("Document collection not initialized")

            expr = f'document_id == "{document_id}"'
            self.doc_collection.delete(expr)

            logger.info(f"✅ Deleted from PERSONALIZATION: {document_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Document delete error: {e}")
            return False

    # ==================== FAQ ====================

    async def insert_faq(
            self,
            faq_id: str,
            question: str,
            answer: str,
            question_vector: List[float]
    ) -> bool:
        """Insert FAQ"""
        try:
            self._check_initialized()

            if not self.faq_collection:
                raise Exception("FAQ collection not initialized")

            try:
                self.faq_collection.load()
            except:
                pass

            if len(faq_id) > 90:
                faq_id = faq_id[:90]
            if len(question) > self.max_question_length:
                question = question[:self.max_question_length - 3] + "..."
            if len(answer) > self.max_answer_length:
                answer = answer[:self.max_answer_length - 3] + "..."

            if len(question_vector) != self.embedding_dim:
                return False

            entities = [[faq_id], [question], [answer], [question_vector]]
            self.faq_collection.insert(entities)
            self.faq_collection.flush()

            logger.info(f"✅ Inserted FAQ to PERSONALIZATION: {faq_id}")
            return True

        except Exception as e:
            logger.error(f"❌ FAQ insert error: {e}")
            return False

    async def delete_faq(self, faq_id: str) -> bool:
        """Delete FAQ"""
        try:
            self._check_initialized()

            if not self.faq_collection:
                raise Exception("FAQ collection not initialized")

            expr = f'faq_id == "{faq_id}"'
            self.faq_collection.delete(expr)

            logger.info(f"✅ Deleted FAQ from PERSONALIZATION: {faq_id}")
            return True

        except Exception as e:
            logger.error(f"❌ FAQ delete error: {e}")
            return False

    # ==================== DOCUMENT URLS ====================

    def insert_url(
            self,
            document_id: str,
            url: str,
            filename: str = "",
            file_type: str = ""
    ) -> bool:
        """Insert document URL"""
        try:
            if not self.url_collection:
                raise Exception("URL collection not initialized")

            if len(document_id) > 100:
                document_id = document_id[:100]
            if len(url) > 500:
                url = url[:500]
            if len(filename) > 200:
                filename = filename[:200]
            if len(file_type) > 20:
                file_type = file_type[:20]

            filename_embedding = self.embed_text(filename)

            try:
                expr = f'document_id == "{document_id}"'
                existing = self.url_collection.query(
                    expr=expr,
                    output_fields=["document_id"],
                    limit=1
                )

                if existing:
                    self.url_collection.delete(expr=f'document_id in ["{document_id}"]')
            except Exception as del_error:
                pass

            entities = [
                [document_id],
                [url],
                [filename],
                [file_type],
                [filename_embedding]
            ]

            self.url_collection.insert(entities)
            self.url_collection.flush()

            logger.info(f"✅ Inserted URL to PERSONALIZATION: {document_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Error inserting URL: {e}")
            return False

    def delete_url(self, document_id: str) -> bool:
        """Delete URL entry"""
        try:
            expr = f'document_id in ["{document_id}"]'
            self.url_collection.delete(expr)
            self.url_collection.flush()
            logger.info(f"✅ Deleted URL from PERSONALIZATION: {document_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Delete error: {e}")
            return False

    # ==================== UTILITIES ====================

    def _validate_and_truncate(
            self,
            data: Dict[str, Any],
            field_limits: Dict[str, int]
    ) -> Dict[str, Any]:
        """Validate and truncate fields"""
        validated = data.copy()

        for field, max_length in field_limits.items():
            if field in validated and isinstance(validated[field], str):
                if len(validated[field]) > max_length:
                    validated[field] = validated[field][:max_length - 3] + "..."

        return validated

    async def health_check(self) -> bool:
        """Check Milvus connection health"""
        try:
            if not self.is_initialized:
                return False
            connections.get_connection_addr("default")
            return True
        except:
            return False

async def main():
    """
    Entry point để:
    - Kết nối Milvus
    - Sử dụng database personalization_db
    - Tạo toàn bộ personalization collections
    """
    logger.info("🚀 Starting Personalization Milvus setup...")

    milvus_manager = PersonalizationMilvusManager(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530"),
        embedding_dim=768,
        database_name="personalization_db"
    )

    try:
        await milvus_manager.initialize()
        logger.info("🎉 Personalization Milvus collections are READY")

        # Optional health check
        if await milvus_manager.health_check():
            logger.info("💚 Milvus health check PASSED")
        else:
            logger.warning("⚠️ Milvus health check FAILED")

    except Exception as e:
        logger.error(f"🔥 Failed to setup personalization Milvus: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

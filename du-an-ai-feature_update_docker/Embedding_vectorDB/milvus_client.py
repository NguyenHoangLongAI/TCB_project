"""
Unified Milvus Manager for all collections:
- document_embeddings: Document content embeddings
- faq_embeddings: FAQ question/answer embeddings
- document_urls: Document URLs with filename embeddings for search
"""

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
from typing import List, Dict, Any
import asyncio
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusManager:
    """Unified manager for all Milvus collections"""

    def __init__(
            self,
            host: str = "localhost",
            port: str = "19530",
            embedding_dim: int = 768
    ):
        self.host = host
        self.port = port
        self.embedding_dim = embedding_dim

        # Collection names
        self.doc_collection_name = "document_embeddings"
        self.faq_collection_name = "faq_embeddings"
        self.url_collection_name = "document_urls"

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
        """Initialize Milvus connection and create all collections"""
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

                # Create all collections
                await self.create_document_collection()
                await self.create_faq_collection()
                await self.create_url_collection()

                self.is_initialized = True
                logger.info("✅ Milvus initialization completed successfully")
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
                "Milvus is not initialized. The service may be unavailable. "
                "Please check Milvus connection and restart the application."
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
        """Create document_embeddings collection with HNSW index"""
        try:
            if utility.has_collection(self.doc_collection_name):
                logger.info(f"📦 Collection {self.doc_collection_name} already exists")
                self.doc_collection = Collection(self.doc_collection_name)
                await self._optimize_collection_index(
                    self.doc_collection,
                    "description_vector"
                )
                self.doc_collection.load()
                logger.info(f"✅ Loaded existing collection {self.doc_collection_name}")
                return

            logger.info(f"🔨 Creating NEW collection: {self.doc_collection_name}")

            # Create new collection
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
                description="Document embeddings collection (768D) - Optimized"
            )

            self.doc_collection = Collection(
                name=self.doc_collection_name,
                schema=schema,
                using='default'
            )

            # HNSW index for fast search
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
            logger.info(f"   → Total fields: {len(fields)}")
            logger.info(f"   → Index type: HNSW (M=16, efConstruction=200)")

        except Exception as e:
            logger.error(f"❌ Document collection creation error: {e}")
            raise e

    async def create_faq_collection(self):
        """Create faq_embeddings collection with HNSW index"""
        try:
            if utility.has_collection(self.faq_collection_name):
                logger.info(f"📦 Collection {self.faq_collection_name} already exists")
                self.faq_collection = Collection(self.faq_collection_name)
                await self._optimize_collection_index(
                    self.faq_collection,
                    "question_vector"
                )
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
                description="FAQ embeddings collection (768D) - Optimized"
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
            logger.info(f"   → Total fields: {len(fields)}")
            logger.info(f"   → Index type: HNSW (M=16, efConstruction=200)")

        except Exception as e:
            logger.error(f"❌ FAQ collection creation error: {e}")
            raise e

    async def create_url_collection(self):
        """Create document_urls collection with filename embeddings"""
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
                    description="File extension (.pdf, .docx, etc.)"
                ),
                FieldSchema(
                    name="filename_vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.embedding_dim,
                    description="Embedding of filename for semantic search"
                )
            ]

            schema = CollectionSchema(
                fields=fields,
                description="Document URLs with filename embeddings for semantic search"
            )

            self.url_collection = Collection(
                name=self.url_collection_name,
                schema=schema,
                using='default'
            )

            # Index for filename search
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
            logger.info(f"   → Total fields: {len(fields)}")
            logger.info(f"   → Index type: IVF_FLAT (nlist=128)")

        except Exception as e:
            logger.error(f"❌ URL collection creation error: {e}")
            raise e

    async def _optimize_collection_index(self, collection: Collection, vector_field: str):
        """Check and optimize existing index if needed"""
        try:
            indexes = collection.indexes

            if not indexes:
                logger.warning(f"No index found on {collection.name}, creating HNSW index...")
                collection.release()

                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {
                        "M": 16,
                        "efConstruction": 200
                    }
                }

                collection.create_index(
                    field_name=vector_field,
                    index_params=index_params
                )

                collection.load()
                logger.info(f"✅ Created HNSW index on {collection.name}")
                return

            # Check if using outdated IVF_FLAT
            for index in indexes:
                if index.params.get('index_type') == 'IVF_FLAT':
                    logger.warning(
                        f"Found IVF_FLAT index on {collection.name}, "
                        f"upgrading to HNSW..."
                    )

                    collection.release()
                    collection.drop_index()

                    index_params = {
                        "metric_type": "COSINE",
                        "index_type": "HNSW",
                        "params": {
                            "M": 16,
                            "efConstruction": 200
                        }
                    }

                    collection.create_index(
                        field_name=vector_field,
                        index_params=index_params
                    )

                    collection.load()
                    logger.info(f"✅ Upgraded to HNSW index on {collection.name}")
                    return

            logger.info(f"✅ Index on {collection.name} is already optimized")

        except Exception as e:
            logger.error(f"Error optimizing index: {e}")

    # ==================== DOCUMENT EMBEDDINGS ====================

    async def insert_embeddings(self, embeddings_data: List[Dict]) -> int:
        """Insert document embeddings with progress logging"""
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

            # Prepare data
            ids = [item["id"] for item in validated_data]
            document_ids = [item["document_id"] for item in validated_data]
            descriptions = [item["description"] for item in validated_data]
            vectors = [item["description_vector"] for item in validated_data]

            # Insert in batches
            batch_size = 100
            total_inserted = 0
            total_batches = (len(validated_data) + batch_size - 1) // batch_size

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

                    current_batch = (i // batch_size) + 1
                    if current_batch % 10 == 0 or current_batch == total_batches:
                        logger.info(
                            f"Inserted batch {current_batch}/{total_batches}: "
                            f"{total_inserted} items"
                        )

                except Exception as batch_error:
                    logger.error(f"Error inserting batch {i // batch_size + 1}: {batch_error}")
                    continue

            self.doc_collection.flush()
            logger.info(f"✅ Total inserted: {total_inserted} embeddings")
            return total_inserted

        except Exception as e:
            logger.error(f"❌ Insert error: {e}")
            raise e

    async def delete_document(self, document_id: str) -> bool:
        """Delete all embeddings for a document"""
        try:
            self._check_initialized()

            if not self.doc_collection:
                raise Exception("Document collection not initialized")

            expr = f'document_id == "{document_id}"'
            self.doc_collection.delete(expr)

            logger.info(f"✅ Deleted all embeddings for document_id: {document_id}")
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

            # Truncate fields
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

            logger.info(f"✅ Inserted FAQ with id: {faq_id}")
            return True

        except Exception as e:
            logger.error(f"❌ FAQ insert error: {e}")
            return False

    async def delete_faq(self, faq_id: str) -> bool:
        """Delete FAQ by ID"""
        try:
            self._check_initialized()

            if not self.faq_collection:
                raise Exception("FAQ collection not initialized")

            expr = f'faq_id == "{faq_id}"'
            self.faq_collection.delete(expr)

            logger.info(f"✅ Deleted FAQ with id: {faq_id}")
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
        """Insert or update document URL with filename embedding"""
        try:
            if not self.url_collection:
                raise Exception("URL collection not initialized")

            # Validate and truncate
            if len(document_id) > 100:
                document_id = document_id[:100]
            if len(url) > 500:
                logger.warning(f"URL too long, truncating: {url[:50]}...")
                url = url[:500]
            if len(filename) > 200:
                filename = filename[:200]
            if len(file_type) > 20:
                file_type = file_type[:20]

            # Generate filename embedding
            logger.info(f"🔄 Embedding filename: {filename}")
            filename_embedding = self.embed_text(filename)

            # Delete existing entry if any
            try:
                expr = f'document_id == "{document_id}"'
                existing = self.url_collection.query(
                    expr=expr,
                    output_fields=["document_id"],
                    limit=1
                )

                if existing:
                    self.url_collection.delete(expr=f'document_id in ["{document_id}"]')
                    logger.debug(f"Deleted existing entry for {document_id}")
            except Exception as del_error:
                logger.debug(f"No existing entry or delete failed: {del_error}")

            # Insert new entry
            entities = [
                [document_id],
                [url],
                [filename],
                [file_type],
                [filename_embedding]
            ]

            self.url_collection.insert(entities)
            self.url_collection.flush()

            logger.info(f"✅ Inserted URL for document: {document_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Error inserting URL: {e}")
            return False

    def batch_insert_urls(self, documents: list) -> int:
        """Batch insert document URLs with filename embeddings"""
        try:
            if not documents:
                return 0

            document_ids = []
            urls = []
            filenames = []
            file_types = []
            filename_vectors = []

            logger.info(f"🔄 Generating embeddings for {len(documents)} filenames...")

            for doc in documents:
                document_id = doc.get('document_id', '')[:100]
                url = doc.get('url', '')[:500]
                filename = doc.get('filename', '')[:200]
                file_type = doc.get('file_type', '')[:20]

                if document_id and url:
                    document_ids.append(document_id)
                    urls.append(url)
                    filenames.append(filename)
                    file_types.append(file_type)

                    filename_vector = self.embed_text(filename)
                    filename_vectors.append(filename_vector)

            if not document_ids:
                logger.warning("No valid documents to insert")
                return 0

            # Delete existing entries
            logger.info(f"🔄 Checking for existing entries...")
            deleted_count = 0

            for doc_id in document_ids:
                try:
                    expr = f'document_id == "{doc_id}"'
                    existing = self.url_collection.query(
                        expr=expr,
                        output_fields=["document_id"],
                        limit=1
                    )

                    if existing:
                        self.url_collection.delete(expr=f'document_id in ["{doc_id}"]')
                        deleted_count += 1
                except Exception as e:
                    logger.debug(f"Could not delete {doc_id}: {e}")
                    continue

            if deleted_count > 0:
                logger.info(f"✅ Deleted {deleted_count} existing entries")

            # Batch insert
            entities = [
                document_ids,
                urls,
                filenames,
                file_types,
                filename_vectors
            ]

            self.url_collection.insert(entities)
            self.url_collection.flush()

            logger.info(f"✅ Batch inserted {len(document_ids)} URLs with embeddings")
            return len(document_ids)

        except Exception as e:
            logger.error(f"❌ Batch insert error: {e}")
            return 0

    def search_by_filename(
            self,
            query: str,
            top_k: int = 5,
            min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Search documents by filename using semantic search"""
        try:
            query_vector = self.embed_text(query)

            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 16}
            }

            results = self.url_collection.search(
                data=[query_vector],
                anns_field="filename_vector",
                param=search_params,
                limit=top_k,
                output_fields=["document_id", "url", "filename", "file_type"]
            )

            documents = []
            for hits in results:
                for hit in hits:
                    if hit.score >= min_score:
                        documents.append({
                            "document_id": hit.entity.get("document_id"),
                            "url": hit.entity.get("url"),
                            "filename": hit.entity.get("filename"),
                            "file_type": hit.entity.get("file_type"),
                            "similarity_score": hit.score
                        })

            logger.info(f"✅ Found {len(documents)} documents for query: {query}")
            return documents

        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []

    def get_url(self, document_id: str) -> dict:
        """Get URL for a document"""
        try:
            expr = f'document_id == "{document_id}"'

            results = self.url_collection.query(
                expr=expr,
                output_fields=["url", "filename", "file_type"],
                limit=1
            )

            if results:
                return {
                    "document_id": document_id,
                    "url": results[0].get("url"),
                    "filename": results[0].get("filename"),
                    "file_type": results[0].get("file_type")
                }

            return None

        except Exception as e:
            logger.error(f"❌ Error getting URL: {e}")
            return None

    def batch_get_urls(self, document_ids: list) -> dict:
        """Get URLs for multiple documents"""
        try:
            results = {}

            for doc_id in document_ids:
                url_info = self.get_url(doc_id)
                if url_info:
                    results[doc_id] = url_info

            return results

        except Exception as e:
            logger.error(f"❌ Batch get error: {e}")
            return {}

    def delete_url(self, document_id: str) -> bool:
        """Delete URL entry"""
        try:
            expr = f'document_id in ["{document_id}"]'
            self.url_collection.delete(expr)
            self.url_collection.flush()
            logger.info(f"✅ Deleted URL for: {document_id}")
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

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        try:
            stats = {"initialized": self.is_initialized}

            if self.doc_collection:
                self.doc_collection.load()
                stats["document_embeddings"] = {
                    "count": self.doc_collection.num_entities,
                    "name": self.doc_collection_name
                }
                indexes = self.doc_collection.indexes
                if indexes:
                    stats["document_embeddings"]["index_type"] = indexes[0].params.get(
                        'index_type', 'unknown'
                    )

            if self.faq_collection:
                self.faq_collection.load()
                stats["faq_embeddings"] = {
                    "count": self.faq_collection.num_entities,
                    "name": self.faq_collection_name
                }
                indexes = self.faq_collection.indexes
                if indexes:
                    stats["faq_embeddings"]["index_type"] = indexes[0].params.get(
                        'index_type', 'unknown'
                    )

            if self.url_collection:
                self.url_collection.load()
                stats["document_urls"] = {
                    "count": self.url_collection.num_entities,
                    "name": self.url_collection_name
                }

            return stats

        except Exception as e:
            logger.error(f"❌ Stats error: {e}")
            return {"error": str(e)}


# ==================== MAIN ====================

async def main():
    """Test the Unified Milvus Manager"""

    manager = MilvusManager(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530")
    )

    # Initialize
    await manager.initialize()

    # Print stats
    stats = await manager.get_collection_stats()
    print("\n=== Collection Statistics ===")
    import json
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
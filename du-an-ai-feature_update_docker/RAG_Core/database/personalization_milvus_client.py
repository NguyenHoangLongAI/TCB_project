from pymilvus import connections, Collection, utility, db
from typing import List, Dict, Any
import numpy as np
from config.settings import settings
import logging
import os

logger = logging.getLogger(__name__)


class PersonalizationMilvusClient:
    """Milvus Client cho personalization_db"""

    def __init__(self, database_name: str = "personalization_db"):
        self.database_name = database_name
        self.alias = "personalization"
        self.connected = False
        self._connect()

    def _connect(self):
        try:
            # Clear old connection
            try:
                connections.disconnect(self.alias)
            except:
                pass

            # Connect
            connections.connect(
                alias=self.alias,
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
                db_name=self.database_name
            )
            logger.info("✅ Connected to Milvus")

            # Create DB if not exists
            if self.database_name not in db.list_database():
                logger.info(f"📁 Creating database: {self.database_name}")
                db.create_database(self.database_name)

            # Switch database
            db.using_database(self.database_name)
            logger.info(f"✅ Using database: {self.database_name}")

            self.connected = True

        except Exception as e:
            logger.error(f"❌ Milvus connection failed: {e}")
            self.connected = False

    def check_connection(self) -> bool:
        try:
            db.using_database(self.database_name)
            utility.list_collections(using=self.alias)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Lost Milvus connection: {e}")
            self.connected = False
            return False

    # ===================== SEARCH DOCUMENT =====================
    def search_documents(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:

        if not self.check_connection():
            raise ConnectionError("Milvus not connected")

        collection = Collection(
            name="personalization_default",
            using=self.alias
        )
        collection.load()

        results = collection.search(
            data=[query_vector.tolist()],
            anns_field="description_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["document_id", "description"]
        )

        docs = []
        for hits in results:
            for hit in hits:
                docs.append({
                    "document_id": hit.entity.get("document_id"),
                    "description": hit.entity.get("description"),
                    "score": hit.score
                })

        return docs

    # ===================== SEARCH FAQ =====================
    def search_faq(
        self,
        query_vector: np.ndarray,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:

        if not self.check_connection():
            raise ConnectionError("Milvus not connected")

        collection = Collection(
            name="personalization_faq_embeddings",
            using=self.alias
        )
        collection.load()

        results = collection.search(
            data=[query_vector.tolist()],
            anns_field="question_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["faq_id", "question", "answer"]
        )

        faqs = []
        for hits in results:
            for hit in hits:
                faqs.append({
                    "faq_id": hit.entity.get("faq_id"),
                    "question": hit.entity.get("question"),
                    "answer": hit.entity.get("answer"),
                    "score": hit.score
                })

        return faqs


# Global instance
personalization_milvus_client = PersonalizationMilvusClient(
    database_name=os.getenv(
        "MILVUS_PERSONALIZATION_DATABASE",
        "personalization_db"
    )
)

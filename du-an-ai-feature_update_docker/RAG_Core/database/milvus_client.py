from pymilvus import connections, Collection, utility
from typing import List, Dict, Any
import numpy as np
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class MilvusClient:
    def __init__(self):
        self.connected = False
        self.expected_dimension = None  # Will be determined from collection schema
        self._connect()

    def _connect(self):
        """Connect to Milvus"""
        try:
            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT
            )
            self.connected = True
            logger.info("Connected to Milvus successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            self.connected = False

    def check_connection(self) -> bool:
        """Check if connected to Milvus (fail fast)"""
        if not self.connected:
            return False

        try:
            utility.list_collections(timeout=2)  # ⏱ không để treo lâu
            return True
        except Exception as e:
            logger.warning(f"Lost connection to Milvus: {e}")
            self.connected = False
            return False

    def _get_collection_dimension(self, collection_name: str, vector_field: str) -> int:
        """Get the expected dimension for a vector field in a collection"""
        try:
            collection = Collection(collection_name)
            schema = collection.schema
            for field in schema.fields:
                if field.name == vector_field:
                    return field.params.get('dim', 0)
            return 0
        except Exception as e:
            logger.error(f"Error getting collection dimension: {str(e)}")
            return 0

    def _validate_vector_dimension(self, vector: np.ndarray, collection_name: str, vector_field: str,
                                   auto_fix: bool = True) -> np.ndarray:
        """Validate and potentially adjust vector dimension"""
        expected_dim = self._get_collection_dimension(collection_name, vector_field)
        actual_dim = vector.shape[0] if vector.ndim == 1 else vector.shape[1]

        if expected_dim == 0:
            logger.warning(f"Could not determine expected dimension for {collection_name}.{vector_field}")
            return vector

        if actual_dim != expected_dim:
            if auto_fix:
                logger.warning(f"Vector dimension mismatch: expected {expected_dim}, got {actual_dim}. Auto-fixing...")
                return self._adjust_vector_dimension(vector, expected_dim)
            else:
                logger.error(f"Vector dimension mismatch: expected {expected_dim}, got {actual_dim}")
                raise ValueError(f"Vector dimension mismatch: expected {expected_dim}, got {actual_dim}. "
                                 f"Please check your embedding model configuration.")

        return vector

    def _adjust_vector_dimension(self, vector: np.ndarray, target_dim: int) -> np.ndarray:
        """Adjust vector to target dimension by padding with zeros or truncating"""
        if vector.ndim > 1:
            # Handle batch vectors
            current_dim = vector.shape[1]
            if current_dim < target_dim:
                # Pad with zeros
                padding = np.zeros((vector.shape[0], target_dim - current_dim), dtype=vector.dtype)
                adjusted_vector = np.concatenate([vector, padding], axis=1)
                logger.info(f"Padded vector from {current_dim} to {target_dim} dimensions")
                return adjusted_vector
            elif current_dim > target_dim:
                # Truncate
                adjusted_vector = vector[:, :target_dim]
                logger.info(f"Truncated vector from {current_dim} to {target_dim} dimensions")
                return adjusted_vector
        else:
            # Handle single vector
            current_dim = vector.shape[0]
            if current_dim < target_dim:
                # Pad with zeros
                padding = np.zeros(target_dim - current_dim, dtype=vector.dtype)
                adjusted_vector = np.concatenate([vector, padding])
                logger.info(f"Padded vector from {current_dim} to {target_dim} dimensions")
                return adjusted_vector
            elif current_dim > target_dim:
                # Truncate
                adjusted_vector = vector[:target_dim]
                logger.info(f"Truncated vector from {current_dim} to {target_dim} dimensions")
                return adjusted_vector

        return vector

    def search_documents(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search in document collection"""
        if not self.check_connection():
            raise ConnectionError("Milvus connection lost")

        try:
            collection = Collection(settings.DOCUMENT_COLLECTION)
            collection.load()

            # Validate vector dimension
            query_vector = self._validate_vector_dimension(
                query_vector, settings.DOCUMENT_COLLECTION, "description_vector"
            )

            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 16}  # Changed from nlist to nprobe for search
            }

            results = collection.search(
                data=[query_vector.tolist()],
                anns_field="description_vector",
                param=search_params,
                limit=top_k,
                output_fields=["document_id", "description"]
            )

            documents = []
            for hits in results:
                for hit in hits:
                    documents.append({
                        "document_id": hit.entity.get("document_id"),
                        "description": hit.entity.get("description"),
                        "similarity_score": hit.score
                    })

            return documents

        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise

    def search_faq(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search in FAQ collection"""
        if not self.check_connection():
            raise ConnectionError("Milvus connection lost")

        try:
            collection = Collection(settings.FAQ_COLLECTION)
            collection.load()

            # Validate vector dimension
            query_vector = self._validate_vector_dimension(
                query_vector, settings.FAQ_COLLECTION, "question_vector"
            )

            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 16}  # Changed from nlist to nprobe for search
            }

            results = collection.search(
                data=[query_vector.tolist()],
                anns_field="question_vector",
                param=search_params,
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
                        "similarity_score": hit.score
                    })

            return faqs

        except Exception as e:
            logger.error(f"Error searching FAQ: {str(e)}")
            raise

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection including schema"""
        try:
            if not utility.has_collection(collection_name):
                return {"error": f"Collection {collection_name} does not exist"}

            collection = Collection(collection_name)
            schema = collection.schema

            fields_info = []
            for field in schema.fields:
                field_info = {
                    "name": field.name,
                    "dtype": str(field.dtype),
                    "params": field.params
                }
                fields_info.append(field_info)

            return {
                "collection_name": collection_name,
                "fields": fields_info,
                "description": schema.description
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {"error": str(e)}


# Global Milvus client instance
milvus_client = MilvusClient()
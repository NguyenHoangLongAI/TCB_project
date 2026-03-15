# RAG_Core/config/settings.py  (OPTIMIZED – search thresholds)
"""
Thay đổi thresholds dựa trên thực tế score của:
  - vietnamese-sbert COSINE: phân phối 0.2–0.5 kể cả khi match tốt
  - Cohere rerank-multilingual-v3.0: phân phối lệch về 0, score 0.02 = có liên quan
"""

from typing import Optional

try:
    from pydantic_settings import BaseSettings
    V2 = True
except Exception:
    from pydantic import BaseSettings
    V2 = False


class Settings(BaseSettings):

    # ===== Milvus – default database =====
    MILVUS_HOST:              str = "milvus"
    MILVUS_PORT:              str = "19530"
    DOCUMENT_COLLECTION:      str = "document_embeddings"
    FAQ_COLLECTION:           str = "faq_embeddings"
    DOCUMENT_URLS_COLLECTION: str = "document_urls"

    # ===== Milvus – user database =====
    USER_DATABASE_NAME: str = "user_db"

    # ===== Ollama / LLM =====
    OLLAMA_URL:      str          = "http://ollama:11434"
    LLM_MODEL:       str          = "gpt-oss:20b"
    OLLAMA_BASE_URL: Optional[str] = None

    # ===== Embedding =====
    EMBEDDING_MODEL: str = "keepitreal/vietnamese-sbert"
    EMBEDDING_DIM:   int = 768

    # ===== Search / RAG =====
    # SIMILARITY_THRESHOLD = 0.0 — không lọc trước rerank.
    # vietnamese-sbert COSINE với tiếng Việt kỹ thuật thường chỉ đạt 0.2-0.45
    # kể cả khi nội dung match hoàn toàn. Lọc threshold 0.5 sẽ loại bỏ
    # docs đúng trước khi Cohere kịp đánh giá lại.
    # → Để toàn bộ TOP_K=20 đi thẳng vào Grader/Rerank, chỉ lọc sau rerank.
    SIMILARITY_THRESHOLD: float = 0.0
    TOP_K:                int   = 20
    MAX_ITERATIONS:       int   = 5

    # ===== FAQ =====
    # FAQ_VECTOR_THRESHOLD hạ 0.5→0.3: cùng lý do trên, tránh miss FAQ đúng
    # FAQ_RERANK_THRESHOLD hạ 0.6→0.1: Cohere multilingual score thực tế
    #   rất thấp, 0.1 đã là "liên quan đáng kể"
    FAQ_VECTOR_THRESHOLD:      float = 0.3
    FAQ_TOP_K:                 int   = 10
    FAQ_RERANK_THRESHOLD:      float = 0.1
    FAQ_QUESTION_WEIGHT:       float = 0.5
    FAQ_QA_WEIGHT:             float = 0.3
    FAQ_ANSWER_WEIGHT:         float = 0.2
    FAQ_CONSISTENCY_BONUS:     float = 1.1
    FAQ_CONSISTENCY_THRESHOLD: float = 0.75

    # ===== Document Grader =====
    # Cohere rerank-multilingual-v3.0 trả về score rất thấp (0.0–0.1 là phổ biến).
    # Log thực tế: Best=0.021~0.029 → threshold 0.7 luôn fail → NOT_ENOUGH_INFO.
    # Hạ xuống 0.02 dựa trên phân phối thực tế của model.
    DOCUMENT_RERANK_THRESHOLD: float = 0.02

    # ===== MinIO / Document URLs =====
    MINIO_INTERNAL_URL:     str           = "http://localhost:9000"
    NGROK_PUBLIC_URL:       Optional[str] = "http://124.158.6.101:9000"
    ENABLE_URL_REPLACEMENT: bool          = True
    URL_FORMAT_STYLE:       str           = "detailed"
    MAX_REFERENCE_URLS:     int           = 5

    # ===== Embedding API (token tracking) =====
    EMBEDDING_API_URL: str = "http://document-api:8000"

    # ===== Cohere =====
    COHERE_API_KEY: str = "NoQ9Jjvz5r1JeRWZG8L9dnl8BxYljmnOdiUfTnfk"

    # ===== Contact =====
    SUPPORT_PHONE: str = "Phòng vận hành 0904540490 - Phòng kinh doanh:0914616081"

    # ===== Optional =====
    DOC_API_PORT: Optional[int] = None
    RAG_API_PORT: Optional[int] = None

    if V2:
        model_config = {"env_file": ".env", "extra": "ignore"}
    else:
        class Config:
            env_file = ".env"
            extra    = "ignore"

    def get_public_url(self, internal_url: str) -> str:
        if not self.ENABLE_URL_REPLACEMENT or not self.NGROK_PUBLIC_URL:
            return internal_url
        if internal_url.startswith(self.MINIO_INTERNAL_URL):
            return internal_url.replace(
                self.MINIO_INTERNAL_URL,
                self.NGROK_PUBLIC_URL.rstrip("/"),
            )
        return internal_url


settings = Settings()


def get_faq_config() -> dict:
    return {
        "vector_threshold":      settings.FAQ_VECTOR_THRESHOLD,
        "rerank_threshold":      settings.FAQ_RERANK_THRESHOLD,
        "top_k":                 settings.FAQ_TOP_K,
        "weights": {
            "question":         settings.FAQ_QUESTION_WEIGHT,
            "question_answer":  settings.FAQ_QA_WEIGHT,
            "answer":           settings.FAQ_ANSWER_WEIGHT,
        },
        "consistency_bonus":     settings.FAQ_CONSISTENCY_BONUS,
        "consistency_threshold": settings.FAQ_CONSISTENCY_THRESHOLD,
    }
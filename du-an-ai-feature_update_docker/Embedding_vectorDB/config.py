import os


class Config:
    # ===== Milvus =====
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "document_embeddings")
    FAQ_COLLECTION: str = os.getenv("FAQ_COLLECTION", "faq_embeddings")

    # ===== Embedding =====
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "keepitreal/vietnamese-sbert")
    EMBEDDING_DIMENSION: int = int(
        os.getenv("EMBEDDING_DIM", os.getenv("EMBEDDING_DIMENSION", "768"))
    )

    # ===== Document Processing =====
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "100")) * 1024 * 1024  # 100MB
    TESSDATA_PREFIX: str = os.getenv(
        "TESSDATA_PREFIX", "/usr/share/tesseract-ocr/4.00/tessdata"
    )

    # ===== DOCLING SETTINGS =====
    # Enable/disable Docling processor
    USE_DOCLING: bool = os.getenv("USE_DOCLING", "true").lower() == "true"

    # Enable OCR for scanned documents
    USE_OCR: bool = os.getenv("USE_OCR", "true").lower() == "true"

    # OCR languages (comma-separated)
    OCR_LANGUAGES: str = os.getenv("OCR_LANGUAGES", "vi,en")

    # Use GPU for processing (requires CUDA)
    USE_GPU: bool = os.getenv("USE_GPU", "true").lower() == "true"

    # Image scale for OCR (higher = better quality but slower)
    # 1.0 = 108 DPI, 2.0 = 216 DPI, 3.0 = 324 DPI
    IMAGE_SCALE: float = float(os.getenv("IMAGE_SCALE", "2.0"))

    # Force full page OCR (slower but more accurate for scanned PDFs)
    FORCE_FULL_PAGE_OCR: bool = os.getenv("FORCE_FULL_PAGE_OCR", "false").lower() == "true"

    @property
    def ocr_lang_list(self) -> list:
        """Get OCR languages as list"""
        return [lang.strip() for lang in self.OCR_LANGUAGES.split(',')]


config = Config()
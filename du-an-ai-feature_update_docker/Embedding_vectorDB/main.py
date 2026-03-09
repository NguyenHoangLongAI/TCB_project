# Embedding_vectorDB/main.py - UNIFIED API
"""
Unified Document Processing API
Gộp 3 tác vụ: Process -> Embed -> Upload MinIO -> Store URL
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import uuid
import re
from typing import Optional
import json
import logging

# Import processing modules
from document_processor import DocumentProcessor
from embedding_service import EmbeddingService
from milvus_client import MilvusManager

# Import MinIO
from minio import Minio
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unified Document Processing API",
    version="3.0.0",
    description="All-in-one: Process + Embed + Upload MinIO + Store URL"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# GLOBAL SERVICES INITIALIZATION
# =====================================================================

# Milvus Manager
milvus_host = os.getenv('MILVUS_HOST', 'localhost')
milvus_port = os.getenv('MILVUS_PORT', '19530')
milvus_manager = MilvusManager(host=milvus_host, port=milvus_port)

# Document Processor
doc_processor = DocumentProcessor(use_docling=True, use_ocr=True)

# Embedding Service
embedding_service = EmbeddingService()


# MinIO Client - FIX: Separate internal and public endpoints
def get_minio_config():
    """
    Get MinIO configuration with separate internal and public endpoints

    Returns:
        tuple: (internal_endpoint, public_endpoint, access_key, secret_key, bucket, secure)
    """
    # Internal endpoint (for Docker services to communicate)
    internal_endpoint = os.getenv('MINIO_INTERNAL_ENDPOINT', '')
    if not internal_endpoint:
        # Auto-detect
        try:
            import socket
            socket.gethostbyname('minio')
            internal_endpoint = 'minio:9000'
            logger.info("🐳 Docker environment detected")
        except (socket.gaierror, OSError):
            internal_endpoint = 'localhost:9000'
            logger.info("💻 Local environment detected")

    # Public endpoint (for users to access from browser)
    public_endpoint = os.getenv('MINIO_PUBLIC_ENDPOINT', '')
    if not public_endpoint:
        # Default to localhost for public access
        # In production, this should be your actual domain
        public_endpoint = 'localhost:9000'

    access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
    secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
    bucket = os.getenv('MINIO_BUCKET', 'public-documents')
    secure = os.getenv('MINIO_SECURE', 'false').lower() == 'true'

    logger.info(f"🔧 MinIO Configuration:")
    logger.info(f"   Internal Endpoint (Docker): {internal_endpoint}")
    logger.info(f"   Public Endpoint (Browser): {public_endpoint}")
    logger.info(f"   Bucket: {bucket}")
    logger.info(f"   Secure: {secure}")

    return internal_endpoint, public_endpoint, access_key, secret_key, bucket, secure


# Get MinIO configuration
minio_internal_endpoint, minio_public_endpoint, minio_access_key, minio_secret_key, minio_bucket, minio_secure = get_minio_config()

# MinIO client uses INTERNAL endpoint for operations
minio_client = Minio(
    minio_internal_endpoint,
    access_key=minio_access_key,
    secret_key=minio_secret_key,
    secure=minio_secure
)

# Ensure bucket exists
try:
    if not minio_client.bucket_exists(minio_bucket):
        minio_client.make_bucket(minio_bucket)
        logger.info(f"✅ Created MinIO bucket: {minio_bucket}")

    # Set public read policy
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{minio_bucket}/*"]
            }
        ]
    }
    minio_client.set_bucket_policy(minio_bucket, json.dumps(policy))
    logger.info(f"✅ MinIO bucket is public: {minio_bucket}")
except Exception as e:
    logger.warning(f"⚠️ MinIO setup warning: {e}")


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to be safe"""
    if not filename:
        return "unknown_file"

    name, ext = os.path.splitext(filename)
    safe_name = re.sub(r'[^\w\-_.]', '_', name)
    safe_name = re.sub(r'_+', '_', safe_name)
    safe_name = safe_name.strip('_')

    if not safe_name:
        safe_name = "document"

    return safe_name + ext.lower()


def sanitize_id(text: str) -> str:
    """Sanitize document ID"""
    sanitized = re.sub(r"[^\w\-_.]", "_", text)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")


def get_safe_temp_filename(original_filename: str) -> str:
    """Generate safe temporary filename"""
    _, ext = os.path.splitext(original_filename)
    unique_id = str(uuid.uuid4())[:8]
    return f"temp_doc_{unique_id}{ext.lower()}"


def upload_to_minio(file_path: str, document_id: str) -> str:
    """
    Upload file to MinIO and return PUBLIC URL (accessible from browser)

    Args:
        file_path: Local file path
        document_id: Document identifier

    Returns:
        Public URL to the file (using public endpoint)
    """
    try:
        file_ext = Path(file_path).suffix.lower()
        object_name = f"{document_id}{file_ext}"

        # Content type mapping
        content_types = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
            '.txt': 'text/plain'
        }

        content_type = content_types.get(file_ext, 'application/octet-stream')

        # Upload to MinIO using INTERNAL endpoint
        logger.info(f"📤 Uploading {object_name} to MinIO bucket: {minio_bucket}")
        logger.info(f"   Using internal endpoint: {minio_internal_endpoint}")

        minio_client.fput_object(
            minio_bucket,
            object_name,
            file_path,
            content_type=content_type
        )

        # Generate PUBLIC URL using public endpoint
        protocol = "https" if minio_secure else "http"
        public_url = f"{protocol}://{minio_public_endpoint}/{minio_bucket}/{object_name}"

        logger.info(f"✅ Uploaded to MinIO: {object_name}")
        logger.info(f"🔗 Public URL (for browser): {public_url}")

        return public_url

    except Exception as e:
        logger.error(f"❌ MinIO upload failed: {e}")
        raise Exception(f"MinIO upload error: {e}")


# =====================================================================
# STARTUP EVENT
# =====================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize Milvus connection and create collections"""
    try:
        await milvus_manager.initialize()
        logger.info("✅ Unified Document API started successfully")
        logger.info(f"✅ Milvus connected: {milvus_host}:{milvus_port}")
        logger.info(f"✅ MinIO connected (internal): {minio_internal_endpoint}")
        logger.info(f"✅ MinIO public endpoint: {minio_public_endpoint}")
        logger.info("✅ Smart Chunking enabled")
    except Exception as e:
        logger.error(f"⚠️ Warning during startup: {e}")


# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Unified Document Processing API",
        "version": "3.0.0",
        "status": "running",
        "features": {
            "document_processing": "enabled",
            "smart_chunking": "enabled",
            "embedding": "enabled",
            "minio_storage": "enabled",
            "url_management": "enabled"
        },
        "endpoints": {
            "process_document": "/api/v1/process-document (POST)",
            "add_faq": "/api/v1/faq/add (POST)",
            "delete_faq": "/api/v1/faq/delete/{faq_id} (DELETE)",
            "delete_document": "/api/v1/document/delete/{document_id} (DELETE)",
            "health": "/api/v1/health (GET)"
        }
    }


@app.post("/api/v1/process-document")
async def process_document(
        file: UploadFile = File(...),
        document_id: Optional[str] = Form(None),
        chunk_mode: str = Form("smart")
):
    """
    🔥 UNIFIED API: Process + Embed + Upload MinIO + Store URL

    Steps:
    1. Process document to markdown
    2. Create embeddings with smart chunking
    3. Upload original file to MinIO
    4. Store URL in document_urls collection

    Args:
        file: Document file (PDF, DOCX, XLSX, TXT)
        document_id: Optional custom document ID
        chunk_mode: Chunking mode (smart|sentence|legacy)

    Returns:
        Complete processing result with public URL
    """

    temp_file_path = None

    try:
        # ============================================================
        # VALIDATION
        # ============================================================

        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        allowed_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.txt']
        original_filename = file.filename
        file_extension = os.path.splitext(original_filename)[1].lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported. Allowed: {allowed_extensions}"
            )

        # Validate chunk_mode
        valid_modes = ["smart", "sentence", "legacy"]
        if chunk_mode not in valid_modes:
            raise HTTPException(
                status_code=400,
                detail=f"chunk_mode must be one of: {valid_modes}"
            )

        # Generate or sanitize document_id
        if document_id:
            document_id = sanitize_id(document_id)
        else:
            # Use filename without extension as document_id
            document_id = sanitize_id(os.path.splitext(original_filename)[0])

        if not document_id:
            document_id = f"doc_{str(uuid.uuid4())[:8]}"

        logger.info(f"📄 Processing: {original_filename} -> {document_id}")

        # ============================================================
        # STEP 1: SAVE TEMPORARY FILE
        # ============================================================

        safe_temp_name = get_safe_temp_filename(original_filename)
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, safe_temp_name)

        content = await file.read()

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(content)

        logger.info(f"✅ [1/4] Saved temporary file")

        # ============================================================
        # STEP 2: PROCESS DOCUMENT TO MARKDOWN
        # ============================================================

        logger.info(f"📝 [2/4] Processing document...")

        if file_extension == '.pdf':
            markdown_content = doc_processor.process_pdf(temp_file_path)
        elif file_extension in ['.doc', '.docx']:
            markdown_content = doc_processor.process_word(temp_file_path)
        elif file_extension in ['.xls', '.xlsx']:
            markdown_content = doc_processor.process_excel(temp_file_path)
        elif file_extension == '.txt':
            text_content = None
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    with open(temp_file_path, 'r', encoding=encoding) as f:
                        text_content = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            if text_content is None:
                raise HTTPException(status_code=400, detail="Could not decode text file")

            markdown_content = doc_processor.process_text(text_content)

        if not markdown_content or len(markdown_content.strip()) == 0:
            raise HTTPException(
                status_code=422,
                detail="Could not extract content from file"
            )

        logger.info(f"✅ [2/4] Document processed: {len(markdown_content)} chars")

        # ============================================================
        # STEP 3: CREATE EMBEDDINGS WITH SMART CHUNKING
        # ============================================================

        logger.info(f"🔗 [3/4] Creating embeddings (mode: {chunk_mode})...")

        # Parse markdown to chunks
        if chunk_mode == "smart":
            chunks = doc_processor.parse_markdown_to_chunks(markdown_content, filename=original_filename)
        elif chunk_mode == "sentence":
            if hasattr(doc_processor, 'parse_markdown_to_sentences'):
                chunks = doc_processor.parse_markdown_to_sentences(markdown_content)
            else:
                logger.warning("⚠️ sentence mode not available, using smart")
                chunks = doc_processor.parse_markdown_to_chunks(markdown_content, filename=original_filename)
        else:  # legacy
            if hasattr(doc_processor, 'parse_markdown_to_sentences'):
                chunks = doc_processor.parse_markdown_to_sentences(markdown_content)
            else:
                chunks = doc_processor.parse_markdown_to_chunks(markdown_content, filename=original_filename)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="Could not parse markdown into chunks"
            )

        logger.info(f"✅ Created {len(chunks)} chunks")

        # Generate embeddings
        embeddings_data = []
        successful_embeddings = 0

        for i, chunk in enumerate(chunks):
            try:
                embedding = embedding_service.get_embedding(chunk['content'])

                metadata = {
                    "section_title": chunk.get('section_title', 'Unknown Section'),
                    "chunk_index": i,
                    "content_length": len(chunk['content']),
                    "chunk_mode": chunk_mode
                }

                if chunk_mode == "smart":
                    metadata.update({
                        "token_count": chunk.get('token_count', 0),
                        "chunk_type": chunk.get('chunk_type', 'unknown'),
                        "context": chunk.get('context', ''),
                        "level": chunk.get('level', 0)
                    })

                    if 'context_path' in chunk:
                        metadata["context_path"] = " > ".join(chunk['context_path'])

                embedding_data = {
                    "id": f"{document_id}_{chunk_mode}_{i}",
                    "document_id": document_id,
                    "description": chunk['content'],
                    "description_vector": embedding,
                    "metadata": metadata
                }

                embeddings_data.append(embedding_data)
                successful_embeddings += 1

            except Exception as e:
                logger.error(f"❌ Embedding error for chunk {i}: {e}")
                continue

        if not embeddings_data:
            raise HTTPException(
                status_code=422,
                detail="Could not create embeddings for any chunks"
            )

        # Store embeddings in Milvus
        stored_count = await milvus_manager.insert_embeddings(embeddings_data)

        logger.info(f"✅ [3/4] Embeddings stored: {stored_count} vectors")

        # ============================================================
        # STEP 4: UPLOAD TO MINIO & STORE URL
        # ============================================================

        logger.info(f"📤 [4/4] Uploading to MinIO...")

        # Upload file - returns PUBLIC URL
        public_url = upload_to_minio(temp_file_path, document_id)

        # Store URL in document_urls collection
        safe_filename = sanitize_filename(original_filename)

        url_stored = milvus_manager.insert_url(
            document_id=document_id,
            url=public_url,
            filename=safe_filename,
            file_type=file_extension
        )

        if not url_stored:
            logger.warning("⚠️ URL storage failed but continuing...")

        logger.info(f"✅ [4/4] File uploaded and URL stored")

        # ============================================================
        # PREPARE RESPONSE
        # ============================================================

        return {
            "status": "success",
            "message": "Document processed, embedded, and uploaded successfully",
            "document_info": {
                "document_id": document_id,
                "original_filename": original_filename,
                "safe_filename": safe_filename,
                "file_type": file_extension,
                "file_size_bytes": len(content)
            },
            "processing_stats": {
                "markdown_length": len(markdown_content),
                "total_chunks": len(chunks),
                "successful_embeddings": successful_embeddings,
                "stored_embeddings": stored_count,
                "chunk_mode": chunk_mode,
                "processor": "docling+smart_chunker"
            },
            "storage": {
                "public_url": public_url,
                "storage_provider": "minio",
                "bucket": minio_bucket,
                "url_stored_in_milvus": url_stored,
                "note": "URL is accessible from browser"
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    finally:
        # Cleanup temporary file
        try:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                logger.debug(f"🗑️ Cleaned up: {temp_file_path}")
        except Exception as cleanup_error:
            logger.warning(f"⚠️ Cleanup warning: {cleanup_error}")


@app.post("/api/v1/faq/add")
async def add_faq(request: dict):
    """
    Add FAQ - Thêm câu hỏi và câu trả lời FAQ
    """
    try:
        question = request.get("question", "").strip()
        answer = request.get("answer", "").strip()
        faq_id = request.get("faq_id", "").strip()

        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        if not answer:
            raise HTTPException(status_code=400, detail="Answer is required")

        if not faq_id:
            faq_id = f"faq_{str(uuid.uuid4())[:8]}"
        else:
            faq_id = sanitize_id(faq_id)

        question_embedding = embedding_service.get_embedding(question)
        success = await milvus_manager.insert_faq(
            faq_id, question, answer, question_embedding
        )

        return {
            "status": "success",
            "faq_id": faq_id,
            "question": question,
            "answer": answer,
            "message": "FAQ added successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Add FAQ error: {str(e)}")


@app.delete("/api/v1/faq/delete/{faq_id}")
async def delete_faq(faq_id: str):
    """Delete FAQ by ID"""
    try:
        if not faq_id or not faq_id.strip():
            raise HTTPException(status_code=400, detail="FAQ ID is required")

        faq_id = faq_id.strip()
        success = await milvus_manager.delete_faq(faq_id)

        return {
            "status": "success",
            "faq_id": faq_id,
            "message": "FAQ deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete FAQ error: {str(e)}")


@app.delete("/api/v1/document/delete/{document_id}")
async def delete_document(document_id: str):
    """
    Delete document - Xóa embeddings + URL
    """
    try:
        if not document_id or not document_id.strip():
            raise HTTPException(status_code=400, detail="Document ID is required")

        document_id = document_id.strip()

        # Delete embeddings
        success = await milvus_manager.delete_document(document_id)

        # Delete URL
        milvus_manager.delete_url(document_id)

        return {
            "status": "success",
            "document_id": document_id,
            "message": "Document embeddings and URL deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete document error: {str(e)}")


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    try:
        milvus_status = await milvus_manager.health_check()
        embedding_status = embedding_service.is_ready()

        # Check MinIO
        minio_status = False
        try:
            minio_client.bucket_exists(minio_bucket)
            minio_status = True
        except:
            pass

        return {
            "status": "healthy",
            "service": "unified-document-api",
            "version": "3.0.0",
            "port": 8000,
            "services": {
                "milvus": milvus_status,
                "embedding_model": embedding_status,
                "minio": minio_status,
                "document_processor": "ready",
                "smart_chunking": "enabled"
            },
            "environment": {
                "milvus_host": os.getenv('MILVUS_HOST', 'milvus'),
                "milvus_port": os.getenv('MILVUS_PORT', '19530'),
                "minio_internal_endpoint": minio_internal_endpoint,
                "minio_public_endpoint": minio_public_endpoint,
                "minio_bucket": minio_bucket
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "unified-document-api",
            "port": 8000,
            "error": str(e)
        }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
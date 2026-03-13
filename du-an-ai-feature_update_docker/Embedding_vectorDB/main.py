# Embedding_vectorDB/main.py  (UPDATED – user_groups + ACL + /create/user)
"""
Unified Document Processing API  v4.0
New:
  • /create/user            – tạo user mới (tokens bắt đầu = 0)
  • /api/v1/process-document – thêm trường user_id, tự build ACL từ user_groups
  • user_groups collection   – seed 5 users mẫu lúc startup
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, tempfile, os, uuid, re, json, hashlib, logging
from typing import Optional
from pathlib import Path

from document_processor import DocumentProcessor
from embedding_service   import EmbeddingService
from milvus_client       import MilvusManager          # updated version
from user_groups_collection import UserGroupsManager, seed_sample_users

from minio import Minio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Unified Document Processing API", version="4.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────────────

milvus_host = os.getenv("MILVUS_HOST", "localhost")
milvus_port = os.getenv("MILVUS_PORT", "19530")

milvus_manager = MilvusManager(host=milvus_host, port=milvus_port)
doc_processor  = DocumentProcessor(use_docling=True, use_ocr=True)
embedding_svc  = EmbeddingService()
user_mgr       = UserGroupsManager(host=milvus_host, port=milvus_port)


def get_minio_config():
    internal = os.getenv("MINIO_INTERNAL_ENDPOINT", "")
    if not internal:
        import socket
        try:
            socket.gethostbyname("minio")
            internal = "minio:9000"
        except Exception:
            internal = "localhost:9000"
    public      = os.getenv("MINIO_PUBLIC_ENDPOINT", "localhost:9000")
    access_key  = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key  = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    bucket      = os.getenv("MINIO_BUCKET", "public-documents")
    secure      = os.getenv("MINIO_SECURE", "false").lower() == "true"
    return internal, public, access_key, secret_key, bucket, secure


minio_internal, minio_public, minio_ak, minio_sk, minio_bucket, minio_secure = get_minio_config()
minio_client = Minio(minio_internal, access_key=minio_ak, secret_key=minio_sk, secure=minio_secure)

try:
    if not minio_client.bucket_exists(minio_bucket):
        minio_client.make_bucket(minio_bucket)
    policy = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"*"},"Action":["s3:GetObject"],"Resource":[f"arn:aws:s3:::{minio_bucket}/*"]}]}
    minio_client.set_bucket_policy(minio_bucket, json.dumps(policy))
except Exception as e:
    logger.warning(f"MinIO setup: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    await milvus_manager.initialize()
    logger.info("Seeding sample users …")
    seed_sample_users(user_mgr)
    logger.info("✅ Unified Document API v4.0 ready")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_filename(name: str) -> str:
    n, ext = os.path.splitext(name)
    safe = re.sub(r"[^\w\-_.]", "_", n)
    safe = re.sub(r"_+", "_", safe).strip("_") or "document"
    return safe + ext.lower()


def sanitize_id(text: str) -> str:
    s = re.sub(r"[^\w\-_.]", "_", text)
    return re.sub(r"_+", "_", s).strip("_")


def upload_to_minio(file_path: str, document_id: str) -> str:
    ext = Path(file_path).suffix.lower()
    obj = f"{document_id}{ext}"
    ct_map = {".pdf":"application/pdf",".docx":"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
              ".doc":"application/msword",".xlsx":"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
              ".xls":"application/vnd.ms-excel",".txt":"text/plain"}
    minio_client.fput_object(minio_bucket, obj, file_path, content_type=ct_map.get(ext, "application/octet-stream"))
    proto = "https" if minio_secure else "http"
    return f"{proto}://{minio_public}/{minio_bucket}/{obj}"

# ─────────────────────────────────────────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"service": "Unified Document Processing API", "version": "4.0.0",
            "endpoints": {
                "process_document": "POST /api/v1/process-document (requires user_id)",
                "create_user":      "POST /create/user",
                "get_user":         "GET  /user/{user_id}",
                "list_users":       "GET  /users",
                "health":           "GET  /api/v1/health",
            }}

# ─────────────────────────────────────────────────────────────────────────────
# USER MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/create/user")
async def create_user(request: dict):
    """
    Tạo user mới với cost_llm_tokens khởi điểm = 0.

    Body:
      user_id       : string (required)
      username      : string (required)
      password      : string (required)
      group_id      : string (required)  – e.g. "Techcombank_group"
      company_id    : string | null      – empty = wildcard
      department_id : string | null      – empty = wildcard
    """
    user_id       = (request.get("user_id", "") or "").strip()
    username      = (request.get("username", "") or "").strip()
    password      = (request.get("password", "") or "").strip()
    group_id      = (request.get("group_id", "") or "").strip()
    company_id    = request.get("company_id")    or ""
    department_id = request.get("department_id") or ""

    if not user_id:   raise HTTPException(400, "user_id is required")
    if not username:  raise HTTPException(400, "username is required")
    if not password:  raise HTTPException(400, "password is required")
    if not group_id:  raise HTTPException(400, "group_id is required")

    ok = user_mgr.create_user(
        user_id=sanitize_id(user_id),
        username=username,
        password=password,
        group_id=group_id,
        company_id=company_id,
        department_id=department_id,
        initial_tokens=0,
    )
    if not ok:
        raise HTTPException(409, f"user_id '{user_id}' already exists")

    return {
        "status":  "success",
        "message": f"User '{user_id}' created successfully",
        "user": {
            "user_id":       user_id,
            "username":      username,
            "group_id":      group_id,
            "company_id":    company_id    or None,
            "department_id": department_id or None,
            "cost_llm_tokens": 0,
        },
    }


@app.get("/user/{user_id}")
async def get_user(user_id: str):
    u = user_mgr.get_user(user_id)
    if not u:
        raise HTTPException(404, f"user_id '{user_id}' not found")
    return {"status": "success", "user": u}


@app.post("/user/{user_id}/tokens")
async def increment_tokens(user_id: str, request: dict):
    """
    Increment cost_llm_tokens for a user.
    Called by RAG API after each LLM invocation.

    Body: { "tokens_used": int }
    """
    tokens_used = int(request.get("tokens_used", 0))
    if tokens_used <= 0:
        return {"status": "skipped", "reason": "tokens_used <= 0"}

    ok = user_mgr.update_token_cost(user_id, tokens_used)
    if not ok:
        raise HTTPException(404, f"user_id '{user_id}' not found or update failed")

    updated_user = user_mgr.get_user(user_id)
    return {
        "status": "success",
        "user_id": user_id,
        "tokens_added": tokens_used,
        "total_tokens": updated_user.get("cost_llm_tokens", 0) if updated_user else "unknown",
    }


@app.get("/users")
async def list_users(limit: int = 100):
    users = user_mgr.list_users(limit=limit)
    return {"status": "success", "count": len(users), "users": users}


@app.post("/auth/login")
async def login(request: dict):
    """Authenticate user. Returns user info on success."""
    username = (request.get("username") or "").strip()
    password = (request.get("password") or "").strip()
    if not username or not password:
        raise HTTPException(400, "username and password required")
    user = user_mgr.authenticate(username, password)
    if not user:
        raise HTTPException(401, "Invalid username or password")
    return {"status": "success", "user": user}

# ─────────────────────────────────────────────────────────────────────────────
# PROCESS DOCUMENT  (ACL-aware)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/process-document")
async def process_document(
    file:        UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    chunk_mode:  str = Form("smart"),
    user_id:     Optional[str] = Form(None),   # NEW
    # Explicit ACL override (optional – if not provided, inherits from user_groups)
    acl_group_id:      Optional[str] = Form(None),
    acl_company_id:    Optional[str] = Form(None),
    acl_department_id: Optional[str] = Form(None),
    acl_user_id:       Optional[str] = Form(None),
):
    """
    Unified: Process → Embed → Upload MinIO → Store URL
    Now requires user_id; ACL is derived from user_groups unless overridden.
    """
    temp_file_path = None
    try:
        # ── Validate file ────────────────────────────────────────────────────
        if not file.filename:
            raise HTTPException(400, "No file provided")

        allowed = [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".txt"]
        original_filename = file.filename
        file_ext = os.path.splitext(original_filename)[1].lower()
        if file_ext not in allowed:
            raise HTTPException(400, f"File type {file_ext} not supported")

        if chunk_mode not in ["smart", "sentence", "legacy"]:
            raise HTTPException(400, "chunk_mode must be smart|sentence|legacy")

        # ── Resolve document_id ──────────────────────────────────────────────
        if document_id:
            document_id = sanitize_id(document_id)
        else:
            document_id = sanitize_id(os.path.splitext(original_filename)[0])
        if not document_id:
            document_id = f"doc_{str(uuid.uuid4())[:8]}"

        # ── Resolve ACL from user_groups ────────────────────────────────────
        acl: dict = {}
        uploader = user_id or ""

        if user_id:
            perms = user_mgr.get_user_permissions(user_id)
            if perms:
                acl = {
                    "group_id":      acl_group_id      or perms.get("group_id",      ""),
                    "company_id":    acl_company_id    or perms.get("company_id",    ""),
                    "department_id": acl_department_id or perms.get("department_id", ""),
                    "user_id":       acl_user_id       or "",   # leave empty = dept-wide by default
                }
            else:
                logger.warning(f"user_id '{user_id}' not in user_groups, using empty ACL")
        else:
            # No user_id → public document (no ACL restriction)
            acl = {
                "group_id": acl_group_id or "", "company_id": acl_company_id or "",
                "department_id": acl_department_id or "", "user_id": acl_user_id or "",
            }

        logger.info(f"📄 Processing: {original_filename} | doc_id={document_id} | user={user_id} | ACL={acl}")

        # ── [1] Save temp file ───────────────────────────────────────────────
        temp_file_path = os.path.join(tempfile.gettempdir(), f"tmp_{uuid.uuid4().hex[:8]}{file_ext}")
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(400, "Uploaded file is empty")
        with open(temp_file_path, "wb") as f:
            f.write(content)

        # ── [2] Process document ─────────────────────────────────────────────
        if file_ext == ".pdf":
            markdown_content = doc_processor.process_pdf(temp_file_path)
        elif file_ext in [".doc", ".docx"]:
            markdown_content = doc_processor.process_word(temp_file_path)
        elif file_ext in [".xls", ".xlsx"]:
            markdown_content = doc_processor.process_excel(temp_file_path)
        else:
            text_content = None
            for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
                try:
                    with open(temp_file_path, "r", encoding=enc) as f:
                        text_content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            if text_content is None:
                raise HTTPException(400, "Could not decode text file")
            markdown_content = doc_processor.process_text(text_content)

        if not markdown_content or not markdown_content.strip():
            raise HTTPException(422, "Could not extract content")

        # ── [3] Chunk + Embed ────────────────────────────────────────────────
        if chunk_mode == "smart":
            chunks = doc_processor.parse_markdown_to_chunks(markdown_content, filename=original_filename)
        else:
            chunks = doc_processor.parse_markdown_to_chunks(markdown_content, filename=original_filename)

        if not chunks:
            raise HTTPException(422, "Could not parse markdown into chunks")

        embeddings_data = []
        for i, chunk in enumerate(chunks):
            embedding = embedding_svc.get_embedding(chunk["content"])
            embeddings_data.append({
                "id":          f"{document_id}_{chunk_mode}_{i}",
                "document_id": document_id,
                "description": chunk["content"],
                "description_vector": embedding,
            })

        if not embeddings_data:
            raise HTTPException(422, "Could not create embeddings")

        stored = await milvus_manager.insert_embeddings(
            embeddings_data, acl=acl, uploaded_by=uploader
        )

        # ── [4] Upload MinIO + Store URL ─────────────────────────────────────
        public_url   = upload_to_minio(temp_file_path, document_id)
        safe_fname   = sanitize_filename(original_filename)
        url_stored   = milvus_manager.insert_url(document_id, public_url, safe_fname, file_ext)

        return {
            "status":  "success",
            "message": "Document processed, embedded, and uploaded successfully",
            "document_info": {
                "document_id":       document_id,
                "original_filename": original_filename,
                "file_type":         file_ext,
                "file_size_bytes":   len(content),
                "uploaded_by":       uploader,
            },
            "acl": acl,
            "processing_stats": {
                "markdown_length":      len(markdown_content),
                "total_chunks":         len(chunks),
                "successful_embeddings": len(embeddings_data),
                "stored_embeddings":    stored,
            },
            "storage": {
                "public_url":           public_url,
                "url_stored_in_milvus": url_stored,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)
        raise HTTPException(500, f"Processing error: {e}")
    finally:
        try:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# FAQ / DOCUMENT DELETE  (unchanged logic, kept minimal)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/faq/add")
async def add_faq(request: dict):
    question = (request.get("question") or "").strip()
    answer   = (request.get("answer")   or "").strip()
    faq_id   = sanitize_id(request.get("faq_id") or f"faq_{str(uuid.uuid4())[:8]}")
    if not question: raise HTTPException(400, "question required")
    if not answer:   raise HTTPException(400, "answer required")
    vec  = embedding_svc.get_embedding(question)
    await milvus_manager.insert_faq(faq_id, question, answer, vec)
    return {"status": "success", "faq_id": faq_id}


@app.delete("/api/v1/faq/delete/{faq_id}")
async def delete_faq(faq_id: str):
    await milvus_manager.delete_faq(faq_id.strip())
    return {"status": "success", "faq_id": faq_id}


@app.delete("/api/v1/document/delete/{document_id}")
async def delete_document(document_id: str):
    await milvus_manager.delete_document(document_id.strip())
    milvus_manager.delete_url(document_id.strip())
    return {"status": "success", "document_id": document_id}

# ─────────────────────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/health")
async def health_check():
    milvus_ok = await milvus_manager.health_check()
    emb_ok    = embedding_svc.is_ready()
    minio_ok  = False
    try:
        minio_client.bucket_exists(minio_bucket)
        minio_ok = True
    except Exception:
        pass
    return {
        "status":  "healthy" if (milvus_ok and emb_ok and minio_ok) else "degraded",
        "version": "4.0.0",
        "services": {
            "milvus":            milvus_ok,
            "embedding_model":   emb_ok,
            "minio":             minio_ok,
            "user_groups":       True,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
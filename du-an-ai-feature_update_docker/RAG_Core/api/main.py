# RAG_Core/api/main.py  (FIXED – EMBEDDING_API_BASE from env, token tracking all paths)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import logging, os, json, asyncio, httpx
from typing import List, AsyncIterator, Optional

from .schemas import ChatRequest, ChatResponse, StreamChunk, HealthResponse, DocumentReference
from workflow.rag_workflow import RAGWorkflow
from database.milvus_client import milvus_client
from services.document_url_service import document_url_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Multi-Agent Chatbot API",
    description="RAG chatbot with streaming, ACL, and token tracking",
    version="2.3.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

rag_workflow: Optional[RAGWorkflow] = None

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global rag_workflow
    try:
        rag_workflow = RAGWorkflow()
        logger.info("✅ RAG Workflow ready")
    except Exception as e:
        logger.error(f"⚠️  RAG Workflow init failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TOKEN TRACKING HELPER
# ─────────────────────────────────────────────────────────────────────────────

# ✅ FIX: đọc từ env var — khi Docker: http://document-api:8000
#                           khi local:  http://localhost:8000
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_URL", "http://localhost:8000").rstrip("/")
logger.info(f"📡 Token tracking endpoint: {EMBEDDING_API_BASE}")


async def track_token_usage(user_id: str, tokens_used: int):
    """Update cost_llm_tokens for user_id via Embedding API."""
    if not user_id or tokens_used <= 0:
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{EMBEDDING_API_BASE}/user/{user_id}/tokens",
                json={"tokens_used": tokens_used},
            )
            if resp.status_code == 200:
                logger.info(f"💰 Token tracked: user={user_id} tokens={tokens_used}")
            else:
                logger.warning(
                    f"Token tracking non-200: user={user_id} status={resp.status_code} body={resp.text[:200]}"
                )
    except Exception as e:
        logger.warning(f"Token tracking failed for {user_id}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def enrich_references_with_urls(references: List[dict]) -> List[dict]:
    try:
        return document_url_service.enrich_references_with_urls(references)
    except Exception as e:
        logger.error(f"enrich_references error: {e}")
        return references


async def generate_streaming_response(
    question: str, history: List, user_id: Optional[str]
) -> AsyncIterator[str]:
    """SSE generator — captures __token_usage__ sentinel from answer_stream."""
    try:
        yield f"data: {json.dumps({'type':'start','content':None,'references':None,'status':'processing'})}\n\n"
        await asyncio.sleep(0.01)

        result        = await rag_workflow.run_with_streaming(question, history, user_id=user_id)
        answer_stream = result.get("answer_stream")
        references    = result.get("references", [])

        total_tokens = 0

        if answer_stream:
            async for item in answer_stream:
                if isinstance(item, dict) and "__token_usage__" in item:
                    # ✅ Sentinel từ generator/faq — capture tokens
                    total_tokens = item["__token_usage__"]
                    logger.info(f"📊 Streaming captured {total_tokens} tokens (user={user_id})")
                elif isinstance(item, str) and item:
                    yield f"data: {json.dumps({'type':'chunk','content':item,'references':None,'status':None})}\n\n"
                    await asyncio.sleep(0.001)

        # References
        if references:
            enriched = enrich_references_with_urls(references)
            serial_refs = []
            for ref in enriched:
                r = {
                    "document_id": ref.get("document_id", ""),
                    "type":        ref.get("type", "DOCUMENT"),
                    "description": ref.get("description", ""),
                }
                if ref.get("url"):
                    r.update({
                        "url":       ref["url"],
                        "filename":  ref.get("filename", ""),
                        "file_type": ref.get("file_type", ""),
                    })
                serial_refs.append(r)
            yield f"data: {json.dumps({'type':'references','content':None,'references':serial_refs,'status':None})}\n\n"

        # ✅ Track tokens AFTER stream completes
        if total_tokens > 0 and user_id:
            await track_token_usage(user_id, total_tokens)

        end_payload = {
            "type":        "end",
            "content":     None,
            "references":  None,
            "status":      result.get("status", "SUCCESS"),
            "token_usage": {"total_tokens": total_tokens, "user_id": user_id} if total_tokens else None,
        }
        yield f"data: {json.dumps(end_payload)}\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        yield f"data: {json.dumps({'type':'error','content':f'Lỗi: {e}','references':None,'status':'ERROR'})}\n\n"

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service":  "RAG Multi-Agent Chatbot API",
        "version":  "2.3.0",
        "token_tracking_endpoint": EMBEDDING_API_BASE,
        "endpoints": {"chat": "/chat", "health": "/health"},
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    if not rag_workflow:
        raise HTTPException(503, "Workflow not initialized")

    user_id = (request.user_id or "").strip() or None
    logger.info(f"📨 question={request.question[:80]} stream={request.stream} user_id={user_id}")

    # ── STREAMING ────────────────────────────────────────────────────────────
    if request.stream:
        return StreamingResponse(
            generate_streaming_response(request.question, request.history, user_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control":    "no-cache",
                "Connection":       "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── NON-STREAMING ────────────────────────────────────────────────────────
    result       = rag_workflow.run(request.question, request.history, user_id=user_id)
    token_usage  = result.get("token_usage", {})
    total_tokens = token_usage.get("total_tokens", 0)

    # ✅ Track tokens for non-streaming too
    if total_tokens > 0 and user_id:
        await track_token_usage(user_id, total_tokens)

    raw_refs      = result.get("references", [])
    enriched_refs = enrich_references_with_urls(raw_refs)
    references    = [
        DocumentReference(
            document_id=r.get("document_id", "unknown"),
            type=r.get("type", "DOCUMENT"),
            description=r.get("description"),
            url=r.get("url"),
            filename=r.get("filename"),
            file_type=r.get("file_type"),
        )
        for r in enriched_refs
    ]

    return ChatResponse(
        answer=result.get("answer", "Lỗi xử lý câu hỏi"),
        references=references,
        status=result.get("status", "ERROR"),
        token_usage={"total_tokens": total_tokens, "user_id": user_id} if total_tokens else None,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        db_ok       = milvus_client.check_connection()
        workflow_ok = rag_workflow is not None
        if db_ok and workflow_ok:
            return HealthResponse(status="healthy",  message="Hệ thống hoạt động bình thường", database_connected=True)
        elif workflow_ok:
            return HealthResponse(status="degraded", message="Mất kết nối DB",    database_connected=False)
        else:
            return HealthResponse(status="unhealthy", message="Hệ thống lỗi",    database_connected=False)
    except Exception as e:
        return HealthResponse(status="unhealthy", message=str(e), database_connected=False)


@app.get("/agents")
async def list_agents():
    return {
        "agents": {
            "SUPERVISOR":      "Điều phối",
            "FAQ":             "Câu hỏi thường gặp",
            "RETRIEVER":       "Tìm kiếm (ACL-filtered)",
            "GRADER":          "Đánh giá",
            "GENERATOR":       "Tạo câu trả lời (streaming + token tracking)",
            "NOT_ENOUGH_INFO": "Thiếu thông tin",
            "CHATTER":         "Cảm xúc",
            "REPORTER":        "Báo cáo",
            "OTHER":           "Ngoài phạm vi",
        },
        "token_tracking_endpoint": EMBEDDING_API_BASE,
        "status": "ready" if rag_workflow else "not_initialized",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501, log_level="info")
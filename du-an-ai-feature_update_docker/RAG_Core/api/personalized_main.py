# RAG_Core/api/personalized_main.py - WITH DOCUMENT URLs INTEGRATION

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, AsyncIterator
import logging
import json
import asyncio

from workflow.personalized_rag_workflow import PersonalizedRAGWorkflow
from database.milvus_client import milvus_client

# NEW: Import personalization document URL service
from services.personalization_document_url_service import personalization_document_url_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Personalized RAG Chatbot API with Document URLs",
    description="API cho hệ thống chatbot RAG với cá nhân hóa và document URLs từ personalization_db",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

personalized_workflow = None


# ================================================================
# SCHEMAS
# ================================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class PersonalizedChatRequest(BaseModel):
    """
    Request schema cho personalized chat

    Example:
    {
        "question": "giải pháp phù hợp với công ty truyền thông?",
        "history": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Xin chào..."}
        ],
        "stream": true,
        "name": "Nguyen Hoang Long",
        "introduction": "Tổng giám đốc công ty công nghệ và truyền thông VTC NetViet"
    }
    """
    question: str = Field(..., description="Câu hỏi của khách hàng")
    history: Optional[List[ChatMessage]] = Field(
        default=[],
        description="Lịch sử hội thoại"
    )
    stream: Optional[bool] = Field(
        default=True,
        description="Sử dụng streaming mode"
    )
    name: Optional[str] = Field(
        default="",
        description="Tên khách hàng"
    )
    introduction: Optional[str] = Field(
        default="",
        description="Giới thiệu về khách hàng (chức danh, công ty, lĩnh vực...)"
    )


class DocumentReference(BaseModel):
    """
    Document reference with optional URL information from personalization_db

    Fields:
    - document_id: Unique document identifier
    - type: Reference type (FAQ, DOCUMENT, SUPPORT, SYSTEM)
    - description: Document description/content preview
    - url: Public URL to document (NEW)
    - filename: Original filename (NEW)
    - file_type: File extension like .pdf, .docx (NEW)
    """
    document_id: str
    type: str
    description: Optional[str] = None

    # ===== NEW FIELDS FROM PERSONALIZATION_DB =====
    url: Optional[str] = None  # https://ngrok.../file.pdf
    filename: Optional[str] = None  # file.pdf
    file_type: Optional[str] = None  # .pdf


class PersonalizedChatResponse(BaseModel):
    answer: str
    references: List[DocumentReference]
    status: str = "SUCCESS"
    personalized: bool = False
    customer_name: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    message: str
    database_connected: bool
    personalization_enabled: bool
    url_service_enabled: bool


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def enrich_references_with_urls(references: List[dict]) -> List[dict]:
    """
    Helper function to enrich references with document URLs from personalization_db

    Args:
        references: List of reference dicts

    Returns:
        List of enriched references
    """
    try:
        return personalization_document_url_service.enrich_references_with_urls(references)
    except Exception as e:
        logger.error(f"Error enriching references: {e}")
        return references


# ================================================================
# STARTUP
# ================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize Personalized RAG Workflow"""
    global personalized_workflow
    try:
        personalized_workflow = PersonalizedRAGWorkflow()
        logger.info("✅ Personalized RAG Workflow initialized successfully")
        logger.info("✅ Personalization Document URL service initialized")
    except Exception as e:
        logger.error(f"⚠️ Failed to initialize workflow: {e}")


# ================================================================
# ENDPOINTS
# ================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    from config.settings import settings

    return {
        "service": "Personalized RAG Chatbot API",
        "version": "2.0.0",
        "port": 8502,
        "features": [
            "personalized-responses",
            "customer-profiling",
            "adaptive-tone",
            "streaming",
            "context-aware",
            "document-urls"  # NEW
        ],
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "info": "/info"
        },
        "url_config": {
            "ngrok_enabled": settings.NGROK_PUBLIC_URL is not None,
            "url_replacement_enabled": settings.ENABLE_URL_REPLACEMENT,
            "source_database": "personalization_db"
        }
    }


async def generate_personalized_streaming_response(
        question: str,
        history: List,
        customer_name: str,
        customer_introduction: str
) -> AsyncIterator[str]:
    """
    Async generator cho personalized streaming response with URL support

    Args:
        question: Câu hỏi
        history: Lịch sử hội thoại
        customer_name: Tên khách hàng
        customer_introduction: Giới thiệu khách hàng

    Yields:
        SSE-formatted chunks
    """
    try:
        logger.info(f"🚀 Starting personalized streaming")
        logger.info(f"   Customer: {customer_name}")
        logger.info(f"   Question: {question[:60]}...")

        # Send start chunk
        start_chunk = {
            "type": "start",
            "content": None,
            "references": None,
            "status": "processing",
            "personalized": bool(customer_name or customer_introduction),
            "customer_name": customer_name
        }
        yield f"data: {json.dumps(start_chunk, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.01)

        # Run personalized workflow
        result = await personalized_workflow.run_with_personalization_streaming(
            question=question,
            history=history,
            customer_name=customer_name,
            customer_introduction=customer_introduction
        )

        # Get answer stream
        answer_stream = result.get("answer_stream")
        references = result.get("references", [])
        personalized = result.get("personalized", False)

        logger.info(f"📝 Streaming answer (personalized={personalized})")

        # Stream chunks
        if answer_stream:
            chunk_count = 0
            async for chunk in answer_stream:
                if chunk:
                    chunk_count += 1
                    chunk_data = {
                        "type": "chunk",
                        "content": chunk,
                        "references": None,
                        "status": None
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.001)

            logger.info(f"✅ Streamed {chunk_count} chunks")
        else:
            logger.warning("⚠️ No answer stream available")
            error_chunk = {
                "type": "chunk",
                "content": "Không thể tạo câu trả lời.",
                "references": None,
                "status": None
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"

        # ===== NEW: ENRICH REFERENCES WITH URLs FROM PERSONALIZATION_DB =====
        if references:
            logger.info("🔗 Enriching references with document URLs from personalization_db...")
            enriched_refs = enrich_references_with_urls(references)

            serializable_refs = []
            for ref in enriched_refs:
                ref_dict = {
                    "document_id": ref.get("document_id", ""),
                    "type": ref.get("type", "DOCUMENT"),
                    "description": ref.get("description", "")
                }

                # Add URL fields if available
                if ref.get("url"):
                    ref_dict["url"] = ref["url"]
                    ref_dict["filename"] = ref.get("filename", "")
                    ref_dict["file_type"] = ref.get("file_type", "")
                    logger.info(f"  ✅ {ref['document_id']}: {ref.get('filename', 'N/A')}")

                serializable_refs.append(ref_dict)

            ref_chunk = {
                "type": "references",
                "content": None,
                "references": serializable_refs,
                "status": None
            }
            yield f"data: {json.dumps(ref_chunk, ensure_ascii=False)}\n\n"
            logger.info(f"📚 Sent {len(serializable_refs)} enriched references")

        # Send end chunk
        end_chunk = {
            "type": "end",
            "content": None,
            "references": None,
            "status": result.get("status", "SUCCESS"),
            "personalized": personalized,
            "customer_name": customer_name
        }
        yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
        logger.info("✅ Streaming completed")

    except Exception as e:
        logger.error(f"❌ Streaming error: {e}", exc_info=True)
        error_chunk = {
            "type": "error",
            "content": f"Lỗi: {str(e)}",
            "references": None,
            "status": "ERROR"
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"


@app.post("/chat")
async def personalized_chat(request: PersonalizedChatRequest):
    """
    Main personalized chat endpoint with document URLs support

    Hỗ trợ cả streaming và non-streaming mode
    Cá nhân hóa dựa trên name và introduction
    URLs từ personalization_document_urls collection

    Example Request (Streaming):
    ```json
    {
        "question": "giải pháp phù hợp với công ty truyền thông?",
        "history": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Xin chào..."}
        ],
        "stream": true,
        "name": "Nguyen Hoang Long",
        "introduction": "Tổng giám đốc công ty công nghệ và truyền thông VTC NetViet"
    }
    ```
    """
    try:
        if not personalized_workflow:
            raise HTTPException(
                status_code=503,
                detail="Workflow not initialized"
            )

        logger.info(f"📨 Personalized chat request:")
        logger.info(f"   Question: {request.question[:100]}...")
        logger.info(f"   Customer: {request.name}")
        logger.info(f"   Stream: {request.stream}")

        # Convert history
        history = []
        for msg in request.history:
            history.append({
                "role": msg.role,
                "content": msg.content
            })

        # STREAMING MODE
        if request.stream:
            logger.info("🔄 Using STREAMING mode with personalization and URLs")
            return StreamingResponse(
                generate_personalized_streaming_response(
                    question=request.question,
                    history=history,
                    customer_name=request.name or "",
                    customer_introduction=request.introduction or ""
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )

        # NON-STREAMING MODE
        logger.info("📋 Using NON-STREAMING mode with personalization and URLs")
        result = personalized_workflow.run_with_personalization(
            question=request.question,
            history=history,
            customer_name=request.name or "",
            customer_introduction=request.introduction or ""
        )

        # ===== NEW: ENRICH REFERENCES WITH URLs =====
        raw_references = result.get("references", [])
        enriched_references = enrich_references_with_urls(raw_references)

        # Prepare references
        references = []
        for ref in enriched_references:
            ref_obj = DocumentReference(
                document_id=ref.get("document_id", "unknown"),
                type=ref.get("type", "DOCUMENT"),
                description=ref.get("description", None),
                url=ref.get("url", None),
                filename=ref.get("filename", None),
                file_type=ref.get("file_type", None)
            )
            references.append(ref_obj)

        logger.info(f"✅ Personalized response ready with {len(references)} references")
        logger.info(f"   Personalized: {result.get('personalized', False)}")

        return PersonalizedChatResponse(
            answer=result.get("answer", "Lỗi xử lý câu hỏi"),
            references=references,
            status=result.get("status", "ERROR"),
            personalized=result.get("personalized", False),
            customer_name=request.name or None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with URL service status"""
    try:
        db_connected = False
        try:
            db_connected = milvus_client.check_connection()
        except Exception as db_error:
            logger.warning(f"Database check failed: {db_error}")

        workflow_ready = personalized_workflow is not None
        url_service_ready = personalization_document_url_service.collection is not None

        if db_connected and workflow_ready:
            message = "Hệ thống hoạt động bình thường với personalization"
            if url_service_ready:
                message += " và document URLs"

            return HealthResponse(
                status="healthy",
                message=message,
                database_connected=True,
                personalization_enabled=True,
                url_service_enabled=url_service_ready
            )
        elif workflow_ready and not db_connected:
            return HealthResponse(
                status="degraded",
                message="Mất kết nối cơ sở dữ liệu",
                database_connected=False,
                personalization_enabled=True,
                url_service_enabled=False
            )
        else:
            return HealthResponse(
                status="unhealthy",
                message="Hệ thống gặp sự cố",
                database_connected=False,
                personalization_enabled=False,
                url_service_enabled=False
            )

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            message=f"Lỗi: {str(e)}",
            database_connected=False,
            personalization_enabled=False,
            url_service_enabled=False
        )


@app.get("/info")
async def system_info():
    """Thông tin hệ thống với URL configuration"""
    from config.settings import settings

    return {
        "service": "Personalized RAG API",
        "version": "2.0.0",
        "port": 8502,
        "features": {
            "personalization": {
                "enabled": True,
                "description": "Cá nhân hóa câu trả lời theo thông tin khách hàng",
                "supported_fields": ["name", "introduction"]
            },
            "customer_profiling": {
                "enabled": True,
                "auto_detect": ["title", "seniority", "industry", "tone"]
            },
            "streaming": {
                "enabled": True,
                "format": "SSE (Server-Sent Events)"
            },
            "base_rag": {
                "enabled": True,
                "agents": ["FAQ", "RETRIEVER", "GRADER", "GENERATOR"]
            },
            "document_urls": {
                "enabled": True,
                "source_collection": "personalization_document_urls",
                "source_database": "personalization_db",
                "ngrok_enabled": settings.NGROK_PUBLIC_URL is not None
            }
        },
        "workflow_ready": personalized_workflow is not None,
        "url_service_ready": personalization_document_url_service.collection is not None
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8502,
        log_level="info"
    )
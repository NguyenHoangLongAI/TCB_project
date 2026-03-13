# RAG_Core/api/schemas.py  (UPDATED – user_id + token tracking)

from pydantic import BaseModel
from typing import List, Optional, Union, Literal


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history:  Optional[Union[List[str], List[ChatMessage]]] = []
    stream:   Optional[bool] = False
    user_id:  Optional[str]  = None    # NEW – required for ACL + token tracking


class StreamChunk(BaseModel):
    type:       Literal["start", "chunk", "references", "end", "error"]
    content:    Optional[str] = None
    references: Optional[List["DocumentReference"]] = None
    status:     Optional[str] = None
    token_usage: Optional[dict] = None   # NEW


class DocumentReference(BaseModel):
    document_id: str
    type:        str
    description: Optional[str] = None
    url:         Optional[str] = None
    filename:    Optional[str] = None
    file_type:   Optional[str] = None


class ChatResponse(BaseModel):
    answer:      str
    references:  List[DocumentReference]
    status:      str = "SUCCESS"
    token_usage: Optional[dict] = None   # NEW – total tokens used in this call


class HealthResponse(BaseModel):
    status:             str
    message:            str
    database_connected: bool
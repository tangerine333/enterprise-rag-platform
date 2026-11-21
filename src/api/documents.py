# ============================================================
# 文档管理 API
# ============================================================
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from src.core.database import store
from src.document_pipeline import doc_parser
from src.chunking import chunker

router = APIRouter()

ALLOWED_TYPES = {"pdf", "docx", "pptx", "xlsx", "md", "txt", "png", "jpg", "jpeg"}
MAX_SIZE_MB = 50


# --- 响应模型 ---
class DocResponse(BaseModel):
    id: int
    title: str
    file_type: str
    parse_status: str
    chunk_count: int
    parse_error: str | None = None
    created_at: str
    model_config = {"from_attributes": True}


class DocListResponse(BaseModel):
    items: list[DocResponse]
    total: int


# ------------------------------------------------------------------
# POST /upload
# ------------------------------------------------------------------
@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
):
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_TYPES:
        raise HTTPException(400, f"不支持的文件类型: .{ext}")

    content = await file.read()
    if len(content) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"超过 {MAX_SIZE_MB}MB 限制")

    # 存到临时文件
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # 创建文档记录
    doc_id = store.insert_document(
        title=file.filename or "未命名",
        file_type=ext,
        file_path=tmp_path,
        file_size_bytes=len(content),
        parse_status="parsing",
    )

    # 同步解析（开发环境，生产应走 Celery）
    try:
        parsed = await doc_parser.parse(tmp_path, ext)
        chunks = chunker.chunk_document(
            parsed.content_markdown,
            title=parsed.title,
            tables=parsed.tables,
        )

        # 写入 chunks 到 SQLite
        chunk_records = []
        for c in chunks["children"]:
            chunk_records.append({
                "doc_id": doc_id,
                "chunk_type": c.chunk_type,
                "chunk_index": c.chunk_index,
                "parent_chunk_id": c.parent_index,
                "content": c.content,
                "heading": c.heading,
                "metadata_json": str(c.metadata),
            })
        store.insert_chunks(chunk_records)

        # 写入向量库和关键词索引
        # embedding + qdrant_retriever.index_chunks + es_retriever.index_chunks

        store.update_document(doc_id, parse_status="completed",
                              chunk_count=len(chunks["children"]))
    except Exception as e:
        store.update_document(doc_id, parse_status="failed", parse_error=str(e))

    doc = store.get_document(doc_id)
    return {
        "id": doc_id,
        "title": doc["title"],
        "file_type": doc["file_type"],
        "parse_status": doc["parse_status"],
        "chunk_count": doc["chunk_count"],
        "message": "文档解析完成" if doc["parse_status"] == "completed" else f"解析异常: {doc.get('parse_error', '')}",
    }


# ------------------------------------------------------------------
# GET / — 文档列表
# ------------------------------------------------------------------
@router.get("/")
async def list_documents(
    status: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    docs = store.list_documents(status=status, limit=page_size, offset=(page - 1) * page_size)
    total = store.count_documents()
    return DocListResponse(
        items=[DocResponse(**d) for d in docs],
        total=total,
    )


# ------------------------------------------------------------------
# GET /{doc_id}
# ------------------------------------------------------------------
@router.get("/{doc_id}")
async def get_document(doc_id: int):
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")
    return DocResponse(**doc)


# ------------------------------------------------------------------
# DELETE /{doc_id}
# ------------------------------------------------------------------
@router.delete("/{doc_id}")
async def delete_document(doc_id: int):
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")
    store.delete_document(doc_id)
    return {"message": f"已删除: {doc['title']}", "doc_id": doc_id}


# ------------------------------------------------------------------
# GET /{doc_id}/status
# ------------------------------------------------------------------
@router.get("/{doc_id}/status")
async def get_status(doc_id: int):
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")
    return {"doc_id": doc_id, "title": doc["title"],
            "parse_status": doc["parse_status"], "chunk_count": doc["chunk_count"]}

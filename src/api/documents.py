# ============================================================
# 文档管理 API — 上传 → 解析 → 双索引分块 → embedding → Qdrant + ES
# 含增量更新（checksum 去重）、权限组、表格 LLM 语义摘要
# 上传走 BackgroundTasks：解析+索引在后台执行，接口立即返回
# ============================================================
from __future__ import annotations

import json
import tempfile

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from src.core.config import get_settings
from src.core.database import compute_checksum, store
from src.core.metrics import DOCUMENT_TOTAL, INDEX_SYNC_PENDING
from src.document_pipeline import doc_parser
from src.chunking import chunker
from src.retrieval import es_retriever, qdrant_retriever, retrieval_cache

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
# POST /upload — 上传并提交后台解析
# ------------------------------------------------------------------
@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    version: str = Form(""),
    department: str = Form(""),
    permission_groups: str = Form("*"),
):
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_TYPES:
        raise HTTPException(400, f"不支持的文件类型: .{ext}")

    content = await file.read()
    if len(content) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"超过 {MAX_SIZE_MB}MB 限制")

    # --- 增量更新：checksum 去重 ---
    checksum = compute_checksum(content)
    existing = store.find_by_checksum(checksum)
    if existing:
        return {
            "id": existing["id"],
            "title": existing["title"],
            "file_type": existing["file_type"],
            "parse_status": existing["parse_status"],
            "chunk_count": existing["chunk_count"],
            "message": "文档已存在（增量更新跳过重建索引）",
        }

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    doc_id = store.insert_document(
        title=file.filename or "未命名",
        file_type=ext,
        file_path=tmp_path,
        file_size_bytes=len(content),
        version=version or None,
        department=department or None,
        permission_groups=permission_groups or "*",
        checksum=checksum,
        parse_status="parsing",
    )

    # 后台解析 + 索引（接口立即返回，前端/脚本轮询 status）
    background_tasks.add_task(
        _process_document, doc_id, tmp_path, ext, permission_groups or "*"
    )

    return {
        "id": doc_id,
        "title": file.filename or "未命名",
        "file_type": ext,
        "parse_status": "parsing",
        "chunk_count": 0,
        "message": "文档已提交，后台解析中（可轮询 /documents/{id}/status）",
    }


async def _process_document(doc_id: int, tmp_path: str, ext: str, permission_groups: str):
    """后台任务：解析 → 分块 → 表格摘要 → 双路索引。"""
    try:
        parsed = await doc_parser.parse(tmp_path, ext)
        result = chunker.chunk_document(
            parsed.content_markdown, title=parsed.title, tables=parsed.tables,
        )
        result["table_summaries"] = await _enrich_table_summaries(
            parsed.tables, result["table_summaries"]
        )
        await _index_document(doc_id, result, permission_groups)

        total_chunks = (
            len(result["children"]) + len(result["parents"])
            + len(result["table_rows"]) + len(result["table_summaries"])
        )
        store.update_document(doc_id, parse_status="completed", chunk_count=total_chunks)
        DOCUMENT_TOTAL.labels(status="completed").inc()
        # 文档内容已变更：使旧检索/答案缓存整体失效
        await retrieval_cache.bump_version()
    except Exception as e:
        store.update_document(doc_id, parse_status="failed", parse_error=str(e))
        DOCUMENT_TOTAL.labels(status="failed").inc()


async def _index_document(doc_id: int, result: dict, permission_groups: str):
    """把分块结果写入 SQLite，并对可检索单元建 Qdrant + ES 双路索引。"""

    def _rec(c, ctype, parent_index=None):
        return {
            "doc_id": doc_id,
            "chunk_type": ctype,
            "chunk_index": c.chunk_index,
            "parent_chunk_id": parent_index,
            "content": c.content,
            "heading": c.heading,
            "page_range": c.page_range,
            "metadata_json": json.dumps(c.metadata, ensure_ascii=False, default=str),
        }

    child_recs = [_rec(c, "child", parent_index=c.parent_index) for c in result["children"]]
    parent_recs = [_rec(p, "parent") for p in result["parents"]]
    row_recs = [_rec(r, "table_row") for r in result["table_rows"]]
    summary_recs = [_rec(s, "table_summary") for s in result["table_summaries"]]

    child_ids = store.insert_chunks(child_recs)
    parent_ids = store.insert_chunks(parent_recs)
    row_ids = store.insert_chunks(row_recs)
    summary_ids = store.insert_chunks(summary_recs)

    index_payloads = []
    for rec, cid in zip(child_recs, child_ids):
        index_payloads.append({**rec, "chunk_id": cid, "table_summary": "", "permission_groups": permission_groups})
    for rec, cid in zip(row_recs, row_ids):
        index_payloads.append({**rec, "chunk_id": cid, "table_summary": "", "permission_groups": permission_groups})
    for rec, cid in zip(summary_recs, summary_ids):
        index_payloads.append({**rec, "chunk_id": cid, "table_summary": rec.get("content", "")[:200], "permission_groups": permission_groups})

    if not index_payloads:
        return

    texts = [p["content"] for p in index_payloads]
    embeddings = await qdrant_retriever.encode(texts)
    await qdrant_retriever.index_chunks(index_payloads, embeddings)
    await es_retriever.index_chunks(index_payloads)


async def _enrich_table_summaries(tables: list[dict], summaries: list) -> list:
    """用 LLM 为表格生成语义摘要（表级索引用），失败则回退规则模板。"""
    valid_tables = [t for t in tables if (t.get("headers") or t.get("rows"))]
    if not valid_tables or not summaries:
        return summaries

    settings = get_settings()
    if not settings.openai_api_key or settings.openai_api_key == "your-api-key":
        return summaries

    try:
        from src.core.llm import create_llm
        llm = create_llm(model=settings.intent_model, temperature=0.0, max_tokens=512)
        if llm is None:
            return summaries
        table_descs = []
        for t in valid_tables:
            headers = [str(h) for h in (t.get("headers") or [])][:6]
            rows_preview = t.get("rows") or []
            caption = t.get("caption", "表格")
            table_descs.append(
                f"{caption} | 字段:{'、'.join(headers)} | 行数:{len(rows_preview)} | "
                f"示例行:{'; '.join(str(c) for c in rows_preview[0][:6]) if rows_preview else ''}"
            )

        prompt = (
            "为以下每个表格生成一句不超过50字的语义摘要（概括表格主题与关键维度），"
            "输出 JSON 字符串数组，元素顺序与输入一致：\n"
            + "\n".join(f"{i+1}. {d}" for i, d in enumerate(table_descs))
            + "\n\nJSON 数组:"
        )
        resp = await llm.ainvoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        import re
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            arr = json.loads(m.group())
            if isinstance(arr, list):
                for i, s in enumerate(summaries):
                    if i < len(arr):
                        new_summary = str(arr[i])[:200]
                        s.table_summary = new_summary
                        s.content = f"[表格摘要] {s.metadata.get('table_caption', '表格')}\n{new_summary}"
    except Exception:
        pass  # 任何失败都回退规则摘要

    return summaries


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
    return DocListResponse(items=[DocResponse(**d) for d in docs], total=total)


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
# 索引清理：Qdrant/ES 删除失败 → 补偿任务
# ------------------------------------------------------------------
async def _cleanup_index(doc_id: int) -> list[str]:
    """删除 Qdrant/ES 中的文档切片。失败的目标写入补偿任务并返回。"""
    failed: list[str] = []
    for target, retriever in (("qdrant", qdrant_retriever), ("es", es_retriever)):
        try:
            await retriever.delete_by_doc(doc_id)
        except Exception as exc:
            store.add_sync_task(doc_id, target, "delete", str(exc))
            failed.append(target)
    INDEX_SYNC_PENDING.set(store.count_pending_sync_tasks())
    return failed


async def reconcile_index_sync(limit: int = 100) -> dict:
    """重试待补偿的索引同步任务（文档删除后 Qdrant/ES 清理失败）。"""
    tasks = store.list_pending_sync_tasks(limit=limit)
    done = failed = 0
    errors: list[dict] = []
    for task in tasks:
        target = task["target"]
        try:
            if target == "qdrant":
                await qdrant_retriever.delete_by_doc(task["doc_id"])
            elif target == "es":
                await es_retriever.delete_by_doc(task["doc_id"])
            else:
                raise RuntimeError(f"未知同步目标: {target}")
            store.mark_sync_task_done(task["id"])
            done += 1
        except Exception as exc:
            store.mark_sync_task_failed(task["id"], str(exc))
            failed += 1
            if len(errors) < 5:
                errors.append({"task_id": task["id"], "target": target, "error": str(exc)})
    INDEX_SYNC_PENDING.set(store.count_pending_sync_tasks())
    return {"processed": len(tasks), "done": done, "failed": failed, "errors": errors}


# ------------------------------------------------------------------
# DELETE /{doc_id}
# ------------------------------------------------------------------
@router.delete("/{doc_id}")
async def delete_document(doc_id: int):
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")
    # 先删元数据（对外立即不可见），再传播到索引层；失败落补偿任务
    store.delete_document(doc_id)
    failed = await _cleanup_index(doc_id)
    await retrieval_cache.bump_version()
    return {
        "message": f"已删除: {doc['title']}",
        "doc_id": doc_id,
        "index_cleanup": "completed" if not failed else "pending",
        "pending_targets": failed,
    }


# ------------------------------------------------------------------
# PUT /{doc_id} — 更新文档：清理旧索引 → 重新解析建索引
# ------------------------------------------------------------------
@router.put("/{doc_id}")
async def replace_document(
    doc_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    version: str = Form(""),
    department: str = Form(""),
    permission_groups: str = Form("*"),
):
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")

    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_TYPES:
        raise HTTPException(400, f"不支持的文件类型: .{ext}")

    content = await file.read()
    if len(content) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"超过 {MAX_SIZE_MB}MB 限制")

    checksum = compute_checksum(content)
    if doc.get("checksum") == checksum:
        return {"id": doc_id, "message": "内容未变化，跳过重建索引",
                "parse_status": doc["parse_status"]}

    # 旧索引清理失败则中止，避免新旧切片混杂、检索命中过期内容
    failed = await _cleanup_index(doc_id)
    if failed:
        raise HTTPException(
            503, f"旧索引清理未完成，请稍后重试或调用 /system/reconcile: {failed}"
        )

    store.delete_chunks_by_doc(doc_id)
    await retrieval_cache.bump_version()

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    store.update_document(
        doc_id,
        title=file.filename or doc["title"],
        file_type=ext,
        file_path=tmp_path,
        file_size_bytes=len(content),
        version=version or None,
        department=department or None,
        permission_groups=permission_groups or "*",
        checksum=checksum,
        parse_status="parsing",
        parse_error=None,
        chunk_count=0,
    )
    background_tasks.add_task(
        _process_document, doc_id, tmp_path, ext, permission_groups or "*"
    )
    return {"id": doc_id, "message": "文档已更新，后台重建索引中", "parse_status": "parsing"}


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

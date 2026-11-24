"""分块：Small-to-Big 父子分块 + 表格行级/表级双索引。"""
from src.chunking import chunker


def test_small_to_big_chunking():
    content = "# 标题\n\n" + "段落内容。" * 200
    result = chunker.chunk_document(content, title="doc.md", tables=[])

    assert "children" in result and "parents" in result
    assert len(result["children"]) > 0
    assert len(result["parents"]) > 0
    # 每个 child 都有 parent_index 指向父块
    for c in result["children"]:
        assert c.parent_index is not None
    assert result["children"][0].chunk_type == "child"
    assert result["parents"][0].chunk_type == "parent"


def test_table_dual_index():
    content = "# 政策\n\n文本内容。\n"
    tables = [{
        "caption": "退款条件",
        "headers": ["条件", "说明", "时效"],
        "rows": [["未使用", "全额退款", "3天"], ["已使用", "按比例退款", "15天"]],
        "page": 1,
    }]
    result = chunker.chunk_document(content, title="policy.md", tables=tables)

    # 行级索引：每行一个独立检索单元
    assert len(result["table_rows"]) == 2
    # 表级索引：一个语义摘要单元
    assert len(result["table_summaries"]) == 1

    # 行级单元携带表头上下文
    row = result["table_rows"][0]
    assert row.chunk_type == "table_row"
    assert "表头" in row.content
    assert "未使用" in row.content

    # 表级单元含摘要
    summary = result["table_summaries"][0]
    assert summary.chunk_type == "table_summary"
    assert summary.table_summary


def test_empty_input():
    result = chunker.chunk_document("", title="empty.md", tables=[])
    assert result["children"] == []
    assert result["parents"] == []

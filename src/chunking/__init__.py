# ============================================================
# Small-to-Big 父子分块 + 表格行级/表级双索引
# 子块 256 token → 精确向量检索
# 父块 1024 token → 生成阶段回溯完整上下文
# 表格 → 行级索引（每行独立检索单元，携带表头上下文）
#         + 表级索引（语义摘要独立参与向量检索）
# ============================================================
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ChunkResult:
    content: str
    chunk_type: str        # "child" | "parent" | "table_row" | "table_summary"
    chunk_index: int
    parent_index: int | None = None
    heading: str | None = None
    page_range: str | None = None
    token_count: int = 0
    metadata: dict = field(default_factory=dict)
    table_summary: str | None = None


class SmallToBigChunker:
    """Small-to-Big 父子分块 + 表格双索引。"""

    def __init__(self, child_size: int = 256, parent_size: int = 1024, overlap: int = 64):
        self.child_size = child_size
        self.parent_size = parent_size
        self.overlap = overlap

    def chunk_document(
        self,
        content: str,
        title: str = "",
        tables: list[dict] | None = None,
    ) -> dict:
        """主入口。返回 {'children', 'parents', 'table_rows', 'table_summaries'}。"""
        sections = self._split_by_headings(content)

        all_children = []
        for sec in sections:
            children = self._chunk_text(
                sec["content"], chunk_size=self.child_size,
                overlap=self.overlap, heading=sec.get("heading", title),
            )
            for c in children:
                c.heading = sec.get("heading", title)
            all_children.extend(children)

        for i, c in enumerate(all_children):
            c.chunk_index = i

        # 生成父块（每 4 个子块合并为 1 个父块，供生成阶段回溯）
        parents = []
        group_size = 4
        for i in range(0, len(all_children), group_size):
            group = all_children[i : i + group_size]
            parent = ChunkResult(
                content="\n\n".join(c.content for c in group),
                chunk_type="parent",
                chunk_index=len(parents),
                heading=group[0].heading,
                token_count=sum(c.token_count for c in group),
            )
            parents.append(parent)
            for c in group:
                c.parent_index = parent.chunk_index

        # 表格行级/表级双索引
        table_rows, table_summaries = self._build_table_index(tables or [], title)

        return {
            "children": all_children,
            "parents": parents,
            "table_rows": table_rows,
            "table_summaries": table_summaries,
        }

    # ------------------------------------------------------------------
    # 表格双索引
    # ------------------------------------------------------------------
    def _build_table_index(self, tables: list[dict], doc_title: str):
        """行级索引 + 表级索引。

        行级：每行 = 独立检索单元（携带表头上下文），精确定位"某区域 Q3 数据"。
        表级：语义摘要 = 独立检索单元，回答"季度销售趋势"这类整体理解问题。
        """
        table_rows: list[ChunkResult] = []
        table_summaries: list[ChunkResult] = []

        for tbl in tables:
            headers = [str(h) for h in (tbl.get("headers") or [])]
            rows = tbl.get("rows") or []
            caption = tbl.get("caption", "表格")
            page = tbl.get("page")

            if not headers and not rows:
                continue

            # 表级索引：语义摘要（LLM 生成，规则模板兜底）
            summary = self._generate_table_summary(tbl)
            table_summaries.append(ChunkResult(
                content=f"[表格摘要] {caption}\n{summary}",
                chunk_type="table_summary",
                chunk_index=len(table_summaries),
                heading=doc_title,
                page_range=str(page) if page else None,
                token_count=self._count_tokens(summary),
                metadata={"has_table": True, "table_caption": caption,
                          "headers": headers, "is_table_summary": True},
                table_summary=summary,
            ))

            # 行级索引：每行一个独立检索单元，携带表头上下文
            for ri, row in enumerate(rows):
                cells = [str(c) for c in row]
                parts = []
                for j, val in enumerate(cells):
                    if j < len(headers) and val:
                        parts.append(f"{headers[j]}: {val}")
                    elif val:
                        parts.append(f"列{j+1}: {val}")
                if not parts:
                    continue
                row_text = "；".join(parts)
                row_content = (
                    f"[表格] {caption}（表头: {'、'.join(headers[:8])}）\n"
                    f"{row_text}"
                )
                table_rows.append(ChunkResult(
                    content=row_content,
                    chunk_type="table_row",
                    chunk_index=len(table_rows),
                    heading=doc_title,
                    page_range=str(page) if page else None,
                    token_count=self._count_tokens(row_content),
                    metadata={"has_table": True, "table_caption": caption,
                              "row_index": ri, "headers": headers[:8]},
                ))

        return table_rows, table_summaries

    @staticmethod
    def _generate_table_summary(tbl: dict) -> str:
        """表格语义摘要（生产用 LLM，此处为规则模板兜底，保证零依赖可跑）。

        示例: "该表记录了 2024 年 Q1-Q4 各区域销售额及同比增长率，共 12 行 4 列"
        """
        headers = [str(h) for h in (tbl.get("headers") or [])]
        rows = tbl.get("rows") or []
        caption = tbl.get("caption", "表格")

        if not headers:
            return f"{caption}，共 {len(rows)} 行"
        col_desc = "、".join(str(h)[:10] for h in headers[:5])
        return f"{caption}，包含字段：{col_desc}，共 {len(rows)} 行 {len(headers)} 列"

    # ------------------------------------------------------------------
    # 内部分块逻辑
    # ------------------------------------------------------------------
    def _split_by_headings(self, text: str) -> list[dict]:
        sections = []
        pattern = r"^(#{1,6})\s+(.+)$"
        current_heading = ""
        current_content: list[str] = []

        for line in text.split("\n"):
            m = re.match(pattern, line.strip())
            if m:
                if current_content:
                    sections.append({"heading": current_heading or "", "content": "\n".join(current_content)})
                current_heading = m.group(2)
                current_content = [line]
            else:
                current_content.append(line)

        if current_content:
            sections.append({"heading": current_heading or "", "content": "\n".join(current_content)})

        return sections if sections else [{"heading": "", "content": text}]

    def _chunk_text(self, text: str, chunk_size: int, overlap: int,
                    heading: str | None = None) -> list[ChunkResult]:
        separators = ["\n\n", "\n", "。", "；", ". ", "; ", " ", ""]
        raw_chunks = self._recursive_split(text, separators, chunk_size)
        return [
            ChunkResult(
                content=c.strip(),
                chunk_type="child",
                chunk_index=i,
                token_count=self._count_tokens(c.strip()),
            )
            for i, c in enumerate(raw_chunks) if c.strip()
        ]

    def _recursive_split(self, text: str, separators: list[str],
                         chunk_size: int) -> list[str]:
        """递归按分隔符切分为字符串列表。"""
        if not text.strip():
            return []

        sep = separators[0] if separators else ""
        parts = text.split(sep) if sep else [text]

        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part
            if self._count_tokens(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if self._count_tokens(part) > chunk_size and len(separators) > 1:
                    sub = self._recursive_split(part, separators[1:], chunk_size)
                    chunks.extend(sub)
                else:
                    current = part

        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _count_tokens(text: str) -> int:
        """粗略 token 估算（中文~2字符/token，英文~4字符/token）。"""
        chinese = len(re.findall(r'[一-鿿]', text))
        other = len(text) - chinese
        return chinese // 2 + other // 4


# 单例
chunker = SmallToBigChunker()

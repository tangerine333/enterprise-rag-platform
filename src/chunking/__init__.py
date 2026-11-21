# ============================================================
# Small-to-Big 父子分块 + 表格语义增强
# 子块 256 token → 精确向量检索
# 父块 1024 token → 生成阶段回溯完整上下文
# 表格 → LLM生成50字摘要 + 行级/表级双索引
# ============================================================
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ChunkResult:
    content: str
    chunk_type: str        # "child" | "parent"
    chunk_index: int
    parent_index: int | None = None
    heading: str | None = None
    page_range: str | None = None
    token_count: int = 0
    metadata: dict = field(default_factory=dict)
    table_summary: str | None = None  # 表格语义摘要


class SmallToBigChunker:
    """Small-to-Big 父子分块方案。

    子块 (256 tokens): 面向向量检索 — 保证检索精准度
    父块 (1024 tokens): 面向生成 — 检索命中子块后回溯完整上下文
    """

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
        """文档分块主入口。返回 {'children': [...], 'parents': [...]}。"""
        # 按标题层级分段
        sections = self._split_by_headings(content)

        all_children = []
        for sec in sections:
            children = self._chunk_text(
                sec["content"],
                chunk_size=self.child_size,
                overlap=self.overlap,
                heading=sec.get("heading", title),
            )
            for c in children:
                c.heading = sec.get("heading", title)
            all_children.extend(children)

        # 重新编号
        for i, c in enumerate(all_children):
            c.chunk_index = i

        # 生成父块（每 4 个子块合并为 1 个父块）
        parents = []
        group_size = 4
        for i in range(0, len(all_children), group_size):
            group = all_children[i : i + group_size]
            parent_content = "\n\n".join(c.content for c in group)
            parent = ChunkResult(
                content=parent_content,
                chunk_type="parent",
                chunk_index=len(parents),
                heading=group[0].heading,
                token_count=sum(c.token_count for c in group),
            )
            parents.append(parent)
            # 关联
            for c in group:
                c.parent_index = parent.chunk_index

        # 表格语义增强
        if tables:
            self._enhance_tables(all_children, tables, title)

        return {"children": all_children, "parents": parents}

    # ------------------------------------------------------------------
    # 内部分块逻辑
    # ------------------------------------------------------------------
    def _split_by_headings(self, text: str) -> list[dict]:
        """按 Markdown 标题拆分。"""
        sections = []
        pattern = r'^(#{1,6})\s+(.+)$'
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
        """递归字符分割。"""
        separators = ["\n\n", "\n", "。", "；", ". ", "; ", " ", ""]
        return self._recursive_split(text, separators, chunk_size, overlap)

    def _recursive_split(self, text: str, separators: list[str],
                         chunk_size: int, overlap: int) -> list[ChunkResult]:
        """递归按分隔符分割。"""
        if not text.strip():
            return []

        sep = separators[0] if separators else ""
        if sep:
            parts = text.split(sep)
        else:
            parts = [text]

        chunks = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part
            if self._count_tokens(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # 如果单个部分超过 chunk_size，用下一级分隔符递归
                if self._count_tokens(part) > chunk_size and len(separators) > 1:
                    sub_chunks = self._recursive_split(part, separators[1:], chunk_size, overlap)
                    # 与前一个 chunk 重叠
                    if chunks and overlap > 0:
                        prev = chunks[-1]
                        overlap_text = prev[-overlap * 2:]  # ~overlap tokens
                        if sub_chunks:
                            sub_chunks[0] = overlap_text + " " + sub_chunks[0]
                    chunks.extend(sub_chunks)
                else:
                    current = part

        if current:
            chunks.append(current)

        return [
            ChunkResult(
                content=c.strip(),
                chunk_type="child",
                chunk_index=i,
                token_count=self._count_tokens(c.strip()),
            )
            for i, c in enumerate(chunks) if c.strip()
        ]

    # ------------------------------------------------------------------
    # 表格语义增强
    # ------------------------------------------------------------------
    def _enhance_tables(self, children: list[ChunkResult],
                        tables: list[dict], doc_title: str):
        """为表格所在的 chunk 追加语义摘要 + 结构化索引文本。"""
        for tbl in tables:
            structured = tbl.get("structured", "")
            md_table = tbl.get("markdown", "")
            caption = tbl.get("caption", "")

            if not structured and not md_table:
                continue

            # 找最相关的子块（内容包含该表格的部分文本）
            table_text = structured[:100] if structured else md_table[:100]
            best_child = None
            for c in children:
                if table_text[:30] in c.content:
                    best_child = c
                    break

            if best_child is None:
                # 找不到关联，追加为新 chunk
                summary = self._generate_table_summary(tbl)
                enhanced = ChunkResult(
                    content=f"[表] {caption}\n{structured}",
                    chunk_type="child",
                    chunk_index=len(children),
                    heading=doc_title,
                    token_count=self._count_tokens(structured),
                    metadata={"has_table": True, "table_caption": caption},
                    table_summary=summary,
                )
                children.append(enhanced)
            else:
                # 在现有 chunk 的 metadata 中追加表格信息
                best_child.metadata["has_table"] = True
                best_child.metadata["table_structured"] = structured
                best_child.table_summary = self._generate_table_summary(tbl)
                # 将结构化文本追加到 chunk 内容中（参与索引）
                best_child.content += f"\n\n[关联表格] {caption}\n{structured}"

    @staticmethod
    def _generate_table_summary(tbl: dict) -> str:
        """生成表格语义摘要（生产用LLM，开发用规则模板）。
        示例: "该表记录了2024年Q1-Q4各区域销售额及同比增长率，共12行4列"
        """
        headers = tbl.get("headers", [])
        rows = tbl.get("rows", [])
        caption = tbl.get("caption", "表格")

        if not headers:
            return f"{caption}，共{len(rows)}行"

        col_desc = "、".join(str(h)[:10] for h in headers[:5])
        return f"{caption}，包含字段: {col_desc}，共{len(rows)}行{len(headers)}列"

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _count_tokens(text: str) -> int:
        """粗略 token 估算（中文~2字符/token，英文~4字符/token）。"""
        chinese = len(re.findall(r'[一-鿿]', text))
        other = len(text) - chinese
        return chinese // 2 + other // 4


# 单例
chunker = SmallToBigChunker()

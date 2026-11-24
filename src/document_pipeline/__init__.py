# ============================================================
# 文档解析流水线 — 实用工具，不为"自研"造轮子
# 路由：文本 PDF→PyMuPDF / 扫描页→PaddleOCR / 复杂版面→MinerU(CLI)
#       Word→python-docx / PPT→python-pptx / Excel→openpyxl / Markdown
# 表格→结构化 + Markdown（下游做行级/表级双索引）
# ============================================================
from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# MinerU 通过子进程调用，避免重型模型依赖污染主环境
MINERU_CMD = os.getenv("MINERU_CMD", "mineru")
MINERU_TIMEOUT = int(os.getenv("MINERU_TIMEOUT", "300"))
MINERU_SCAN_PAGES = int(os.getenv("MINERU_SCAN_PAGES", "20"))


@dataclass
class ParsedDocument:
    """统一中间格式。"""
    title: str
    file_type: str
    content_markdown: str = ""
    pages: list[dict] = field(default_factory=list)    # [{page_num, text, tables, images}]
    tables: list[dict] = field(default_factory=list)    # [{page, markdown, summary, headers, rows}]
    images: list[dict] = field(default_factory=list)    # [{page, description, ocr_text}]
    metadata: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class DocumentParser:
    """文档解析路由器。"""

    async def parse(self, file_path: str, file_type: str) -> ParsedDocument:
        ext = file_type.lower().lstrip(".")
        parser_map = {
            "pdf": self._parse_pdf,
            "docx": self._parse_docx,
            "pptx": self._parse_pptx,
            "xlsx": self._parse_xlsx,
            "md": self._parse_markdown,
            "txt": self._parse_text,
            "png": self._parse_image,
            "jpg": self._parse_image,
            "jpeg": self._parse_image,
        }
        parser = parser_map.get(ext, self._parse_text)
        return await parser(file_path, ext)

    # ------------------------------------------------------------------
    # PDF — PyMuPDF (文本) + PaddleOCR (扫描件)
    # ------------------------------------------------------------------
    async def _parse_pdf(self, path: str, ext: str) -> ParsedDocument:
        errors = []
        pages = []
        tables = []
        content_parts = []

        try:
            import fitz
            doc = fitz.open(path)
            metadata = {"page_count": len(doc), "format": "pdf"}

            # 复杂版面（多栏/多表/矢量绘制密集）优先走 MinerU；
            # 未安装或失败时回退 PyMuPDF，保证主链路可用。
            if self._pdf_needs_mineru(doc):
                doc.close()
                mineru_result = await self._parse_pdf_mineru(path, ext)
                if mineru_result is not None:
                    return mineru_result
                errors.append("MinerU 不可用或解析失败，回退 PyMuPDF")
                doc = fitz.open(path)

            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                content_parts.append(text)
                page_data = {"page_num": page_num, "text": text, "tables": [], "is_scanned": False}

                # 页级路由：空白文本 → OCR；复杂页已在文档级交给 MinerU
                if self.classify_pdf_page(page) == "scanned":
                    page_data["is_scanned"] = True
                    ocr_text = await self._ocr_page(page)
                    if ocr_text:
                        page_data["text"] = ocr_text
                        content_parts[-1] = ocr_text

                # 表格提取
                page_tables = page.find_tables()
                for tbl in page_tables:
                    rows = tbl.extract()
                    if rows:
                        table_data = self._structure_table(rows, page_num)
                        tables.append(table_data)
                        page_data["tables"].append(table_data)

                pages.append(page_data)

            doc.close()
        except ImportError:
            errors.append("PyMuPDF 未安装")
            metadata = {"format": "pdf"}
            content_parts = [f"(PyMuPDF 未安装，无法解析: {Path(path).name})"]
        except Exception as e:
            errors.append(str(e))
            metadata = {"format": "pdf"}
            content_parts = [f"(解析失败: {e})"]

        return ParsedDocument(
            title=Path(path).name,
            file_type=ext,
            content_markdown="\n\n".join(content_parts),
            pages=pages,
            tables=tables,
            metadata=metadata,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # PDF 页级/文档级路由
    # ------------------------------------------------------------------
    @staticmethod
    def classify_pdf_page(page) -> str:
        """页级路由：text（文本型）/ scanned（扫描页）/ complex（复杂版面）。

        判定依据：文本密度、文本块左右分布（多栏）、表格数量、矢量绘制密度。
        阈值偏保守——宁可走重型解析，也不要漏掉结构复杂页。
        """
        try:
            text = page.get_text("text") or ""
        except Exception:
            text = ""
        if len(text.strip()) < 50:
            return "scanned"

        try:
            width = float(page.rect.width) or 1.0
        except Exception:
            width = 1.0

        try:
            blocks = page.get_text("blocks") or []
        except Exception:
            blocks = []
        text_blocks = [b for b in blocks if len(b) >= 7 and b[6] == 0]
        left = sum(1 for b in text_blocks if float(b[2]) <= width * 0.52)
        right = sum(1 for b in text_blocks if float(b[0]) >= width * 0.48)
        multi_column = left >= 2 and right >= 2

        try:
            finder = page.find_tables()
            table_count = len(getattr(finder, "tables", finder) or [])
        except Exception:
            table_count = 0

        try:
            drawings = len(page.get_drawings() or [])
        except Exception:
            drawings = 0

        if multi_column or table_count >= 2 or drawings > 200:
            return "complex"
        return "text"

    @classmethod
    def _pdf_needs_mineru(cls, doc) -> bool:
        """扫描前若干页，任一页判定为复杂版面则整篇交给 MinerU。"""
        for i, page in enumerate(doc):
            if i >= MINERU_SCAN_PAGES:
                break
            if cls.classify_pdf_page(page) == "complex":
                return True
        return False

    async def _parse_pdf_mineru(self, path: str, ext: str) -> ParsedDocument | None:
        """调用 MinerU CLI 解析复杂版面 PDF。

        通过子进程隔离重型依赖；未安装、超时、非零退出或无输出均返回 None，
        由调用方回退 PyMuPDF。新版用 `mineru`，旧版可设 MINERU_CMD=magic-pdf。
        """
        cmd = os.getenv("MINERU_CMD", MINERU_CMD)
        timeout = int(os.getenv("MINERU_TIMEOUT", str(MINERU_TIMEOUT)))

        with tempfile.TemporaryDirectory(prefix="mineru_") as out_dir:
            try:
                proc = await asyncio.create_subprocess_exec(
                    cmd, "-p", path, "-o", out_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except (OSError, NotImplementedError):
                return None

            try:
                await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return None

            if proc.returncode != 0:
                return None

            md_files = sorted(
                Path(out_dir).rglob("*.md"),
                key=lambda p: p.stat().st_size,
                reverse=True,
            )
            if not md_files:
                return None

            content = md_files[0].read_text(encoding="utf-8")
            return ParsedDocument(
                title=Path(path).name,
                file_type=ext,
                content_markdown=content,
                tables=self._extract_markdown_tables(content),
                metadata={"format": "pdf", "parser": "mineru"},
                errors=[],
            )

    async def _ocr_page(self, page) -> str:
        """PaddleOCR 识别扫描页。"""
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(lang="ch")
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            result = ocr.ocr(img_bytes)
            if result and result[0]:
                return "\n".join(line[1][0] for line in result[0])
        except ImportError:
            pass
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    # Word — python-docx
    # ------------------------------------------------------------------
    async def _parse_docx(self, path: str, ext: str) -> ParsedDocument:
        errors = []
        content_parts = []
        tables = []

        try:
            from docx import Document as DocxDocument
            doc = DocxDocument(path)

            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    content_parts.append("")
                    continue
                if para.style.name.startswith("Heading"):
                    level = int(para.style.name.split()[-1]) if para.style.name.split()[-1].isdigit() else 1
                    content_parts.append(f"{'#' * min(level, 6)} {text}")
                else:
                    content_parts.append(text)

            for i, table in enumerate(doc.tables):
                rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
                if rows:
                    tables.append(self._structure_table(rows, page_num=None, caption=f"Table {i+1}"))

        except ImportError:
            errors.append("python-docx 未安装")
        except Exception as e:
            errors.append(str(e))

        return ParsedDocument(
            title=Path(path).name,
            file_type=ext,
            content_markdown="\n\n".join(content_parts),
            tables=tables,
            metadata={"format": "docx"},
            errors=errors,
        )

    # ------------------------------------------------------------------
    # PPT — python-pptx
    # ------------------------------------------------------------------
    async def _parse_pptx(self, path: str, ext: str) -> ParsedDocument:
        errors = []
        slides_content = []

        try:
            from pptx import Presentation
            prs = Presentation(path)
            for i, slide in enumerate(prs.slides, 1):
                slide_texts = []
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for para in shape.text_frame.paragraphs:
                            t = para.text.strip()
                            if t:
                                slide_texts.append(t)
                    if shape.has_table:
                        rows = [[cell.text.strip() for cell in row.cells] for row in shape.table.rows]
                        # inline table as markdown
                        if rows:
                            slide_texts.append("| " + " | ".join(rows[0]) + " |")
                            slide_texts.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
                            for row in rows[1:]:
                                slide_texts.append("| " + " | ".join(row) + " |")
                slides_content.append(f"## Slide {i}\n\n" + "\n".join(slide_texts))
                # 备注
                if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                    notes = slide.notes_slide.notes_text_frame.text.strip()
                    if notes:
                        slides_content[-1] += f"\n\n> 备注: {notes}"

        except ImportError:
            errors.append("python-pptx 未安装")
        except Exception as e:
            errors.append(str(e))

        return ParsedDocument(
            title=Path(path).name,
            file_type=ext,
            content_markdown="\n\n---\n\n".join(slides_content),
            metadata={"format": "pptx"},
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Excel — openpyxl → JSON树 + Markdown 表
    # ------------------------------------------------------------------
    async def _parse_xlsx(self, path: str, ext: str) -> ParsedDocument:
        errors = []
        sheets_content = []
        tables = []

        try:
            from openpyxl import load_workbook
            wb = load_workbook(path, data_only=True)
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                rows = []
                for row in ws.iter_rows(values_only=True):
                    rows.append([str(c) if c is not None else "" for c in row])
                if rows:
                    sheets_content.append(f"## Sheet: {sheet_name}\n")
                    md = self._rows_to_markdown(rows)
                    sheets_content[-1] += md
                    tables.append(self._structure_table(rows, caption=sheet_name))
        except ImportError:
            errors.append("openpyxl 未安装")
        except Exception as e:
            errors.append(str(e))

        return ParsedDocument(
            title=Path(path).name,
            file_type=ext,
            content_markdown="\n\n".join(sheets_content),
            tables=tables,
            metadata={"format": "xlsx"},
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Markdown / 纯文本
    # ------------------------------------------------------------------
    async def _parse_markdown(self, path: str, ext: str) -> ParsedDocument:
        try:
            content = Path(path).read_text(encoding="utf-8")
        except Exception:
            content = f"(无法读取: {Path(path).name})"

        tables = self._extract_markdown_tables(content)
        return ParsedDocument(
            title=Path(path).name, file_type=ext,
            content_markdown=content, tables=tables,
            metadata={"format": "markdown"},
        )

    async def _parse_text(self, path: str, ext: str) -> ParsedDocument:
        return await self._parse_markdown(path, ext)

    @staticmethod
    def _extract_markdown_tables(content: str) -> list[dict]:
        """从 Markdown 文本提取 `| a | b |` 表格，转为结构化表格 dict。"""
        tables = []
        lines = content.split("\n")
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i].strip()
            if not line.startswith("|"):
                i += 1
                continue
            # 收集连续的表格行
            block = []
            while i < n and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1

            rows = []
            for bl in block:
                cells = [c.strip() for c in bl.strip("|").split("|")]
                # 跳过纯分隔符行
                if all(set(c) <= {"-", ":", " "} for c in cells):
                    continue
                rows.append(cells)

            if len(rows) >= 2:
                tbl = DocumentParser._structure_table(rows, caption=f"Markdown Table {len(tables)+1}")
                tables.append(tbl)
        return tables

    # ------------------------------------------------------------------
    # 图片 — OCR + BLIP描述
    # ------------------------------------------------------------------
    async def _parse_image(self, path: str, ext: str) -> ParsedDocument:
        ocr_text = ""
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(lang="ch")
            result = ocr.ocr(path)
            if result and result[0]:
                ocr_text = "\n".join(line[1][0] for line in result[0])
        except Exception:
            pass

        return ParsedDocument(
            title=Path(path).name,
            file_type=ext,
            content_markdown=ocr_text or f"[图片: {Path(path).name}]",
            metadata={"format": "image", "has_ocr": bool(ocr_text)},
        )

    # ------------------------------------------------------------------
    # 表格结构化为 Markdown + "表头:值"格式
    # ------------------------------------------------------------------
    @staticmethod
    def _structure_table(rows: list[list[str]], page_num: int | None = None,
                         caption: str = "Table") -> dict:
        """将表格转为 Markdown + 结构化描述。"""
        if not rows:
            return {"markdown": "", "structured": "", "headers": [], "rows": [], "page": page_num, "caption": caption}

        headers = rows[0] if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []

        # Markdown 格式
        md = "| " + " | ".join(str(h) for h in headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in data_rows:
            padded = list(row) + [""] * (len(headers) - len(row))
            md += "| " + " | ".join(str(c)[:100] for c in padded[:len(headers)]) + " |\n"

        # 结构化描述 ("表头:值") — LLM可直接理解
        structured_parts = []
        for row in data_rows[:20]:  # 最多前20行
            parts = []
            for j, val in enumerate(row):
                if j < len(headers) and val:
                    parts.append(f"{headers[j]}:{val}")
            if parts:
                structured_parts.append("; ".join(parts))

        return {
            "markdown": md,
            "structured": "\n".join(structured_parts),
            "headers": [str(h) for h in headers],
            "rows": [[str(c) for c in row] for row in data_rows],
            "page": page_num,
            "caption": caption,
        }

    @staticmethod
    def _rows_to_markdown(rows: list[list[str]]) -> str:
        if not rows:
            return ""
        headers = rows[0]
        md = "| " + " | ".join(str(h) for h in headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows[1:]:
            padded = list(row) + [""] * (len(headers) - len(row))
            md += "| " + " | ".join(str(c)[:100] for c in padded[:len(headers)]) + " |\n"
        return md


# 单例
doc_parser = DocumentParser()

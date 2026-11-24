"""解析路由：文本页 / 扫描页 / 复杂版面 → PyMuPDF / PaddleOCR / MinerU。"""
import asyncio
from pathlib import Path

from src.document_pipeline import doc_parser


class FakeRect:
    width = 600.0


class FakeTableFinder:
    def __init__(self, count: int):
        self.tables = [object()] * count


class FakePage:
    def __init__(self, text: str = "", blocks=None, tables: int = 0, drawings: int = 0):
        self.text = text
        self.blocks = blocks or []
        self._tables = tables
        self._drawings = drawings
        self.rect = FakeRect()

    def get_text(self, kind: str = "text"):
        return self.blocks if kind == "blocks" else self.text

    def find_tables(self):
        return FakeTableFinder(self._tables)

    def get_drawings(self):
        return [None] * self._drawings


class FakeDoc:
    def __init__(self, pages):
        self.pages = pages

    def __iter__(self):
        return iter(self.pages)


def test_classify_text_page():
    assert doc_parser.classify_pdf_page(FakePage("正常文本内容。" * 50)) == "text"


def test_classify_scanned_page():
    assert doc_parser.classify_pdf_page(FakePage("")) == "scanned"
    assert doc_parser.classify_pdf_page(FakePage("短")) == "scanned"


def test_classify_multi_column_page():
    blocks = [
        (0, 0, 200, 20, "左栏一", 0, 0),
        (0, 30, 200, 50, "左栏二", 0, 0),
        (350, 0, 550, 20, "右栏一", 0, 0),
        (350, 30, 550, 50, "右栏二", 0, 0),
    ]
    page = FakePage("多栏正文内容。" * 50, blocks=blocks)
    assert doc_parser.classify_pdf_page(page) == "complex"


def test_classify_table_heavy_page():
    assert doc_parser.classify_pdf_page(FakePage("表格正文。" * 50, tables=2)) == "complex"


def test_classify_drawing_heavy_page():
    assert doc_parser.classify_pdf_page(FakePage("图形正文。" * 50, drawings=201)) == "complex"


def test_pdf_needs_mineru_if_any_page_complex():
    text_page = FakePage("普通正文。" * 50)
    complex_page = FakePage("表格正文。" * 50, tables=2)
    assert doc_parser._pdf_needs_mineru(FakeDoc([text_page, complex_page])) is True
    assert doc_parser._pdf_needs_mineru(FakeDoc([text_page, FakePage("")])) is False


def test_mineru_missing_returns_none(monkeypatch):
    monkeypatch.setenv("MINERU_CMD", "definitely-not-installed-mineru-xyz")
    result = asyncio.run(doc_parser._parse_pdf_mineru("dummy.pdf", "pdf"))
    assert result is None


def test_mineru_success_parses_markdown(monkeypatch, tmp_path):
    class FakeProc:
        returncode = 0

        async def communicate(self):
            return b"", b""

        def kill(self):
            pass

    async def fake_exec(*args, **kwargs):
        out_dir = Path(args[4])  # cmd -p path -o out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.md").write_text(
            "# 复杂版面\n\n| 区域 | 销售额 |\n| --- | --- |\n| 华东 | 120 |\n",
            encoding="utf-8",
        )
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    result = asyncio.run(doc_parser._parse_pdf_mineru(str(tmp_path / "complex.pdf"), "pdf"))

    assert result is not None
    assert result.metadata["parser"] == "mineru"
    assert "复杂版面" in result.content_markdown
    assert result.tables
    assert result.tables[0]["headers"] == ["区域", "销售额"]

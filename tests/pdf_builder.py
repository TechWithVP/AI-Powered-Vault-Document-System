"""Minimal PDF generator for tests.

Builds a syntactically valid multi-page PDF containing selectable text, so PDF
extraction and page attribution can be exercised without adding a PDF-writing
dependency to the project.
"""

from __future__ import annotations

PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LINE_HEIGHT = 16


BACKSLASH = chr(92)


def _escape(text: str) -> str:
    """Escape the characters that are special inside a PDF string literal."""
    return (
        text.replace(BACKSLASH, BACKSLASH * 2)
        .replace("(", BACKSLASH + "(")
        .replace(")", BACKSLASH + ")")
    )


def _content_stream(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 12 Tf", f"1 0 0 1 72 {PAGE_HEIGHT - 72} Tm", "14 TL"]
    for line in lines:
        commands.append(f"({_escape(line)}) Tj")
        commands.append("T*")
    commands.append("ET")
    return "\n".join(commands).encode("latin-1")


def build_pdf(pages: list[list[str]]) -> bytes:
    """Return a PDF where each element of pages is that page's lines of text."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)  # object numbers are 1-based

    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    # Reserve the catalog and pages-tree object numbers before the page objects
    # so each page can reference its parent.
    catalog_id = len(objects) + 1
    pages_id = catalog_id + 1
    objects.append(b"")  # catalog placeholder
    objects.append(b"")  # pages placeholder

    page_ids: list[int] = []
    for lines in pages:
        stream = _content_stream(lines)
        content_id = add(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
        page_ids.append(
            add(
                f"<< /Type /Page /Parent {pages_id} 0 R "
                f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>".encode()
            )
        )

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[catalog_id - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode()
    objects[pages_id - 1] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()
    return bytes(out)

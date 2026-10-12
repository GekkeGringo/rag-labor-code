from pathlib import Path

from docx import Document


def docx_to_text(docx_path: Path, out_path: Path) -> None:
    doc = Document(docx_path)
    lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    docx_to_text(Path("data/raw/tk_rf.docx"), Path("data/raw/tk_rf.txt"))
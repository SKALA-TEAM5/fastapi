from docling.document_converter import DocumentConverter


def convert_pdf_to_markdown(pdf_path: str) -> str:
    print(f"[{pdf_path}] PDF 변환 시작 (Docling)...")
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()

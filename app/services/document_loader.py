import fitz
import os
import io

import pytesseract

from PIL import Image


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def load_document(file_path):

    pdf = fitz.open(file_path)

    pages = []

    metadata = pdf.metadata

    title = metadata.get("title")

    if not title:
        title = os.path.basename(file_path)

    for page_num in range(len(pdf)):

        page = pdf[page_num]

        # Try normal text extraction first
        page_text = page.get_text()

        # OCR fallback for scanned/image PDFs
        if not page_text or len(page_text.strip()) < 20:

            print(
                f"Running OCR on page {page_num + 1} "
                f"of {os.path.basename(file_path)}"
            )

            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2)
            )

            image_bytes = pix.tobytes("png")

            image = Image.open(
                io.BytesIO(image_bytes)
            )

            page_text = pytesseract.image_to_string(
                image,
                lang="eng"
            )

        if page_text and page_text.strip():

            pages.append(
                {
                    "text": page_text,
                    "page": page_num + 1,
                    "source": title,
                    "file_name": os.path.basename(
                        file_path
                    )
                }
            )

    pdf.close()

    return pages
import fitz
import os
import io

import pytesseract

from PIL import Image


class DocumentLoader:

    def __init__(self):

        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

    def load_document(self, file_path):

        pdf = fitz.open(file_path)

        pages = []

        title = os.path.basename(file_path)

        for page_num in range(len(pdf)):

            page = pdf[page_num]

            page_text = page.get_text()

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
                        "file_name": title
                    }
                )

        pdf.close()

        return pages
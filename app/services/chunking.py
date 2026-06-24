class TextChunker:

    def __init__(
        self,
        chunk_size=800,
        overlap=200
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, pages):

        chunks = []

        for page_data in pages:

            text = page_data["text"]
            page = page_data["page"]
            source = page_data["source"]

            start = 0

            while start < len(text):

                end = start + self.chunk_size

                chunk = text[start:end].strip()

                if len(chunk) >= 50:

                    chunks.append(
                        {
                            "text": chunk,
                            "page": page,
                            "source": source
                        }
                    )

                start += (
                    self.chunk_size
                    - self.overlap
                )

        return chunks
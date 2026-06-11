def chunk_text(
    pages,
    chunk_size=800,
    overlap=200
):

    chunks = []

    for page_data in pages:

        text = page_data["text"]
        page = page_data["page"]
        source = page_data["source"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end].strip()

            # Skip very small chunks
            if len(chunk) >= 50:

                chunks.append(
                    {
                        "text": chunk,
                        "page": page,
                        "source": source
                    }
                )

            start += chunk_size - overlap

    return chunks
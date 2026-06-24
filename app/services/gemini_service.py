from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()


class GeminiService:

    def __init__(self):

        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def generate_embeddings(self, texts):

        embeddings = self.embedding_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings.tolist()

    def generate_embedding(self, text):

        embedding = self.embedding_model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def generate_answer(
        self,
        context,
        question
    ):

        prompt = f"""
Context:
{context}

Question:
{question}

Instructions:
- Answer only from the provided context.
- If the answer is not in the context, say so.
- Be concise and accurate.
"""

        try:

            response = self.client.chat.completions.create(

                model="llama-3.1-8b-instant",

                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful travel assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0.3,
                max_tokens=1024

            )

            return (
                response
                .choices[0]
                .message
                .content
            )

        except Exception as e:

            return f"Groq Error: {str(e)}"
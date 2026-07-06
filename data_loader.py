import google.generativeai as genai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("gemini_api")
genai.configure(api_key=api_key)
EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 768

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        embedding = genai.embed_content(
            model=EMBED_MODEL,
            content=text,
            output_dimensionality=EMBED_DIM
        )
        embeddings.append(embedding["embedding"])
    return embeddings
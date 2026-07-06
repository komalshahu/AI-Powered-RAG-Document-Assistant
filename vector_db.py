from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    _client = None

    def __init__(self, url="http://localhost:6333", collection="docs", dim=768):
        self.collection = collection
        if QdrantStorage._client is None:
            try:
                # Try to connect to remote server first
                client = QdrantClient(url=url, timeout=3)
                client.collection_exists(self.collection)
                QdrantStorage._client = client
            except Exception:
                # Fall back to local storage directory
                QdrantStorage._client = QdrantClient(path="./qdrant_storage")

        self.client = QdrantStorage._client

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        )
        contexts = []
        sources = set()

        for r in response.points:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}
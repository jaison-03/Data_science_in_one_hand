from core.retriever import Retriever


class FakeEmbeddingService:
    def embed_query(self, question: str):
        return [1.0, 0.0, 0.0]


class FakeVectorStore:
    def similarity_search(self, query_embedding, top_k, filter_by=None, threshold=None):
        return [
            type("Result", (), {"chunk_id": "chunk-1", "text": "Alpha beta gamma", "metadata": {"source_file": "report.txt"}, "score": 0.92})(),
            type("Result", (), {"chunk_id": "chunk-2", "text": "Nothing relevant here", "metadata": {"source_file": "notes.txt"}, "score": 0.10})(),
        ]


def test_retriever_returns_results_for_valid_question():
    retriever = Retriever(vector_store=FakeVectorStore(), embedding_service=FakeEmbeddingService(), top_k=2)
    bundle = retriever.search("What is the report about?")
    assert bundle.query == "What is the report about?"
    assert len(bundle.results) == 2
    assert bundle.results[0].chunk_id == "chunk-1"

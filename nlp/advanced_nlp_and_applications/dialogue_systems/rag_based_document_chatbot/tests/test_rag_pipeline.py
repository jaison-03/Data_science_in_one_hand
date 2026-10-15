from core.chunking import chunk_document
from core.document_loader import DocumentLoader
from core.rag_pipeline import RAGPipeline
from core.retriever import Retriever
from core.vector_store import FAISSVectorStore


class FakeEmbeddingService:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, question):
        return [1.0, 0.0, 0.0]


class FakeLLMService:
    def __init__(self):
        self.prompt = None

    def generate(self, prompt):
        self.prompt = prompt
        return "The document says the answer is 42."


def test_uploaded_document_can_be_retrieved_and_answered(tmp_path):
    file_path = tmp_path / "facts.txt"
    file_path.write_text("The answer to the question is 42.", encoding="utf-8")
    document = DocumentLoader().load(file_path)
    chunks = chunk_document(document, chunk_size=200, chunk_overlap=0)

    embeddings = FakeEmbeddingService()
    vector_store = FAISSVectorStore(dimension=3)
    vector_store.add_documents(chunks, embeddings.embed_documents([chunk["text"] for chunk in chunks]))
    retriever = Retriever(vector_store=vector_store, embedding_service=embeddings, top_k=1)
    llm_service = FakeLLMService()
    pipeline = RAGPipeline(vector_store, embeddings, llm_service, retriever)

    result = pipeline.answer_question("What is the answer?")

    assert result["answer"] == "The document says the answer is 42."
    assert result["grounded"] is True
    assert result["sources"][0]["filename"] == "facts.txt"
    assert "The answer to the question is 42." in llm_service.prompt[1].content

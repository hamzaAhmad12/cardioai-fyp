from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
import numpy as np


MEDICAL_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are a clinical guideline assistant helping healthcare professionals understand cardiovascular and hypertension guidelines.

STRICT RULES:
- Answer ONLY from the provided guideline context
- If not in context: "This specific information is not in the provided guidelines. Please consult the full document."
- Always cite the source guideline and page number
- Never give personal medical advice
- For drug dosages: quote exactly as written in guidelines

PREVIOUS CONVERSATION:
{chat_history}

GUIDELINE CONTEXT:
{context}

CLINICAL QUESTION: {question}

ANSWER:"""
)


class SimpleMedicalReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents, top_n: int = 4):
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = float(score)

        sorted_docs = [
            doc for _, doc in sorted(
                zip(scores, documents),
                key=lambda x: x[0],
                reverse=True
            )
        ]
        return sorted_docs[:top_n]


class AdvancedMedicalRAG:
    def __init__(self, persist_dir="./chroma_db"):
        print("🔧 Initializing Advanced Medical RAG...\n")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )

        self.llm = ChatOllama(model="llama3.2", temperature=0.1)
        self.reranker = SimpleMedicalReranker()
        self.chat_history = []
        self.max_history = 3

        print("✅ All components ready!\n")

    def ask(self, question: str, top_k_retrieve: int = 10, top_k_final: int = 4):
        print(f"\n🔍 Retrieving {top_k_retrieve} candidate chunks...")

        raw_results = self.vector_store.similarity_search_with_score(
            question, k=top_k_retrieve
        )

        if not raw_results:
            return {
                "answer": "No relevant information found.",
                "sources": [],
                "confidence": 0.0
            }

        chunks = [doc for doc, _ in raw_results]

        print(f"🎯 Re-ranking to find best {top_k_final}...")
        reranked_chunks = self.reranker.rerank(question, chunks, top_n=top_k_final)

        context_parts = []
        sources = []

        for chunk in reranked_chunks:
            guideline = chunk.metadata.get("guideline", "Unknown")
            page = chunk.metadata.get("page", "N/A")
            score = chunk.metadata.get("rerank_score", 0.0)

            context_parts.append(
                f"[GUIDELINE: {guideline} | PAGE: {page}]\n{chunk.page_content}"
            )
            sources.append({
                "guideline": guideline,
                "page": page,
                "relevance_score": f"{score:.3f}",
                "preview": chunk.page_content[:200] + "..."
            })

        context = "\n\n"  + "\n\n".join(context_parts)

        history_text = ""
        for q, a in self.chat_history[-self.max_history:]:
            history_text += f"Q: {q}\nA: {a[:200]}...\n\n"

        if not history_text:
            history_text = "No previous conversation."

        prompt = MEDICAL_PROMPT.format(
            context=context,
            question=question,
            chat_history=history_text
        )

        print("💭 Generating answer...")
        response = self.llm.invoke(prompt)
        answer = response.content

        if sources:
            avg_score = np.mean([float(s["relevance_score"]) for s in sources])
            confidence = round(min(100, max(0, avg_score * 100)), 1)
        else:
            confidence = 0.0

        self.chat_history.append((question, answer))

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "chunks_retrieved": len(chunks),
            "chunks_used": len(reranked_chunks)
        }

    def display(self, result: dict, question: str):
        print("\n" )
        print(f"❓ QUESTION: {question}")
        print()
        print(f"\n💊 ANSWER:\n{result['answer']}")
        print(f"\n📊 CONFIDENCE: {result['confidence']}%")
        print(f"📄 Retrieved: {result['chunks_retrieved']} → Used: {result['chunks_used']}")
        print("\n📚 SOURCES:")
        for i, src in enumerate(result["sources"], 1):
            print(f"  {i}. {src['guideline']} — Page {src['page']}")
            print(f"     Relevance: {src['relevance_score']}")
            print(f"     \"{src['preview']}\"")
        print()

    def reset_memory(self):
        self.chat_history = []
        print("🔄 Conversation history cleared.")


if __name__ == "__main__":
    print()
    print("🏥 Advanced Medical Guideline Chatbot")
    print()

    rag = AdvancedMedicalRAG()

    print("Commands: 'quit' to exit | 'clear' to reset memory\n")

    while True:
        question = input("📋 Your question: ").strip()

        if not question:
            continue
        if question.lower() in ["quit", "exit"]:
            print("👋 Goodbye!")
            break
        if question.lower() == "clear":
            rag.reset_memory()
            continue

        result = rag.ask(question)
        rag.display(result, question)
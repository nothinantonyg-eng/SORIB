import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from groq import Groq
import chromadb

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATA_PATH = "./data"
CHROMA_PATH = "./chroma_db"

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model
Settings.llm = None

groq_client = Groq(api_key=GROQ_API_KEY)

_index = None

IB_KEYWORDS = [
    "ib", "bachillerato", "diploma", "monografía", "cas", "tok",
    "teoría del conocimiento", "criterio", "evaluación", "materia",
    "grupo", "nivel superior", "nivel medio", "coar", "examen",
    "calificación", "descriptor", "tarea", "exploración", "ia",
    "matemática", "química", "física", "biología", "historia",
    "lengua", "literatura", "inglés", "español", "geografía",
    "economía", "psicología", "filosofía", "arte", "música",
    "teatro", "sociedad digital", "comp sci", "informática"
]

def is_ib_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in IB_KEYWORDS)

def is_clean(text: str) -> bool:
    text = text.strip()
    if len(text) < 150:
        return False
    junk = ["]_PUSHDATA", "PUSHDATA", ".getInfo()", "Bearer ",
            "\x00", "\\u0000", "_M9", "||||"]
    if any(j in text for j in junk):
        return False
    printable_ratio = sum(c.isprintable() for c in text) / len(text)
    if printable_ratio < 0.85:
        return False
    return True

def index_documents():
    global _index
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_or_create_collection("sorib")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    if chroma_collection.count() > 0:
        print(f"Base de datos existente con {chroma_collection.count()} documentos.")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        _index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context)
        return

    print("Cargando documentos por primera vez...")
    documents = SimpleDirectoryReader(
        DATA_PATH,
        recursive=True,
        required_exts=[".pdf", ".docx", ".doc", ".txt"]
    ).load_data()
    print(f"{len(documents)} documentos cargados.")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    _index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context)
    print("Indexación completada.")

def query_documents(question: str, history: list = []) -> dict:
    global _index

    def build_messages(system_prompt, extra_user_content=None):
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": extra_user_content or question})
        return messages

    if not is_ib_question(question):
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=build_messages(
                """Eres SORIB, un asistente académico del Programa del 
Diploma IB del COAR Lima. Responde siempre en el mismo idioma 
en que el usuario escribió su pregunta. Si te preguntan algo 
sobre el IB o el COAR, indica que puedes buscar en los documentos 
oficiales para una respuesta más precisa."""
            ),
            temperature=0.5,
            max_tokens=1024,
        )
        return {"answer": response.choices[0].message.content, "sources": []}

    if _index is None:
        return {"answer": "Error: documentos no indexados.", "sources": []}

    retriever = _index.as_retriever(similarity_top_k=12)
    nodes = retriever.retrieve(question)

    context_parts = []
    sources = []

    for node in nodes:
        text = node.text.strip()
        if is_clean(text):
            context_parts.append(text)
            # Extraer nombre del archivo fuente
            filename = ""
            if hasattr(node, "metadata") and node.metadata:
                filename = (
                    node.metadata.get("file_name")
                    or node.metadata.get("filename")
                    or node.metadata.get("source")
                    or ""
                )
            if filename and filename not in sources:
                sources.append(filename)

    if not context_parts:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=build_messages(
                """Eres SORIB, un asistente académico especializado en el Programa 
del Diploma IB. Responde siempre en el mismo idioma en que el usuario 
escribió su pregunta. Sé claro y preciso."""
            ),
            temperature=0.3,
            max_tokens=1024,
        )
        return {"answer": response.choices[0].message.content, "sources": []}

    context = "\n\n".join(context_parts[:5])
    system_prompt = f"""Eres SORIB, un asistente académico especializado en el Programa 
del Diploma del Bachillerato Internacional (IB) del COAR Lima.

Responde siempre en el mismo idioma en que el usuario escribió su pregunta.
- No generes preguntas adicionales al final
- Si los criterios de evaluación están en el contexto, listarlos claramente
- Si el contexto no es suficiente, complementa con tu conocimiento del IB

Contexto de documentos oficiales:
{context}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=build_messages(system_prompt),
        temperature=0.2,
        max_tokens=1024,
    )

    return {"answer": response.choices[0].message.content, "sources": sources[:3]}
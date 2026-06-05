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

def query_documents(question: str) -> str:
    global _index

    # Si no es pregunta sobre IB/COAR, responde directamente
    if not is_ib_question(question):
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """Eres SORIB, un asistente académico del Programa del 
Diploma IB del COAR Lima. Responde de forma clara y útil en español. 
Si te preguntan algo sobre el IB o el COAR, indica que puedes buscar 
en los documentos oficiales para una respuesta más precisa."""
                },
                {"role": "user", "content": question}
            ],
            temperature=0.5,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    # Si es sobre IB/COAR, busca en documentos
    if _index is None:
        return "Error: documentos no indexados."

    retriever = _index.as_retriever(similarity_top_k=12)
    nodes = retriever.retrieve(question)

    context_parts = []
    for node in nodes:
        text = node.text.strip()
        if is_clean(text):
            context_parts.append(text)

    if not context_parts:
        # Si no encuentra nada limpio, responde con conocimiento propio
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """Eres SORIB, un asistente académico especializado 
en el Programa del Diploma IB. Responde en español con tu conocimiento 
general del IB, siendo claro y preciso."""
                },
                {"role": "user", "content": question}
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    context = "\n\n".join(context_parts[:5])

    prompt = f"""Eres SORIB, un asistente académico especializado en el Programa 
del Diploma del Bachillerato Internacional (IB) del COAR Lima.

Responde de forma clara y estructurada basándote en el contexto proporcionado.
- Responde siempre en español
- No generes preguntas adicionales al final
- Si los criterios de evaluación están en el contexto, listarlos claramente
- Si el contexto no es suficiente, complementa con tu conocimiento del IB

Contexto de documentos oficiales:
{context}

Pregunta: {question}

Respuesta:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )

    return response.choices[0].message.content
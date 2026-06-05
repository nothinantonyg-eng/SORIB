\# SORIB — Asesor Académico IB · COAR Lima



> AI-powered academic assistant for students and teachers of the IB Diploma 

> Programme at COAR Lima, Peru's most selective public school.



!\[Python](https://img.shields.io/badge/Python-3.11-blue)

!\[FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)

!\[LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG-orange)

!\[Groq](https://img.shields.io/badge/Groq-LLaMA\_3.3\_70B-purple)



\---



\## What is SORIB?



SORIB (from \*Asesor IB\*) is a Retrieval-Augmented Generation (RAG) chatbot 

built to help students and teachers of the International Baccalaureate (IB) 

Diploma Programme at COAR Lima navigate course materials, evaluation criteria, 

and academic requirements — instantly and accurately.



\*\*The problem it solves:\*\* COAR Lima's IB programme involves hundreds of 

official documents across 6 subject groups, core components (TOK, EE, CAS), 

and internal assessment guidelines. Students and teachers spend significant 

time searching for specific information across these documents. SORIB makes 

that knowledge instantly accessible through a natural language interface.



\*\*Who built it:\*\* Gerald Rojas Calderón, COAR Lima IB graduate (Class of 2025), 

self-taught, built as a personal project during university application preparation.



\---



\## How it works

User question

│

▼

Is it IB-related?

│

┌──┴──┐

Yes    No

│      └──► Direct LLM response (Groq)

▼

Semantic search over 1,200+ IB/COAR documents

│

▼

Top relevant chunks retrieved (ChromaDB)

│

▼

Context + question sent to LLaMA 3.3 70B via Groq API

│

▼

Structured answer displayed in chat interface



\*\*Tech stack:\*\*

\- \*\*Backend:\*\* Python 3.11, FastAPI, LlamaIndex

\- \*\*Vector store:\*\* ChromaDB (persistent, local)

\- \*\*Embeddings:\*\* sentence-transformers/all-MiniLM-L6-v2 (HuggingFace)

\- \*\*LLM:\*\* LLaMA 3.3 70B Versatile via Groq API (free tier)

\- \*\*Frontend:\*\* Vanilla HTML/CSS/JavaScript with marked.js for markdown rendering



\---



\## Features



\- 🔍 \*\*RAG over 1,200+ documents\*\* — IB subject guides, internal assessment 

&#x20; criteria, TOK, Extended Essay guidelines, COAR-specific materials

\- 💬 \*\*Hybrid response mode\*\* — uses document context for IB-specific questions, 

&#x20; general LLM knowledge for everything else

\- 📝 \*\*Markdown rendering\*\* — structured responses with headers, lists, bold text

\- ⚡ \*\*Fast\*\* — document index built once and persisted locally; subsequent 

&#x20; startups load in seconds

\- 🌐 \*\*Clean web interface\*\* — dark mode, typing indicators, suggested questions



\---



\## Project structure

SORIB/

├── backend/

│   ├── main.py          # FastAPI app, endpoints, startup indexing

│   ├── rag.py           # RAG logic: indexing, retrieval, Groq inference

│   ├── requirements.txt # Python dependencies

│   └── .env.example     # Environment variables template

└── frontend/

└── index.html       # Complete frontend (HTML + CSS + JS)

\---



\## Setup \& Installation



\### Prerequisites

\- Python 3.11

\- A \[Groq API key](https://console.groq.com) (free)

\- Your IB/COAR documents in `backend/data/`



\### Steps



```bash

\# 1. Clone the repository

git clone https://github.com/nothinantonyg-eng/SORIB.git

cd SORIB/backend



\# 2. Create virtual environment

py -3.11 -m venv venv

venv\\Scripts\\activate  # Windows

\# source venv/bin/activate  # Linux/Mac



\# 3. Install dependencies

pip install -r requirements.txt



\# 4. Configure environment

cp .env.example .env

\# Edit .env and add your GROQ\_API\_KEY



\# 5. Add your documents

\# Place PDF, DOCX, DOC, or TXT files in backend/data/



\# 6. Run the backend

python -m uvicorn main:app --reload



\# 7. Open the frontend

\# Open frontend/index.html in your browser

```



> \*\*Note:\*\* First run will index all documents (may take several hours 

> depending on document volume). Subsequent runs load instantly from 

> the persisted ChromaDB database.



\---



\## Why I built this



I graduated from COAR Lima in 2025 after completing the IB Diploma Programme. 

During my time there, I noticed that students — especially in their first year — 

struggled to find specific information about assessment criteria, TOK requirements, 

or subject-specific guidelines buried across hundreds of official documents.



SORIB is my attempt to make that knowledge accessible. It's designed for the 

actual community I came from, using documents I know firsthand.



This project was built entirely self-taught, without formal CS education, 

using open-source tools and free APIs — because access to good tools 

shouldn't depend on resources.



\---



\## Roadmap



\- \[ ] Web deployment (Render + Vercel)

\- \[ ] User authentication for COAR students/teachers

\- \[ ] Source citation — show which document each answer came from

\- \[ ] Multilingual support (Spanish primary, English secondary)

\- \[ ] Mobile-responsive design



\---



\## License



MIT License — feel free to adapt this for your own school or institution.



\---



\*Built by Gerald Rojas Calderón · COAR Lima Class of 2025 · 

\[GitHub](https://github.com/nothinantonyg-eng)\*


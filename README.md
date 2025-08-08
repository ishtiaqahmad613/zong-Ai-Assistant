# 📱 Zong AI Assistant

Zong AI Assistant is a **Retrieval-Augmented Generation (RAG)** based virtual assistant built with **LangChain**, **FAISS**, and **Google Gemini 1.5 Flash**.  
It helps users quickly find accurate answers to Zong Telecom queries such as **packages, service codes, SIM issues, and complaints**, using PDF-sourced data, fuzzy matching, and intent detection.

---

## 🚀 Features
- **RAG-powered search** for accurate, context-aware answers from PDF data.
- **Intent detection** to handle greetings, gratitude, and telecom-specific queries.
- **Fuzzy text matching** to correct typos and slang in user input.
- **Conversational memory** for smooth, multi-turn interactions.
- Handles:
  - Internet/Data packages
  - Call/SMS bundles
  - Social bundles (WhatsApp, YouTube, etc.)
  - Balance check & recharge info
  - SIM activation, blocking, and more
  - Complaint handling and support info

---

## 🛠️ Tech Stack
- **Python**
- **LangChain**
- **FAISS** (vector database)
- **Google Gemini 1.5 Flash** (via `langchain-google-genai`)
- **PyPDFLoader** for PDF ingestion
- **RapidFuzz** for typo correction

---

## 📂 Project Structure

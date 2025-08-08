# zong ai assistant - Streamlit app for a smart telecom assistant powered by LangChain + Gemini
# Author: your-name-here (replace). Purpose: answer Zong-related queries (packages, balance, complaints).
# Notes: keep your GOOGLE_API_KEY in a .env (don't commit it). Update PDF path(s) below.

import streamlit as st
import os
import random
import re
from dotenv import load_dotenv
from rapidfuzz import process, fuzz

# LangChain community components & Google Gen AI connectors
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# ---------------------------
# Configuration & environment
# ---------------------------
load_dotenv()
# store the key in environment for the Google GenAI library to pick up
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# --------------------------------
# Embedding model (cached resource)
# --------------------------------
@st.cache_resource
def get_embeddings():
    """
    Return the embedding model instance.
    Cached so we don't re-create it on every rerun (Streamlit reloads often).
    """
    # model name depends on the langchain-google-genai package — adjust if needed
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# ------------------------------------------------
# Fuzzy input correction (helps with typos / slang)
# ------------------------------------------------
def fuzzy_correct(query, terms, threshold=80):
    """
    Try to find the closest known term (above threshold).
    If no close match, return the original query unchanged.
    This helps map user typos like "internt pakge" -> "internet package".
    """
    result = process.extractOne(query, terms, scorer=fuzz.token_sort_ratio)
    if result:
        match, score, _ = result
        return match if score >= threshold else query
    return query

# A curated list of common keywords / phrases for Zong-related intents.
# Expand this list as you cover more offers or services.
all_known_terms = [
    "internet package", "data package", "weekly package", "monthly package", "daily package",
    "call package", "sms package", "social bundle", "whatsapp package", "youtube package",
    "balance check", "balance", "recharge", "load", "how to subscribe", "unsubscribe",
    "zong code", "subscription code", "activation", "deactivation", "complaint", "network issue",
    "slow internet", "no signals", "zong app", "zong website", "sim", "zong sim", "number check",
    "package details", "zong helpline", "customer service", "zong offer", "stay at home offer",
    "location based offer", "zong 4g", "zong 5g", "international call", "roaming", "sim block"
]

# ---------------------------------------------------------
# Vector store loader (cached) - loads or builds FAISS index
# ---------------------------------------------------------
@st.cache_resource(show_spinner="🔄 Loading and indexing PDFs...")
def load_vector_store():
    """
    Load FAISS index if present; otherwise create it from PDFs.
    Important: keep PDF paths updated; large PDFs will take time to embed.
    """
    embeddings = get_embeddings()

    # If you want to re-index, either delete the faiss_index folder or add a versioning mechanism.
    if os.path.exists("faiss_index/index.faiss"):
        # load previously saved FAISS index (fast)
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        # No index found — build from provided PDFs
        # TODO: adjust pdf_paths to include all relevant data sources
        pdf_paths = [r"C:\Users\Ishtiaq Ahmad\Desktop\zong ai assistant\data\New Microsoft Word Documentaujsshs.pdf"]

        all_pages = []
        for path in pdf_paths:
            # PyPDFLoader splits into page documents (keeps page metadata)
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            # add source filename to metadata so answers can reference where info came from
            for page in pages:
                page.metadata["source"] = os.path.basename(path)
            all_pages.extend(pages)

        # Split long pages into smaller chunks — improves retrieval relevance
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "]
        )

        docs = text_splitter.split_documents(all_pages)

        # Build vector store and save locally for subsequent runs
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local("faiss_index")
        return vector_store

# ------------------------
# Simple intent detection
# ------------------------
def detect_intent(question: str) -> str:
    """
    Lightweight intent classifier:
      - greeting
      - gratitude
      - telecom (mapped to 'medical' in your code; kept same to avoid breaking downstream)
      - about
    NOTE: The name 'medical' is retained from your original code (means telecom in this app).
    """
    q = question.lower().strip()

    greeting_patterns = [r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bwho are you\b"]
    gratitude_patterns = [r"\bthank you\b", r"\bthanks\b", r"\bthx\b"]
    # telecom patterns are broad — tweak as your dataset grows
    telecom_patterns = [
        r"\b(internet|data|weekly|monthly|daily|call|sms|social|whatsapp|youtube|package|bundle|offer|recharge|load|balance|sim|zong|activation|deactivation|subscribe|unsubscribe|code|number check|4g|5g|roaming|international|helpline|customer service|network|signal|slow internet|no signal|coverage|zong app|zong website)\b",
        r"\b(how to subscribe|how to unsubscribe|what is the code|package details|zong offer|stay at home offer|location based offer|complaint|file complaint|network issue|sim block|balance check)\b"
    ]

    def match_any(patterns):
        return any(re.search(p, q) for p in patterns)

    if match_any(greeting_patterns):
        return "greeting"
    elif match_any(gratitude_patterns):
        return "gratitude"
    # heuristic: if it matches telecom or user typed a longer query -> treat as telecom question
    elif match_any(telecom_patterns) or len(q.split()) > 3:
        return "medical"
    return "about"

# ------------------------
# Prompt template for LLM
# ------------------------
zong_prompt = PromptTemplate.from_template('''
You are *Zong AI Assistant*, a professional and friendly virtual assistant trained to help users with Zong Telecom-related queries, including packages, complaints, balance, SIM issues, and service codes.

Respond based on intent:

1. *Greeting* → "Hello! I'm Zong AI Assistant — your go-to guide for Zong packages, offers, and services. How can I assist you today?"

2. *Gratitude* → Respond warmly like GPT. Randomly choose one of the following:
   - "You're very welcome!"
   - "Happy to assist anytime!"
   - "Glad I could help!"
   - "No problem — let me know if there's anything else."
   - "Always here to help with your Zong queries!"

3. *About* → "I'm Zong AI Assistant, built to provide fast and accurate help with Zong Telecom services — from data packages to SIM issues and complaint resolution."

4. *Telecom* → 
   - Start with:
     - "Sure! Here's what I found for you:"
     - "Absolutely! Here's the information you requested:"
   - Then provide accurate telecom-related information.
   - If context is missing or package not found, say:
     "I'm sorry, I couldn't find that information right now. Please try again or provide more details."
   - Add friendly note:
     - "📌 For the most updated offers and services, you can also check the Zong website or My Zong App."
     - "☎️ For urgent support, Zong's helpline is 310."

---
Context:
{context}
---
User Question:
{question}
Answer:
''')

# --------------------------
# LangChain / retrieval setup
# --------------------------
vector_store = load_vector_store()
retriever = vector_store.as_retriever(
    search_type="mmr",  # maximum marginal relevance to increase diversity in results
    search_kwargs={"k": 3, "lambda_mult": 0.5}
)

# LLM configuration — low temperature for deterministic/concise answers
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

# in-memory conversational buffer to maintain short-term context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Build a conversational retrieval chain that uses memory + retriever
qa_with_memory = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": zong_prompt}
)

# -------------------------
# Streamlit UI configuration
# -------------------------
st.set_page_config(page_title="Zong AI Assistant", layout="wide")
st.title("📱 Zong AI Assistant")
st.markdown("Ask me about any Zong package, service code, balance inquiry, or SIM issue. I'm here to help with subscriptions, complaints, and more!")

# -----------------------
# Session state defaults
# -----------------------
# keep a simple chat history list of (role, text) tuples so Streamlit can re-render
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# small flag so we don't repeat the greeting every run
if 'greeted' not in st.session_state:
    st.session_state.greeted = False

# -------------
# Sidebar
# -------------
with st.sidebar:
    def clear_chat():
        st.session_state.chat_history.clear()
        st.session_state.greeted = False

    st.button("🗑️ Clear Chat", on_click=clear_chat)
    st.markdown("---")
    # TODO: add advanced controls here (e.g., model temperature, re-index button, upload PDFs)

# -----------------------
# Render previous messages
# -----------------------
for role, text in st.session_state.chat_history:
    # Streamlit chat_message expects either "user" or "assistant" role
    with st.chat_message(role):
        st.markdown(text)

# -----------------------
# Main user input handler
# -----------------------
if prompt := st.chat_input("Ask anything about Zong services, packages, or SIM issues..."):
    # try to map typos to known terms before intent detection
    corrected_prompt = fuzzy_correct(prompt, all_known_terms)

    # show the raw user message in the chat area
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # nice little spinner while LLM + retriever do their job
        with st.spinner("Zong AI Assistant is thinking..."):
            try:
                # determine intent from the (possibly corrected) prompt
                intent = detect_intent(corrected_prompt)

                # canned responses for trivial intents (fast path)
                responses = {
                    "greeting": "👋 Hello! I'm Zong AI Assistant — your go-to guide for Zong packages, offers, and services. How can I assist you today?",
                    "gratitude": random.choice([
                        "🙏 You're very welcome!",
                        "😊 Happy to assist anytime!",
                        "👍 Glad I could help!",
                        "🫶 No problem — let me know if there's anything else.",
                        "🙌 Always here to help with your Zong queries!"
                    ]),
                    "about": "ℹ️ I'm Zong AI Assistant, built to help you with Zong Telecom services — including internet, calls, SMS bundles, and complaint resolution."
                }

                # If greeting and we haven't greeted already, show the greeting and flip the flag
                if intent == "greeting" and not st.session_state.get("greeted", False):
                    response = responses["greeting"]
                    st.session_state.greeted = True
                elif intent in responses:
                    # for 'gratitude' and 'about' just return canned text
                    response = responses[intent]
                else:
                    # otherwise ask the RAG chain for an answer (uses memory + retriever)
                    result = qa_with_memory({"question": corrected_prompt})
                    # result["answer"] usually contains the assistant text from chain
                    response = result.get("answer", "⚠️ Sorry — I couldn't compose an answer right now. Try again later.")

            except Exception as e:
                # catch-all: avoid exposing raw exceptions to end-user
                # For debugging you can log e or show a more detailed error in dev mode
                response = "⚠️ Sorry, something went wrong while processing your request. Please try again later."

    # persist conversation in session state so it survives reruns
    st.session_state.chat_history.append(("user", prompt))
    st.session_state.chat_history.append(("assistant", response))

    # show the assistant response in the UI
    st.markdown(response)

    # attempt a small JS scroll-to-bottom so conversation behaves like a chat app
    # Streamlit allows this HTML snippet but it's a bit of a hack; keep it if you like the UX.
    st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

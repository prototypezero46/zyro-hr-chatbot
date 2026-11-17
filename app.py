"""Zyro Dynamics HR Help Desk - Streamlit app.

Same pipeline as the Kaggle notebook:
PDFs -> chunks (+ document labels) -> FAISS (meaning) + BM25 (keywords) -> Qwen on Groq -> cleaned answer + sources.
"""
import os
import re
import glob
import time

import streamlit as st

# ---------- 1. Keys and LangSmith tracing (read from Streamlit secrets) ----------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
LANGSMITH_KEY = st.secrets.get("LANGSMITH_API_KEY", "")
if LANGSMITH_KEY:
    for prefix in ["LANGCHAIN", "LANGSMITH"]:
        os.environ[f"{prefix}_API_KEY"] = LANGSMITH_KEY
        os.environ[f"{prefix}_ENDPOINT"] = "https://apac.api.smith.langchain.com"
        os.environ[f"{prefix}_PROJECT"] = "zyro-rag-challenge"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PDF_FOLDER = "policies"

REFUSAL = ("I'm sorry, I can only answer questions about Zyro Dynamics HR policies. "
           "This question is outside the scope of the HR policy documents.")


# ---------- 2. Build the knowledge base once (cached) ----------
def simple_tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


@st.cache_resource(show_spinner="Reading the 11 HR policy PDFs...")
def build_knowledge_base():
    pdf_files = sorted(glob.glob(os.path.join(PDF_FOLDER, "*.pdf")))
    docs = []
    for p in pdf_files:
        docs.extend(PyPDFLoader(p).load())

    chunks = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200).split_documents(docs)

    # Label every chunk with its document name, e.g. "[Work From Home Policy]"
    for c in chunks:
        name = os.path.basename(c.metadata["source"]).replace(".pdf", "")
        name = re.sub(r"^\d+_", "", name).replace("_", " ")
        c.page_content = f"[{name}]\n{c.page_content}"

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    bm25 = BM25Retriever.from_documents(chunks, k=3, preprocess_func=simple_tokens)
    return vectorstore, bm25, len(pdf_files), len(chunks)


def retrieve(question, vectorstore, bm25):
    meaning_chunks = vectorstore.similarity_search(question, k=5)
    keyword_chunks = bm25.invoke(question)
    out = []
    for d in meaning_chunks + keyword_chunks:
        if d not in out:
            out.append(d)
    return out


# ---------- 3. Prompt + LLM chain ----------
@st.cache_resource
def build_chain():
    llm = ChatOpenAI(
        model="qwen/qwen3.8-27b",
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are the HR Help Desk assistant for Zyro Dynamics. "
         "Answer the employee's question using ONLY the policy context provided. "
         "The context is extracted from PDFs, so tables appear as broken lines with each cell on its own line; "
         "read them carefully (e.g. a leave type is followed by its entitlement, carry forward and encashment values). "
         "WORDING: reuse the exact words, terms and numbers from the policy text as closely as possible. "
         "Copy the relevant policy sentence(s) nearly word for word, only lightly adjusting them so they "
         "answer the question. Each answer must be self-contained: include any preceding condition the copied "
         "sentence depends on, and never start with linking words like 'Thereafter' or 'However'. "
         "Keep the policy's own names (e.g. 'Earned Leave', 'Internal Committee', "
         "'L5 and above', 'Rs. 5,00,000') and do not paraphrase or summarise in your own words. "
         "Be complete but concise (1-3 sentences): include all the key figures, conditions and eligibility "
         "the policy states for this question. Plain text, no markdown or bullets. "
         "Never add details that are not written in the context. "
         "Questions about 'my grade', 'my ESOP', 'my leave' etc. are policy questions: answer with the general policy rule. "
         "Any question about Zyro Dynamics employee policies (leave, pay, salary bands, benefits, travel, hotel and daily "
         "allowances, conduct, gifts, IT and passwords, POSH, performance, WFH, onboarding, resignation) MUST be answered "
         "from the context, even if the question says 'my' or the relevant table is broken across lines. Never refuse these. "
         "Reply exactly \"{refusal}\" ONLY when the question is about another company, Zyro's products, Zyro's revenue or "
         "finances, applying for a job as an outside candidate, or general knowledge."),
        ("human", "Policy context:\n{context}\n\nQuestion: {question}"),
    ]).partial(refusal=REFUSAL)
    return prompt | llm | StrOutputParser()


def invoke_with_retry(chain, inputs, tries=6):
    last_err = None
    for _ in range(tries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            last_err = e
            time.sleep(10)
    raise last_err


def ask_bot(question, vectorstore, bm25, chain):
    found = retrieve(question, vectorstore, bm25)
    context = "\n".join(d.page_content for d in found)
    answer = invoke_with_retry(chain, {"context": context, "question": question})
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S)
    answer = re.sub(r"^\W*(Thereafter|However|Additionally|Furthermore|Further),\s*", "", answer, flags=re.I)
    answer = re.sub(r"[*#`]", "", answer).replace("\n", " ").strip()
    refused = "outside the scope" in answer.lower() or "can only answer" in answer.lower()
    if refused:
        return REFUSAL, []          # no sources for a refusal
    return answer, found


# ---------- 4. The chat interface ----------
st.set_page_config(page_title="Zyro HR Help Desk", page_icon="💼")
st.title("Zyro Dynamics HR Help Desk")
st.caption("Ask about leave, pay, WFH, travel, conduct, POSH and other HR policies. "
           "Every answer shows the policy page it came from.")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Add it in the app's Settings > Secrets.")
    st.stop()

vectorstore, bm25, n_pdfs, n_chunks = build_knowledge_base()
chain = build_chain()

with st.sidebar:
    st.header("About")
    st.write(f"Knowledge base: **{n_pdfs} policy PDFs**, **{n_chunks} chunks**")
    st.write("Search: FAISS (meaning, k=5) + BM25 (keywords, k=3)")
    st.write("Model: Qwen 3.8 27B on Groq")
    st.subheader("Try asking")
    st.write("- How many casual leaves do I get per year?")
    st.write("- What's my hotel allowance for international travel?")
    st.write("- Who won the last IPL?")
    if st.button("Clear chat"):
        st.session_state.messages = []


def show_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)} policy chunks)"):
        for d in sources:
            file = os.path.basename(d.metadata.get("source", ""))
            page = d.metadata.get("page", 0) + 1
            text = d.page_content.split("\n", 1)[-1]
            st.markdown(f"**{file} — page {page}**")
            st.caption(text[:400] + ("..." if len(text) > 400 else ""))


if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        show_sources(m.get("sources", []))

question = st.chat_input("Ask an HR policy question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the policies..."):
            try:
                answer, sources = ask_bot(question, vectorstore, bm25, chain)
            except Exception as e:
                answer, sources = f"Sorry, the model is busy right now. Please try again. ({type(e).__name__})", []
        st.write(answer)
        show_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma, Milvus, MongoDBAtlasVectorSearch, ElasticVectorSearch
from langchain_community.vectorstores import FAISS

import os
from dotenv import load_dotenv

load_dotenv()


# ---- Streamlit UI ---- #
st.set_page_config(layout="wide")
st.title("My Local Chatbot")

st.sidebar.header("Settings")
MODEL = st.sidebar.selectbox("Choose a Model", ["llama3.2", "deepseek-r1:1.5b"], index=0)
MAX_HISTORY = st.sidebar.number_input("Max History", 1, 10, 2)
CONTEXT_SIZE = st.sidebar.number_input("Context Size", 1024, 16384, 8192, step=1024)

# ---- Session State Setup ---- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state or st.session_state.get("prev_context_size") != CONTEXT_SIZE:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.session_state.prev_context_size = CONTEXT_SIZE


# ---- LangChain Components ---- #
llm = ChatOllama(model=MODEL, streaming=True)
embeddings = OllamaEmbeddings(model="llama3.2")

# Step 2: Load the existing FAISS index
faiss_index_path = "TCSNSEData16"  

vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

# Step 3: Use the FAISS index as a retriever
retriever = vector_store.as_retriever()

# Optional: Create a Retrieval-based QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 5. Create a custom prompt template
prompt_template = """
Role: You are a helpful AI assistant that answers questions based on the provided Financial PDF document.
Objective: The objective is to provide answers based on the provided PDF document and generate insight. Use only the context provided to answer the question. If you don't know the answer or
can't find it in the context, say so.
Guidelines: If the query contains any abusive or inappropriate content, respond with "I cannot assist with that."

Question: {question}

Answer: Let me help you with that based on the PDF content."""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["question"]
)


# ---- Display Chat History ---- #
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---- Trim Chat Memory ---- #
def trim_memory():
    while len(st.session_state.chat_history) > MAX_HISTORY * 2:
        st.session_state.chat_history.pop(0)  # Remove oldest messages


# ---- Handle User Input ---- #
if prompt := st.chat_input("Say something"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    trim_memory()

    with st.chat_message("assistant"):
        response_container = st.empty()

        # Retrieve relevant documents
        #retrieved_docs = retriever.get_relevant_documents(prompt)

        
        retrieved_docs = retriever.get_relevant_documents(prompt)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Fill the prompt template with context and user question
        filled_prompt = prompt_template.format(context=context, question=prompt)

        full_response = (
            "No relevant documents found." if not retrieved_docs
            else qa({"query": filled_prompt}).get("result", "No response generated.")
        )

        response_container.markdown(full_response)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

        trim_memory()
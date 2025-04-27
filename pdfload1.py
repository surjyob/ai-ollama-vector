from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# 1. Create the model
llm = Ollama(model='llama3')
# embeddings = OllamaEmbeddings(model='znbang/bge:small-en-v1.5-f32')
embeddings = OllamaEmbeddings(
    model="llama3",
)

from langchain_community.vectorstores import DocArrayInMemorySearch

file_path = "5-mb-example-file.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

pages = loader.load_and_split()
store = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
retriever = store.as_retriever()

print(f"Loaded {len(pages)} pages from {file_path}")
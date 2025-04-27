import getpass
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader, CSVLoader, JSONLoader, PyPDFLoader

import os
from dotenv import load_dotenv

load_dotenv()

urls = ["",""]


def WebsiteLoader(urls):
    loader = WebBaseLoader(urls)
    return loader.load()

def CSVFileLoader(file_paths):
    docs = []
    for file_path in file_paths:
        loader = CSVLoader(file_path=file_path)  # Load each CSV file individually
        docs.extend(loader.load())  # Append loaded data to the list
    return docs



def PDFLoader(pdf_files):
    docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(file_path=pdf_file)  # Load each PDF file individually
        docs.extend(loader.load())  # Append loaded data to the list
    return docs



# url_list = WebsiteLoader("https://vincent.codes.finance/posts/documents-llm")

url_list = PDFLoader(["sample-report.pdf"])
# url_list = PDFLoader("./5-mb-example-file.pdf") 

llm = ChatOllama(model="llama3.2" )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)


# Combine all loaded documents
all_documents = url_list

splited_documents = text_splitter.split_documents(all_documents)
print(splited_documents)


embeddings  = OllamaEmbeddings(
  model='llama3',
)
persist_directory = "./chromapdf_db"

vectorstore = Chroma.from_documents(
    documents=splited_documents,  # Ensure splited_documents contains valid data
    embedding=embeddings,
    persist_directory=persist_directory
)

# Step 2: Persist the database
vectorstore.persist()  # ✅ Saves data to disk
print("✅ Data successfully stored in ChromaDB!")


from langchain.chains import RetrievalQA

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})

qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
query = "query?"
result = qa({"query": query})

print(result['result'])
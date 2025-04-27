from langchain_community.document_loaders.pdf import PyPDFLoader

file_path = "C:\Python\SetUp\Scripts\Langtest\LangChroma\hacking_prices.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# Prompt
from langchain_core.prompts import PromptTemplate

prompt_template = """Write a long summary of the following document. 
Only include information that is part of the document. 
Do not include your own opinion or analysis.

Document:
"{document}"
Summary:"""
prompt = PromptTemplate.from_template(prompt_template)

# Define LLM Chain

# from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain


from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3",
    temperature=0,
    # other params...
)
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Create full chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain, document_variable_name="document"
)

# result = stuff_chain.invoke(docs)

# Invoke with limited pages
result = stuff_chain.invoke(docs[:-3])
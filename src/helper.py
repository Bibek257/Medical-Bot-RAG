from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document


# extract text from pdf
def load_pdf(data):
    loader=DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    
    documents=loader.load()
    return documents


# filter to minima
def filter_to_minima(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document(page_content=doc.page_content,metadata={"source":src})
        )
        
    return minimal_docs

# split text to chunks
def text_spliter(minimal_docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len
    )
    text=text_splitter.split_documents(minimal_docs)
    return text


# download embedding from huggingface
from langchain.embeddings import HuggingFaceEmbeddings
def download_embedding():
    """
    Download the sentence transformer model
    """
    model_name="BAAI/bge-small-en-v1.5"
    embedding=HuggingFaceEmbeddings(model_name=model_name)
    return embedding


from dotenv import load_dotenv
import os
from src.helper import text_spliter,filter_to_minima,load_pdf,download_embedding
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

pinecone_api_key=os.getenv("PINECONE_API_KEY")
open_ai_api_key=os.getenv("OPEN_AI_API_KEY")

os.environ["OPENAI_API_KEY"] = open_ai_api_key
os.environ["PINECONE_API_KEY"] = pinecone_api_key

extracted_text=load_pdf("data")
minimal_docs=filter_to_minima(extracted_text)
text_chunk=text_spliter(minimal_docs)
embedding=download_embedding()

pinecone_api_key=pinecone_api_key
pc=Pinecone(
    api_key=pinecone_api_key
)

index_name="medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1"))
    
docsearch=PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embedding,
    index_name=index_name
)
       
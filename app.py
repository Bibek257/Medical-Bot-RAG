from flask import Flask, render_template, request
from src.helper import download_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
open_ai_api_key = os.getenv("OPEN_AI_API_KEY")   

os.environ["OPENAI_API_KEY"] = open_ai_api_key
os.environ["PINECONE_API_KEY"] = pinecone_api_key

embedding = download_embedding()
index_name = "medical-chatbot"

# Order fixed
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatmodel = ChatOpenAI(model_name="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_ans_chain = create_stuff_documents_chain(chatmodel, prompt)
rag_chain = create_retrieval_chain(retriever, question_ans_chain)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["input"]
    response = rag_chain.invoke({"input": user_input})
    return response.get("answer", str(response))

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

import os
import streamlit as st
import tempfile
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from pinecone import ServerlessSpec, PodSpec
from streamlit import session_state
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA , RetrievalQAWithSourcesChain
import time 


PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
PINECONE_ENVIRONMENT = st.secrets['PINECONE_ENVIRONMENT']
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']

index_name = "goldragai"
embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=OPENAI_API_KEY)
use_serverless = True 


vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)

query = f"how to win Olympics?"
qu = vectorstore.similarity_search(query, k=1)

llm = ChatOpenAI(  
    openai_api_key=OPENAI_API_KEY,  
    model_name='gpt-3.5-turbo',  
    temperature=0.0  
) 

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(  
    llm=llm,  
    chain_type="stuff",  
    retriever=vectorstore.as_retriever()  
)

response=qa_with_sources(query)

print(qu[0].page_content)
print("=====================================")
print(response)


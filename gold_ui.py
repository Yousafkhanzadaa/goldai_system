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
import time 


PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
PINECONE_ENVIRONMENT = st.secrets['PINECONE_ENVIRONMENT']
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
correct_password = st.secrets['PASSWORD']

index_name = "goldragai"
embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=PINECONE_API_KEY)
use_serverless = True 

pc = Pinecone(api_key=PINECONE_API_KEY)



def setupEnvironment():
  # configure client  
  

  spec = ServerlessSpec(cloud='aws', region='us-east-1')  
  # check for and delete index if already exists  
  if index_name not in pc.list_indexes().names():  
      # pc.delete_index(index_name)  
      # create a new index  
      pc.create_index(  
          index_name,  
          dimension=1536,  # dimensionality of text-embedding-ada-002  
          metric='cosine',  
          spec=spec  
      )  
  # wait for index to be initialized  
  while not pc.describe_index(index_name).status['ready']:  
      time.sleep(1)
  
  index = pc.Index(index_name)  
  
  global vectorstore 
  vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
  
  return index.describe_index_stats() 


def vectorizeFile(up_files):

  # path to an example text file
  for uf in up_files:
    with st.spinner("Indexing documents... this might take a while⏳"):
      with tempfile.TemporaryDirectory() as tmpdir:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_content = uploaded_file.read()
            st.write("Filename: ", file_name)
            with open(os.path.join(tmpdir, file_name), "wb") as file:
                file.write(file_content)
                
        loader = DirectoryLoader(tmpdir, glob="**/*.pdf", loader_cls=PyMuPDFLoader) 
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        if index_name in pc.list_indexes().names(): 
          vectorstore.add_documents(docs)
        else:
          vectorstore_from_docs = PineconeVectorStore.from_documents(
              docs,
              index_name=index_name,
              embedding=embeddings
          )
        st.success("Ingested File!")
    

def render_header():
   st.title('GOLD AI')
   st.markdown(("### LLM Assisted Custom Knowledgebase "+
                        "\n\n"+
                        "GOLD AI is a Python application that allows you to upload a PDF as vector embedding to pinecone"+
                        "\n\n"+
                        "#### How it works "+
                        "\n\n"+
                        "Upload personal docs"+
                        "\n\n"+
                        "This tool is powered by [OpenAI](https://openai.com), "+
                        "[LangChain](<https://langchain.com/>), and [OpenAI](<https://openai.com>) and made by "
                    ))
   
def clear_submit():
    st.session_state["submit"] = False   

def upload_files():
    uploaded_files = st.file_uploader(
        "Upload multiple files",
        type="pdf",
        help="docs, and txt files are still in beta.",
        accept_multiple_files=True,
        on_change=clear_submit
    )
    
    if uploaded_files is None:
        st.info("Please upload a file of type: " + ", ".join(["pdf"]))
    return uploaded_files
   
if __name__ == '__main__':
  if 'login_successful' not in st.session_state:
    st.session_state.login_successful = False

  if not st.session_state.login_successful:
      password = st.text_input("Enter a password",
                              help="Please enter your password; it's case sensitive",
                              type="password")
      if password == correct_password:
          st.session_state.login_successful = True
          st.rerun()
      elif password:
          st.error("The password you entered is incorrect. Please try again.")
      
  if st.session_state.login_successful:
    render_header()
    
    setupEnvironment()
    
    uploaded_files = upload_files()
    
    if uploaded_files:
      vectorizeFile(uploaded_files)
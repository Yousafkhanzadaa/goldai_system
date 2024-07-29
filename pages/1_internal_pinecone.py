import os
import streamlit as st
import tempfile
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from pinecone import ServerlessSpec, PodSpec
from streamlit import session_state
from langchain.chains import RetrievalQA , RetrievalQAWithSourcesChain
import time 

##########################
#--- Pinecone, Langchain, Openai 

PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
PINECONE_ENVIRONMENT = st.secrets['PINECONE_ENVIRONMENT']
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
correct_password = st.secrets['PASSWORD']

index_name = "gold-internal-data"
embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=OPENAI_API_KEY)
use_serverless = True 

pc = Pinecone(api_key=PINECONE_API_KEY)
##########################

##########################
# -- UI Streamlit code
st.set_page_config(layout="wide")
leftCol, rightCol = st.columns(2)
##########################



##########################
# -- Chat UI
def showChatUi():
  with rightCol:
    st.title("Internal Chat With GOLD AI")

    # Set OpenAI API key from Streamlit secrets
    # client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"]) 


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
      
        # query = f"please summery of given text."
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
          with st.spinner("Bot is Typing...."):
            # qu = vectorstore.similarity_search(query, k=1)
            # st.write(qu[0].page_content)
            llm = ChatOpenAI(  
                openai_api_key=OPENAI_API_KEY,  
                model_name='gpt-4o-mini',  
                temperature=0.5  
            ) 

            qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(  
                llm=llm,  
                chain_type="stuff",  
                retriever=vectorstore.as_retriever()  
            )

            response = qa_with_sources(prompt)
            responsestream = st.write(response.get('answer'))
          st.session_state.messages.append({"role": "assistant", "content": response.get('answer')})
            # st.experimental_rerun()
  # ------- End Chat Ui
##########################

def setupEnvironment():

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
  
  
  # configure client  
  global vectorstore 
  vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY, namespace="np-gold1")
  # wait for index to be initialized  
  while not pc.describe_index(index_name).status['ready']:  
      time.sleep(1)
  
  index = pc.Index(index_name)  
  
  return index.describe_index_stats() 


processed_files = dict()

def vectorizeFile(up_files):

  with leftCol:
    with st.spinner("Indexing documents... this might take a while⏳"):
      for uploaded_file in up_files:
        file_name = uploaded_file.name
  
        # Check if file_name is in processed_files
        if file_name in st.session_state.processed_files:
          st.error(f"{file_name} has already been processed.")
          continue

        # Process uploaded_file here
        with tempfile.TemporaryDirectory() as tmpdir:
          file_content = uploaded_file.read()
          with open(os.path.join(tmpdir, file_name), "wb") as file:
              file.write(file_content)
              
          loader = DirectoryLoader(tmpdir, glob="**/*.pdf", loader_cls=PyMuPDFLoader) 
          documents = loader.load()
          text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
          docs = text_splitter.split_documents(documents)

          if index_name in pc.list_indexes().names(): 
            vectorstore.add_documents(docs)
            st.success(f"Ingested File: {file_name}!")
            st.session_state.processed_files.append(file_name)
          else:
            st.error("Index does not exist. Please create the index before adding documents.") 
        


def render_header():
  with leftCol:
    st.title('Internal GOLD AI Embedding')
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
    
  with leftCol:
    uploaded_files = st.file_uploader(
        "Upload multiple files",
        type="pdf",
        help="docs, and txt files are still in beta.",
        accept_multiple_files=True,
        key=st.session_state.uploader_key,
        on_change=clear_submit
    )
    
    for file in st.session_state.processed_files:
      st.success(f"Processed File: {file}")
    
    if uploaded_files is None:
        st.info("Please upload a file of type: " + ", ".join(["pdf"]))
    return uploaded_files
   
if __name__ == '__main__':
  if 'login_successful' not in st.session_state:
    st.session_state.login_successful = False
    
  if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
    
  if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

  if not st.session_state.login_successful:
      username = st.text_input("User Name",
                              placeholder="Gold Ai",
                              value="Gold AI",
                              disabled=True,
                              type="default")
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
    
    showChatUi()
    
    uploaded_files = upload_files()
    
    if uploaded_files:
      vectorizeFile(uploaded_files)
      st.session_state.uploader_key += 1
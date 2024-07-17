import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter


PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

index_name = "goldai"
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# path to an example text file
loader = PyMuPDFLoader("data/white_paper.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

vectorstore_from_docs = PineconeVectorStore.from_documents(
    docs,
    index_name=index_name,
    embedding=embeddings
)

# vectorstore.add_documents(docs)

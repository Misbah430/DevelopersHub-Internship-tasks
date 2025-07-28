from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

def load_vector_store(file_path="data/ml_tasks.pdf"):
    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split text into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Use local embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

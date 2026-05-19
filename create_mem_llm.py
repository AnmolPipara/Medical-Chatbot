#step 1: Loading raw PDF
#step 2 : Creating chunks
#step 3 : Creating embeddings
#step 4 : storing in vector database (Faiss)

from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATAPATH = "data/"
def load_pdf(file_path):
    loader = DirectoryLoader(file_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

doc = load_pdf(file_path=DATAPATH)

# print(f"Total number of documents: {len(doc)}")

def create_chunks(extracted_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(extracted_docs)
    return chunks 

textchunks = create_chunks(extracted_docs=doc)
# print(f"Total number of chunks: {len(textchunks)}")


def get_embeddings_model(chunks):
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings_model

embeddings_model = get_embeddings_model(chunks=textchunks)

DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(textchunks, embeddings_model)
db.save_local(DB_FAISS_PATH)
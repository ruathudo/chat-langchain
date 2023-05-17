"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader, UnstructuredHTMLLoader, UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


urls = [
    "https://thefcompany.com/",
    "https://thefcompany.com/marketing-as-a-service",
    "https://thefcompany.com/cases",
    "https://thefcompany.com/about",
    "https://thefcompany.com/careers",
    "https://thefcompany.com/events/b2b-demand-generation-how-to-turn-marketing-into-a-revenue-driver",
    "https://thefcompany.com/blog"
]

def ingest_docs():
    """Get documents from web pages."""
    loader = UnstructuredURLLoader(urls=urls)
    # loader = ReadTheDocsLoader("langchain.readthedocs.io/en/latest/")
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()

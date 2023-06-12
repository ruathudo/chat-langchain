"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader, UnstructuredHTMLLoader, UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import pandas as pd



def ingest_docs(urls):
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
    df = pd.read_csv('konecranes.csv')
    urls = df.iloc[:50, 0].tolist()
    #url2 = df.iloc[69:96, 0].tolist()

    # urls = url1 + url2

    ingest_docs(urls)

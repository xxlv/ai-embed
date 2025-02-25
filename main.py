import os
import glob
import requests
from typing import List, Dict, Optional
import chromadb
import markdown
from bs4 import BeautifulSoup
from chromadb.errors import InvalidCollectionException
from dotenv import load_dotenv
import click

load_dotenv()


class Config:
    """Configuration settings for the application."""
    OLLAMA_API: str = os.getenv(
        "OLLAMA_API", "http://localhost:11434/api/embeddings")
    PERSIST_DIRECTORY: str = os.getenv("PERSIST_DIRECTORY")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "nomic-embed-text")
    MD_FILES_PATH: str = os.getenv("MD_FILES_PATH")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "markdown_documents")
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", 512))

    @staticmethod
    def validate():
        required = ["PERSIST_DIRECTORY", "MD_FILES_PATH"]
        missing = [key for key in required if not os.getenv(key)]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {missing}")


class OllamaEmbeddingFunction:
    def __init__(self, model_name: str = Config.MODEL_NAME, api_url: str = Config.OLLAMA_API):
        self.model_name = model_name
        self.api_url = api_url

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in input:
            payload = {"model": self.model_name, "prompt": text}
            response = requests.post(self.api_url, json=payload)
            if response.status_code == 200:
                embeddings.append(response.json()["embedding"])
            else:
                raise Exception(f"Ollama API error: {response.text}")
        return embeddings


def chunk_markdown(text: str, max_chunk_size: int = Config.MAX_CHUNK_SIZE) -> List[str]:
    html = markdown.markdown(text)
    soup = BeautifulSoup(html, features="html.parser")
    plain_text = soup.get_text()
    sentences = [s.strip() for s in plain_text.replace(
        '\n', ' ').split('.') if s.strip()]
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = f"{current_chunk}. {sentence}" if current_chunk else sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def load_markdown_file(file_path: str) -> Optional[str]:
    for encoding in ['utf-8', 'gbk']:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    print(f"Failed to decode file {file_path} with available encodings")
    return None


def process_markdown_files(client: chromadb.Client, collection: chromadb.Collection) -> tuple[int, int]:
    md_files = glob.glob(Config.MD_FILES_PATH, recursive=True)
    print(f"Found {len(md_files)} Markdown files")
    total_chunks = 0
    for file_idx, file_path in enumerate(md_files, 1):
        file_name = os.path.basename(file_path)
        print(f"Processing file {file_idx}/{len(md_files)}: {file_name}")
        md_content = load_markdown_file(file_path)
        if not md_content:
            continue
        chunks = chunk_markdown(md_content)
        if not chunks:
            print(f"  Skipping empty file: {file_path}")
            continue
        total_chunks += len(chunks)
        documents, metadatas, ids = [], [], []
        for chunk_idx, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "source": file_name,
                "full_path": file_path,
                "chunk": chunk_idx,
                "total_chunks": len(chunks)
            })
            ids.append(f"doc_{file_idx}_{chunk_idx}")
        try:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            print(f"  Added {len(chunks)} chunks from {file_name}")
        except Exception as e:
            print(f"  Error adding chunks from {file_name}: {e}")
    return len(md_files), total_chunks


def query_collection(collection: chromadb.Collection, query: str) -> None:
    results = collection.query(query_texts=[query], n_results=3)
    print("\nSearch Results:")
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
        print(f"\nResult {i}:")
        print(f"Source: {metadata['source']}")
        print(f"Path: {metadata['full_path']}")
        print(f"Chunk: {metadata['chunk'] + 1}/{metadata['total_chunks']}")
        print(f"Content: {doc[:150]}...")


def get_collection(client: chromadb.Client) -> chromadb.Collection:
    embedding_function = OllamaEmbeddingFunction()
    try:
        collection = client.get_collection(
            name=Config.COLLECTION_NAME, embedding_function=embedding_function
        )
        print(f"Using existing collection: {Config.COLLECTION_NAME}")
    except InvalidCollectionException:
        collection = client.create_collection(
            name=Config.COLLECTION_NAME, embedding_function=embedding_function
        )
        print(f"Created new collection: {Config.COLLECTION_NAME}")
    return collection


@click.group()
def cli():
    """MarkdownEmbedder CLI for interacting with ChromaDB."""
    Config.validate()
    os.makedirs(Config.PERSIST_DIRECTORY, exist_ok=True)


@cli.command()
def process():
    """Process Markdown files and store them in ChromaDB."""
    client = chromadb.PersistentClient(path=Config.PERSIST_DIRECTORY)
    collection = get_collection(client)
    total_docs, total_chunks = process_markdown_files(client, collection)
    print(
        f"Completed processing. Total documents: {total_docs}, Total chunks: {total_chunks}")


@cli.command()
@click.argument('query')
def query(query: str):
    """Query the ChromaDB collection with a search string."""
    client = chromadb.PersistentClient(path=Config.PERSIST_DIRECTORY)
    collection = get_collection(client)
    print(f"Querying collection with: '{query}'")
    query_collection(collection, query)


if __name__ == "__main__":
    cli()

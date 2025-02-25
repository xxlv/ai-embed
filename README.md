# MarkdownEmbedder

MarkdownEmbedder is a Python tool that processes Markdown files, chunks them into smaller pieces, embeds them using Ollama's embedding model, and stores them in a ChromaDB vector database for efficient querying. This project is ideal for anyone looking to build a searchable knowledge base from Markdown notes or documentation.

## Features

- **Markdown Chunking**: Splits Markdown files into manageable chunks based on sentence boundaries.
- **Custom Embeddings**: Uses Ollama's `nomic-embed-text` model (or any specified model) to generate embeddings.
- **Persistent Storage**: Stores embeddings and metadata in ChromaDB with persistent storage.
- **Querying**: Allows searching the embedded documents with a simple query interface.
- **Configurable**: Settings like file paths and API endpoints are configurable via a `.env` file.

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) running locally with the `nomic-embed-text` model installed.
- A directory containing Markdown files to process.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/xxlv/ai-embed.git
   cd ai-embed
   ```

2. **Install Dependencies**:
   Install the required Python packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Ollama**:

   - Install Ollama following the instructions at [ollama.ai](https://ollama.ai/).
   - Start the Ollama server:
     ```bash
     ollama serve
     ```
   - Ensure the `nomic-embed-text` model is available (pull it if needed):
     ```bash
     ollama pull nomic-embed-text
     ```

4. **Configure the Environment**:
   Create a `.env` file in the project root with the following content:

   ```
   PERSIST_DIRECTORY=/path/to/chroma_db
   MD_FILES_PATH=/path/to/markdown/files/**/*.md
   ```

   - Replace `/path/to/chroma_db` with where you want to store the ChromaDB data (e.g., `./chroma_db`).
   - Replace `/path/to/markdown/files/**/*.md` with the path to your Markdown files (e.g., `./notes/**/*.md`).

   Optionally, you can customize these additional settings:

   ```
   OLLAMA_API=http://localhost:11434/api/embeddings
   MODEL_NAME=nomic-embed-text
   COLLECTION_NAME=markdown_documents
   MAX_CHUNK_SIZE=512
   ```

## Usage

1. **Run the Script**:

   ```bash
   python main.py
   ```

   - The script will process all Markdown files in `MD_FILES_PATH`, embed them, and store them in `PERSIST_DIRECTORY`.
   - After processing, it prompts for a query to search the embedded documents.

2. **Example Output**:
   ```
   Created new collection: markdown_documents
   Found 5 Markdown files
   Processing file 1/5: note1.md
     Added 3 chunks from note1.md
   ...
   Completed processing. Total documents: 5, Total chunks: 15
   You can now query your documents!
   Enter a query to test (or press Enter to skip): What is Python?
   ```



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ChromaDB](https://github.com/chroma-core/chroma) for the vector database.
- [Ollama](https://ollama.ai/) for the embedding model.
- Built with Python and love for open-source knowledge sharing.

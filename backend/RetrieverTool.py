import os
from PyPDF2 import PdfReader
from smolagents import Tool # Ensure smolagents provides 'Tool'
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys # For printing errors

class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search (BM25) to retrieve relevant parts of research papers (PDFs) related to credit card defaults from a specified directory."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query. Should be descriptive, like a statement about the information needed.",
        }
    }
    output_type = "string"

    def __init__(self, pdf_directory, k_results=3, **kwargs):
        super().__init__(**kwargs)
        self.pdf_directory = pdf_directory
        self.k = k_results # Number of documents to retrieve
        self.retriever = self._initialize_retriever()

    def _initialize_retriever(self):
        print(f"DEBUG [RetrieverTool]: Initializing retriever for directory: {self.pdf_directory}")
        if not os.path.isdir(self.pdf_directory):
             print(f"WARN [RetrieverTool]: PDF directory '{self.pdf_directory}' does not exist or is not a directory.", file=sys.stderr)
             # Return a dummy retriever or handle appropriately? For now, continue, processing will yield empty docs.
             # return None # Or raise error?
        docs = self.process_pdfs(self.pdf_directory)
        if not docs:
             print(f"WARN [RetrieverTool]: No documents processed from '{self.pdf_directory}'. Retriever will be empty.", file=sys.stderr)
             # Create an empty retriever or handle? Langchain might handle empty docs list.
             # Return an empty retriever to avoid errors later when invoke is called.
             # Note: Depending on langchain version, how BM25 handles empty lists might vary.
             try:
                 # Attempt to create from empty texts; might require specific handling based on version
                 return BM25Retriever.from_texts([], k=self.k)
             except Exception as init_err:
                 print(f"WARN [RetrieverTool]: Could not init BM25Retriever with empty docs: {init_err}. Returning None.", file=sys.stderr)
                 return None # Indicate failure clearly
        print(f"DEBUG [RetrieverTool]: Processed {len(docs)} document chunks.")
        try:
            return BM25Retriever.from_documents(docs, k=self.k)
        except Exception as e:
             print(f"ERROR [RetrieverTool]: Failed to initialize BM25Retriever: {e}", file=sys.stderr)
             # Fallback or raise error
             return None # Indicate failure

    def process_pdfs(self, pdf_directory):
        processed_docs = []
        if not os.path.isdir(pdf_directory): # Check again just in case
             return processed_docs

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        for filename in os.listdir(pdf_directory):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(pdf_directory, filename)
                print(f"DEBUG [RetrieverTool]: Processing PDF: {filename}")
                content = self.extract_content(file_path)
                if content:
                    chunks = text_splitter.split_text(content)
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={"source": filename, "chunk": i+1} # Start chunk index at 1 for readability
                        )
                        processed_docs.append(doc)
                else:
                     print(f"WARN [RetrieverTool]: No content extracted from {filename}", file=sys.stderr)

        return processed_docs

    def extract_content(self, pdf_path):
        text = ""
        try:
            # Use strict=False to potentially handle minor PDF issues
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file, strict=False)
                # Check if encrypted
                if pdf_reader.is_encrypted:
                    try:
                        pdf_reader.decrypt('') # Try decrypting with empty password
                    except Exception as decrypt_err:
                        # More specific check for password-protected files if possible
                        if "read" in str(decrypt_err).lower(): # Example check, might need adjustment
                             print(f"WARN [RetrieverTool]: PDF {os.path.basename(pdf_path)} is password-protected. Skipping.", file=sys.stderr)
                        else:
                            print(f"WARN [RetrieverTool]: Could not decrypt {os.path.basename(pdf_path)}: {decrypt_err}. Skipping.", file=sys.stderr)
                        return ""

                # Check for linearization dictionary issues explicitly if needed
                # if pdf_reader.is_linearized:
                #    print(f"DEBUG [RetrieverTool]: PDF {os.path.basename(pdf_path)} is linearized.")

                num_pages = len(pdf_reader.pages)
                print(f"DEBUG [RetrieverTool]: Reading {num_pages} pages from {os.path.basename(pdf_path)}")
                for i, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                             text += page_text + "\n"
                        # else:
                        #     print(f"DEBUG [RetrieverTool]: No text on page {i+1} of {os.path.basename(pdf_path)}")
                    except Exception as page_err:
                         print(f"WARN [RetrieverTool]: Error extracting text from page {i+1} of {os.path.basename(pdf_path)}: {page_err}", file=sys.stderr)

            return self.clean_text(text) if text else ""
        except FileNotFoundError:
             print(f"ERROR [RetrieverTool]: File not found: {pdf_path}", file=sys.stderr)
             return ""
        except Exception as e:
            # Catch specific PyPDF2 errors if possible, e.g., PdfReadError
            print(f"ERROR [RetrieverTool]: Failed to read or extract content from {os.path.basename(pdf_path)}: {str(e)}", file=sys.stderr)
            return ""

    def clean_text(self, text):
        # Basic cleaning
        text = text.replace('\x00', '')  # Remove null bytes
        text = ' '.join(text.split())  # Remove extra whitespace, newlines handled by splitter
        return text

    def forward(self, query: str) -> str:
        if self.retriever is None:
            return "Error: Retriever tool was not initialized successfully (check PDF directory and file processing)."
        if not isinstance(query, str) or not query.strip():
            return "Error: Search query must be a non-empty string."

        print(f"DEBUG [RetrieverTool]: Performing retrieval for query: '{query}'")
        try:
            docs = self.retriever.invoke(query)
            if not docs:
                return "No relevant documents found for your query in the provided PDF directory."

            # Format results
            result_string = "Retrieved documents:\n" + "".join(
                [
                    f"\n\n===== Document: {doc.metadata.get('source', 'Unknown Source')} (Chunk {doc.metadata.get('chunk', '?')}) =====\n{doc.page_content}"
                    for doc in docs
                ]
            )
            return result_string
        except Exception as e:
            print(f"ERROR [RetrieverTool]: Error during BM25 retrieval: {e}", file=sys.stderr)
            return f"Error during search: {e}"
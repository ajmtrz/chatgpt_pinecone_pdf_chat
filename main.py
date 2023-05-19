import os
import re
import threading
import argparse
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from typing import List, Tuple
from urllib.parse import urlparse, urljoin

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import PyPDFDirectoryLoader, SeleniumURLLoader
from langchain.schema import Document
from langchain.vectorstores import Pinecone
import pinecone


def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def sanitize_url(url: str) -> str:
    parsed_url = urlparse(url)
    reconstructed_url = f"{parsed_url.path}{parsed_url.params}?{parsed_url.query}"
    return urljoin(url, reconstructed_url)


def read_documents(path: str) -> List[Document]:
    loader = PyPDFDirectoryLoader(path)
    return loader.load()


def read_urls(path: str) -> List[Document]:
    urls = []
    with open(path, 'r') as archivo:
        for linea in archivo:
            url = linea.strip()
            if is_valid_url(sanitize_url(url)):
                urls.append(url)
    loader = SeleniumURLLoader(urls=urls)
    return loader.load()


def clean_documents(documents: List[Document]) -> None:
    rm_word = "Copyright"
    rm_char = "\t\n"
    rm_table = str.maketrans("", "", rm_char)
    for document in documents:
        document.page_content = re.sub(fr"\b{rm_word}\b", "", document.page_content, flags=re.IGNORECASE)
        document.page_content = document.page_content.translate(rm_table)
        document.page_content = re.sub(r" {2,}", " ", document.page_content)


def get_llm_and_embeddings(model_type: str, model_temp: float) -> Tuple[ChatOpenAI, OpenAIEmbeddings]:
    llm = ChatOpenAI(model_name=model_type, temperature=model_temp)
    embeddings = OpenAIEmbeddings()
    return llm, embeddings


def get_vectorstore(pinecone_api_key: str, pinecone_env: str, documents: List[Document], embeddings: OpenAIEmbeddings) -> Pinecone:
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    index_name = "langchain"
    dimension = 1536
    metric = "cosine"
    pod_type = "p1"
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)
    pinecone.create_index(index_name, dimension=dimension, metric=metric, pod_type=pod_type)
    return Pinecone.from_documents(documents, embeddings, index_name=index_name)


def get_qa(llm: ChatOpenAI, vectorstore: Pinecone, chain_type: str) -> RetrievalQAWithSourcesChain:
    return RetrievalQAWithSourcesChain.from_chain_type(
        retriever=vectorstore.as_retriever(),
        chain_type=chain_type,
        llm=llm,
    )


def fetch_answer(query: str, qa: RetrievalQAWithSourcesChain, response_area: ScrolledText) -> None:
    with get_openai_callback() as cb:
        result = qa({"question": query})
    response_area.insert(tk.END, f'{result["answer"]}\n\nSpent a total of {cb.total_tokens} tokens')


def send_query(qa: RetrievalQAWithSourcesChain, text_area: ScrolledText, response_area: ScrolledText) -> None:
    query = text_area.get('1.0', 'end-1c')
    text_area.delete('1.0', tk.END)
    response_area.delete('1.0', tk.END)
    threading.Thread(target=fetch_answer, args=(query, qa, response_area)).start()


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="This program extracts and processes information from specified PDFs and \
                    URLs using OpenAI's NLP models. The resulting embeddings are vectorized with \
                    Pinecone, creating an efficient queryable index for user interaction."
    )
    parser.add_argument("--pdf_folder", type=str, help="Folder containing PDF documents")
    parser.add_argument("--urls_file", type=str, help="Text file containing the URLs to examine")
    parser.add_argument("--openai_api_key", type=str, help="OpenAI API Key (Example: my-openai-api-key)")
    parser.add_argument("--openai_model", type=str, help="OpenAI model type (Example: gpt-3.5-turbo)")
    parser.add_argument("--openai_temp", type=float, help="Sets temperature in OpenAI (Default: 0.3)")
    parser.add_argument("--pinecone_api_key", type=str, help="Pinecone API Key (Example: my-pinecone-api-key)")
    parser.add_argument("--pinecone_env", type=str, help="Pinecone environment (region) (Example: us-west-2)")
    parser.add_argument("--chain_type", type=str, help="Methods to pass multiple documents (stuff, map_reduce, refine, map-rerank)")
    args = parser.parse_args()

    # Set Envs
    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    os.environ["PYTHONHTTPSVERIFY"] = "0"

    # Read and clean documents
    pdfs, urls = [], []
    if args.pdf_folder:
        pdfs = read_documents(args.pdf_folder)
    if args.urls_file:
        urls = read_urls(args.urls_file)
    documents = pdfs + urls
    if not documents:
        print(f'Any document provided')
        exit(0)
    clean_documents(documents)

    # Initialize OpenAI, Pinecone, and QA
    llm, embeddings = get_llm_and_embeddings(args.openai_model, args.openai_temp)
    vectorstore = get_vectorstore(args.pinecone_api_key, args.pinecone_env, documents, embeddings)
    qa = get_qa(llm, vectorstore, args.chain_type)

    # Initialize GUI
    root = tk.Tk()
    text_label = tk.Label(root, text='Your Question:')
    text_label.grid(row=0, column=0)
    text_area = ScrolledText(root, width=40, height=10)
    text_area.grid(row=0, column=0, sticky="nsew")
    response_label = tk.Label(root, text='AI Response:')
    response_label.grid(row=2, column=0)
    response_area = ScrolledText(root, width=40, height=10)
    response_area.grid(row=1, column=0, sticky="nsew")
    button = tk.Button(root, text='Submit', command=lambda: send_query(qa, text_area, response_area))
    button.grid(row=2, column=0)
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    root.mainloop()


if __name__ == "__main__":
    main()

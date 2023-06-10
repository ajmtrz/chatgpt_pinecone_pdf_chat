import os
import re
import time
import queue
import threading
import argparse
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from typing import List
from urllib.parse import urlparse, urljoin

from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader, SeleniumURLLoader
from langchain.schema import Document
from langchain.vectorstores import Pinecone
import pinecone

# Queue
log_queue = queue.Queue()


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


def read_pdfs(path: str) -> List[Document]:
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


def join_data(pdf_folder: str, urls_file: str) -> List[Document]:
    # Read and clean documents
    pdfs, urls = [], []
    if pdf_folder:
        pdfs = read_pdfs(pdf_folder)
    if urls_file:
        urls = read_urls(urls_file)
    docs = pdfs + urls
    if not docs:
        print(f'Any document provided')
        exit(0)
    clean_documents(docs)
    docs_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=0)
    documents = docs_splitter.split_documents(docs)
    return documents


def get_vectorstore(
        pinecone_api_key: str,
        pinecone_env: str,
        pinecone_index_name: str,
        pdf_folder: str,
        urls_file: str
) -> Pinecone:
    embeddings = OpenAIEmbeddings()
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    pinecone_indexes = pinecone.list_indexes()
    if pinecone_index_name not in pinecone_indexes:
        for index in pinecone_indexes:
            pinecone.delete_index(index)
        # Create index
        pinecone.create_index(pinecone_index_name, dimension=1536, metric="cosine", pod_type="p1")
        # Get documents
        documents = join_data(pdf_folder, urls_file)
        # Wait to ready
        while True:
            try:
                index_info = pinecone.describe_index(name=pinecone_index_name)
                if len(index_info) > 0:
                    if index_info[7]['state'] == 'Ready':
                        pinecone_object = Pinecone.from_documents(documents, embeddings, index_name=pinecone_index_name)
                        break
            except Exception as e:
                print(f'Waiting for index to be ready, current status: {e}')
            time.sleep(5)
    else:
        pinecone_object = Pinecone.from_existing_index(index_name=pinecone_index_name, embedding=embeddings)
    return pinecone_object


def get_qa(temp: str, model: str, vectorstore: Pinecone) -> BaseConversationalRetrievalChain:
    retriever = vectorstore.as_retriever()
    streaming_llm_4 = ChatOpenAI(temperature=temp, model=model, streaming=True,
                                 callbacks=[StreamingStdOutCallbackHandler()])
    llm_3 = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
    memory = ConversationSummaryBufferMemory(llm=llm_3, max_token_limit=100, memory_key="chat_history",
                                             return_messages=True, input_key='question', output_key='answer')
    return ConversationalRetrievalChain.from_llm(llm=streaming_llm_4, chain_type="refine", retriever=retriever,
                                                 condense_question_llm=llm_3, memory=memory,
                                                 condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                                                 return_source_documents=True)


def fetch_answer(query: str, qa: BaseConversationalRetrievalChain, response_area: ScrolledText) -> None:
    log_queue.put("Waiting for response...")
    with get_openai_callback() as cb:
        result = qa({"question": query})
    log_queue.put("Ready to query")
    response_area.insert(tk.END, f'{result["answer"]}\n\nSpent a total of {cb.total_tokens} tokens')


def send_query(qa: BaseConversationalRetrievalChain, text_area: ScrolledText, response_area: ScrolledText) -> None:
    query = text_area.get('1.0', 'end-1c')
    text_area.delete('1.0', tk.END)
    response_area.delete('1.0', tk.END)
    threading.Thread(target=fetch_answer, args=(query, qa, response_area)).start()


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--pinecone_index_name", type=str, help="Pinecone Index name")
    return parser.parse_args()


def gui(qa: BaseConversationalRetrievalChain) -> None:
    def update_status_bar():
        try:
            message = log_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            status_text.set(message)
        root.after(1000, update_status_bar)

    # Initialize GUI
    root = tk.Tk()
    root.geometry("800x600")
    root.title("ChatGPTDoMyHomework")

    text_label = tk.Label(root, text="Your question about the context:")
    text_label.grid(row=0, column=0, sticky='W')

    text_area = ScrolledText(root, width=40, height=10)
    text_area.grid(row=1, column=0, sticky="nsew")

    button = tk.Button(root, text='Submit', command=lambda: send_query(qa, text_area, response_area))
    button.grid(row=2, column=0)

    response_label = tk.Label(root, text="AI response about the context:")
    response_label.grid(row=3, column=0, sticky='W')

    response_area = ScrolledText(root, width=40, height=10)
    response_area.grid(row=4, column=0, sticky="nsew")

    status_text = tk.StringVar()
    status_bar = tk.Label(root, textvariable=status_text, bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.grid(row=5, column=0, sticky='we')

    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    root.grid_rowconfigure(4, weight=1)
    root.after(1000, update_status_bar)
    root.mainloop()


def main() -> None:
    # log
    print("Initializing, please wait...")

    # Parse arguments
    args = parse_args()

    # Set Envs
    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    os.environ["PYTHONHTTPSVERIFY"] = "0"

    # Initialize OpenAI, Pinecone, and QA
    vectorstore = get_vectorstore(args.pinecone_api_key, args.pinecone_env, args.pinecone_index_name, args.pdf_folder,
                                  args.urls_file)
    qa = get_qa(args.openai_temp, args.openai_model, vectorstore)

    # log
    print("Ready")

    # Initialize GUI
    log_queue.put("Ready to query")
    gui(qa)


# Run
if __name__ == "__main__":
    main()

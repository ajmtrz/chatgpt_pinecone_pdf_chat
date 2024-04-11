# ChatGPT Pinecone PDF and URL Chat

Welcome to "ChatGPT Pinecone PDF and URL Chat", a sophisticated Python script designed for individuals seeking efficient academic support. This advanced tool is adept at extracting information from both PDFs and URLs, leveraging the power of OpenAI to enhance and integrate content seamlessly. Through a secure integration with Pinecone, it establishes a queryable index, providing a robust platform for accessing information.

## Prerequisites

* An OpenAI API Key: Get yours from [OpenAI](https://platform.openai.com/account/api-keys)
* A Pinecone API Key: Get yours from [Pinecone](https://www.pinecone.io/)

## Usage

Our Python script accepts a variety of arguments to get your homework done the way you want:

* `--pdf_folder`: Directory containing PDF files for the script to read.
* `--urls_file`: Text file containing the URLs the script should examine.
* `--openai_api_key`: Your personal OpenAI API Key.
* `--openai_model`: The type of OpenAI model to use (Example: `gpt-3.5-turbo`).
* `--openai_temp`: Sets the temperature for the OpenAI model.
* `--pinecone_api_key`: Your personal Pinecone API Key.
* `--pinecone_env`: The Pinecone environment (region) to use.
* `--pinecone_index_name`: Name of your Pinecone Index.

Once your arguments are set, the script gets to work, reading and cleaning documents from the specified sources, initializing OpenAI and Pinecone, and setting up a GUI for you to send queries about the content of the documents.
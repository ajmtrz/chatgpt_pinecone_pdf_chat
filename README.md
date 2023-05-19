# ChatGPT Do My Homework

*No pain, all gain! Be the homework hero with this hilarious piece of wizardry...*

Welcome to "ChatGPT Do My Homework", the repository for those who value their sanity. This wonder of a Python script reads your boring homework from PDFs and URLs, embezzles, sorry I meant 'embeds', them with the magic of OpenAI, and then does a secret handshake with Pinecone to create a queryable index of information. In simpler terms, it's the homework-doing, mind-blowing sidekick you always wanted.

## Prerequisites

* An OpenAI API Key: Get yours from [OpenAI](https://platform.openai.com/account/api-keys)
* A Pinecone API Key: Get yours from [Pinecone](https://www.pinecone.io/)

## Usage

Our Python script accepts a variety of arguments to get your homework done the way you want:

* `--pdf_folder`: Directory containing PDF files for the script to read.
* `--urls_file`: Text file containing the URLs the script should examine.
* `--openai_api_key`: Your personal OpenAI API Key. Guard this with your life.
* `--openai_model`: The type of OpenAI model to use (Example: `gpt-3.5-turbo`). More models, more fun!
* `--openai_temp`: Sets the temperature for the OpenAI model. Keep it cool, my friends.
* `--pinecone_api_key`: Your personal Pinecone API Key. Guard this as you would your Netflix password.
* `--pinecone_env`: The Pinecone environment (region) to use. Choose your own adventure!
* `--chain_type`: How to process multiple documents (`stuff`, `map_reduce`, `refine`, `map-rerank`). Variety is the spice of life!

Once your arguments are set, the script gets to work, reading and cleaning documents from the specified sources, initializing OpenAI and Pinecone, and setting up a GUI for you to send queries to your new homework helper. Homework has never been so... entertaining!

## Disclaimer

This README does not encourage or endorse cheating or shirking your educational responsibilities. This project is intended as a funny and engaging way to explore NLP, embeddings, vectorization, and related technologies. Please use responsibly, and always do your homework! 😉

## License

MIT License. Because we believe in sharing the funny.

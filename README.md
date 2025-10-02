# Integrating documents into large language models (LLMs)

## Skills
Python: LangChain, general class structures</br>
LLMs: Google Gemini API, prompt engineering

## Executive Summary
Internal documentation can easily surbase 100 pages for a single document.
This makes it difficult for stakeholders and new employees to quickly fined the necessary information.
These documents are often internal and proprietary, so we cannot rely on public large language models (LLMs) to have prior knowledge of them.
Documents are loaded into memory, and using **LangChain**, they are passed to Google Gemini for processing.
Results are returned via the **API**, and are printed to the terminal.
The object-oriented structure allows for different "libraries" where specific documents can be placed.
While this demo uses Gemini, the architecture supports other LLMs like GPT-4 or Claude, and can be extended to PDFs, emails, or structured databases.
See an example [parsing my PhD thesis](thesis.md).

## Business Problem 
How can integrating proprietary documents with LLMs improve internal knowledge retrieval and reduce time spent searching for information in enterprise settings?

## Methodology
I built a simple interface using LangChain and the Google Gemini API to:
1. Load important documents
1. Send the parsed text to Gemini for processing
1. Simple **prompt engineering** to ensure the desired information is extracted

## Business Recommendations
All internal documents should be add to a private copy of the repository.
Using the "library" structure, different subsets of documents can be organized to enable faster processing.

## Next Steps
1. Expand to use a Retrieval-Augmented Generation (RAG)
1. Use a vector data base like Chroma to embed documents for better context-aware searches
<br><br><br>

<img src="https://raw.githubusercontent.com/bryates/Alexandria/main/logo.jpg" width="50%">

# Alexandria
[![Python application](https://github.com/bryates/Alexandria/actions/workflows/test_library.yml/badge.svg)](https://github.com/bryates/Alexandria/actions/workflows/test_library.yml)<br>
Add your documents to the LLM library

## Installation
Clone the repo with
```bash
git clone git@github.com:bryates/Alexandria.git
```
and install with
```bash
pip install .
```

## Runnig
For details on adding documents, see an example in [thesis.py](thesis.py)
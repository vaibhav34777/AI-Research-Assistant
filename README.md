# AI Research Assistant

A lightweight, interactive AI-powered research assistant built using Retrieval-Augmented Generation (RAG). This app helps users ask questions based on indexed research papers in **Machine Learning** and **Electronics**.

## Features

-  Semantic search over research paper chunks using **FAISS** and **Sentence Transformers**
-  Answer generation using **Google FLAN-T5 Large** via Hugging Face Inference API
-  Displays citations and relevant context for each answer
-  Currently supports 20 papers, easily extendable

## Demo

<img src="assets/demo.png" alt="demo" width="600"/>

##  Tech Stack

- [Streamlit](https://streamlit.io/)
- [FAISS](https://github.com/facebookresearch/faiss) (vector similarity search)
- [Sentence-Transformers](https://www.sbert.net/) for embeddings
- [Hugging Face Inference API](https://huggingface.co/inference-api) for answer generation

## Setup Instructions

1. **Clone this repo**

```bash
git clone https://github.com/vaibhav34777/AI-Research-Assistant.git
cd rag-research-assistant
2. **Install Dependencies**

```bash
pip install -r requirements.txt






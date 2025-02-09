## Vehicle License plate and brand detection & Vehicle Query System
This project integrates video processing, vehicle detection, tracking, license plate recognition, and a retrieval-augmented query system. It comprises two main components:
1. Video Processing & Tracking Pipeline: Uses YOLOv8 for vehicle detection and brand classification. Uses OpenALPR for license plate recognition. Tracks vehicles across video frames and logs results to a JSON file.
2. Data Ingestion, Knowledge Graph & Query System: Ingests tracking results into a ChromaDB vector store, Builds a knowledge graph (using NetworkX) to capture relationships (vehicle, plate, brand)., Provides a retrieval-augmented question-answering interface using a Llama 3.2 1B model (via Hugging Face Transformers) and a Streamlit UI.

## Prerequisites

Python 3.9 or later, opencv-python, FFmpeg, opencv-python, ultralytics, chromadb, sentence-transformers, networkx, matplotlib, torch, transformers, streamlit

## Installation
Clone this repository and install all the required pre-requisities as mentioned above

## Usage
Place the input video in the project folder. 
Run the video processing script : readLicensePlateAndBrand.py, it will generata a tracking_results.json and tracked_output.avi file

Then to run the chatbot on Streamlit UI, use python script vehicleQAUI.py, for running on command line use vehicleQACmdLine.py
It will create a chromadb vector db instance automatically in your project folder and you can use the chatbot. Note, the chatbot is not fully functional and some improvements can be made.

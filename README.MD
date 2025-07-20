# Image Search using FastAPI and Streamlit (Dockerized)

This is a simple image search system where you can enter a query and get back relevant images. It uses FastAPI as the backend, Streamlit as the frontend, and is containerized using Docker.

---

## Features
- Search for images using text queries
- Shows matching images with short explanations
- Uses open-source models (no OpenAI)
- Works on CPU (no GPU needed)
- Docker-ready and cloud-deployable

---

## Folder Structure
The folder includes:

The folder includes:

- app.py: Streamlit UI  
- api/: FastAPI backend code  
  - main.py: handles routing and query calls  
- download_images.py: script to download images  
- photos_url.csv: list of image URLs  
- generate_caption.py: generates image captions  
- search.py: handles similarity search logic  
- explanation.py: creates explanations for matches  
- embeddings/: stores generated image vectors and filenames    
- requirements.txt: Python dependencies  
- Dockerfile: container setup  
- README.md: project instructions


---

## Run Locally (Without Docker)

1. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate 
```

2. Install packages:
```bash
pip install -r requirements.txt
```

3. Start FastAPI:
```bash
uvicorn api.main:app --reload --port 8080
```
Go to: http://localhost:8080/docs

4. In a new terminal, start Streamlit:
```bash
streamlit run app.py
```
Go to: http://localhost:8501

---

## Run with Docker

1. Build the Docker image:
```bash
docker build -t image-search .
```

2. Run FastAPI:
```bash
docker run -p 8080:8080 image-search
```
Visit: http://localhost:8080/docs

3. (Optional) Run Streamlit:
```bash
docker run -p 8501:8501 image-search streamlit run app.py
```
Visit: http://localhost:8501

---

## Deploying on Render

1. Push this project to GitHub
2. Go to https://render.com and create:
   - A web service for FastAPI using the Dockerfile (port 8080)
   - Another web service for Streamlit (port 8501)

---

## Notes
- Works without OpenAI or GPU
- Add `KMP_DUPLICATE_LIB_OK=TRUE` if OpenMP errors occur

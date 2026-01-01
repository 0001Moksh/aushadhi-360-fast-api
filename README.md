# Aushadhi 360 API

A FastAPI-based medical information retrieval system using AI embeddings and Groq LLM.

## Features
- MongoDB integration
- Sentence transformers for embeddings
- Groq LLM API integration
- FAISS vector search
- CORS enabled

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Environment variables:**
Create a `.env` file:
```
DATABASE_URL=your_mongodb_connection_string
GROQ_API_KEY=your_groq_api_key
```

3. **Run locally:**
```bash
uvicorn main:app --reload
```

## Deployment

### Render
1. Push to GitHub
2. Connect your GitHub repo to Render
3. Render will automatically use `render.yaml` for deployment
4. Add environment variables in Render dashboard

## API Documentation
Once running, visit `http://localhost:8000/docs` for interactive API documentation.

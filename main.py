import uvicorn
import os
import json
import math
import time
import numpy as np
import pandas as pd
import dotenv
from contextlib import asynccontextmanager
from functools import lru_cache
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from groq import Groq

# ================= ENV =================
dotenv.load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# ================= LIFESPAN =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Aushadhi 360 API starting...")
    yield
    print("🛑 Aushadhi 360 API shutting down...")
    if mongo_client:
        mongo_client.close()

# ================= APP =================
app = FastAPI(title="Aushadhi 360 API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= DB =================
mongo_client = MongoClient(DATABASE_URL)
db = mongo_client["aushadhi360"]
users_collection = db["users"]

# ================= MODELS =================
embed_model = SentenceTransformer("intfloat/multilingual-e5-base")

# ================= GLOBAL CACHE =================
# Per-user in-memory cache (LRU: keep last 50 users)
USER_CACHE = {}

@lru_cache(maxsize=50)
def get_cache_key(email: str):
    """LRU cache for tracking active users"""
    return email

# ================= UTILS =================
def safe_numpy(arr):
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def llm(prompt: str, api_key: str) -> str:
    """Invoke Groq using the requesting user's key."""
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def get_user_data(mail: str, password: str):
    return list(users_collection.find({"email": mail, "password": password}).limit(1))

def build_fallback_payload(query: str, medicines: list, reason: str):
    meds = []
    for i, m in enumerate(medicines[:5], start=1):
        clean = {k: (None if pd.isna(v) else v) for k, v in m.items()}
        clean.update({
            "S.no": i,
            "Instructions": "No AI guidance. Please consult a doctor."
        })
        meds.append(clean)

    return {
        "AI Response": f"AI unavailable ({reason}). Showing similarity-based results.",
        "Medicines": meds,
        "Score": "N/A",
        "overall instructions": "Consult a healthcare professional before use.",
        "fallback": True,
        "query": query
    }

# ================= EMBEDDING BUILDER =================
def build_embeddings_for_user(mail: str, password: str):
    """
    Build and store embeddings in MongoDB for a user.
    Called on login or when cache is invalidated.
    """
    user_data = get_user_data(mail, password)
    if not user_data:
        return False

    user_id = user_data[0]["_id"]
    medicines = user_data[0].get("medicines", [])
    if not medicines:
        return False

    df = pd.DataFrame(medicines)

    df["text"] = (
        df.get("Cover Disease", "").fillna("").astype(str) + " || " +
        df.get("Symptoms", "").fillna("").astype(str)
    ).str.strip()

    df = df[df["text"] != ""]
    df = df[df["text"] != "||"]

    if df.empty:
        return False

    # Compute embeddings
    embeddings = embed_model.encode(df["text"].tolist(), show_progress_bar=False)
    embeddings = safe_numpy(embeddings)

    vectors = np.array(embeddings, dtype=np.float32)
    dim = vectors.shape[1]

    # Create FAISS index
    import faiss
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # Save to MongoDB
    current_time = int(time.time())
    df_clean = df.drop(["text"], axis=1, errors="ignore")
    
    users_collection.update_one(
        {"_id": user_id},
        {
            "$set": {
                "embeddings": {
                    "vectors": embeddings.tolist(),
                    "medicine_ids": df_clean.to_dict(orient="records"),
                    "computed_at": current_time,
                    "embedding_dim": dim
                },
                "last_medicine_update": current_time
            }
        }
    )

    # Cache in memory for fast access
    USER_CACHE[mail] = {
        "faiss_index": index,
        "medicine_df": df_clean.reset_index(drop=True),
        "embedding_dim": dim,
        "computed_at": current_time,
        "last_medicine_update": current_time
    }

    return True


def load_user_embeddings(mail: str, password: str):
    """
    Load user embeddings from cache or MongoDB.
    Returns (faiss_index, medicine_df, embedding_dim) or (None, None, None)
    """
    user_data = get_user_data(mail, password)
    if not user_data:
        return None, None, None

    user = user_data[0]
    last_medicine_update = user.get("last_medicine_update", 0)
    embeddings_data = user.get("embeddings", {})
    computed_at = embeddings_data.get("computed_at", 0)

    # Check if embeddings are stale (medicine data changed after embeddings were computed)
    if last_medicine_update > computed_at:
        # Stale embeddings, rebuild
        build_embeddings_for_user(mail, password)
        return load_user_embeddings(mail, password)  # Recursive call to load fresh data

    # Check in-memory cache first
    if mail in USER_CACHE:
        return (
            USER_CACHE[mail]["faiss_index"],
            USER_CACHE[mail]["medicine_df"],
            USER_CACHE[mail]["embedding_dim"]
        )

    # Load from MongoDB
    if embeddings_data and "vectors" in embeddings_data:
        vectors = np.array(embeddings_data["vectors"], dtype=np.float32)
        dim = embeddings_data.get("embedding_dim", vectors.shape[1])

        import faiss
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)

        medicine_ids = embeddings_data.get("medicine_ids", [])
        medicine_df = pd.DataFrame(medicine_ids)

        # Cache in memory
        USER_CACHE[mail] = {
            "faiss_index": index,
            "medicine_df": medicine_df,
            "embedding_dim": dim,
            "computed_at": computed_at,
            "last_medicine_update": last_medicine_update
        }

        return index, medicine_df, dim

    return None, None, None

# ================= SEARCH =================
def collect_data_for(query: str, mail: str, password: str, k: int = 3):
    """
    Search medicines for a specific user.
    Automatically handles cache invalidation based on medicine data changes.
    """
    FAISS_INDEX, MEDICINE_DF, EMBEDDING_DIM = load_user_embeddings(mail, password)

    # If nothing cached yet, build once and retry so first query after login works
    if FAISS_INDEX is None or MEDICINE_DF is None:
        build_embeddings_for_user(mail, password)
        FAISS_INDEX, MEDICINE_DF, EMBEDDING_DIM = load_user_embeddings(mail, password)

    if FAISS_INDEX is None or MEDICINE_DF is None:
        raise HTTPException(status_code=404, detail="No medicines found for this user. Add medicines and try again.")

    vec = embed_model.encode(query)
    vec = safe_numpy(vec)

    svec = np.array(vec, dtype=np.float32).reshape(1, -1)
    distances, indices = FAISS_INDEX.search(svec, k)

    result = MEDICINE_DF.iloc[indices[0]]
    result = result.drop(
        ["embedding", "text", "status_import", "Total_Quantity"],
        axis=1,
        errors="ignore"
    )

    result = result.replace([np.nan, np.inf, -np.inf], None)
    return result.to_dict(orient="records")

# ================= ROUTES =================
@app.get("/")
def root():
    return {"status": "running", "service": "Aushadhi 360 API"}

# ---------- LOGIN ----------
@app.post("/login")
def login(
    background_tasks: BackgroundTasks,
    mail: str = "demo@aushdi.com",
    password: str = "demo123"
):
    user = get_user_data(mail, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    background_tasks.add_task(build_embeddings_for_user, mail, password)
    return {"status": "success", "message": "Login successful. AI preparing."}

# ---------- INVALIDATE CACHE ----------
@app.post("/invalidate-cache")
def invalidate_cache(mail: str, password: str):
    """
    Invalidate embeddings cache for a user.
    Call this after updating/deleting medicines so the API recomputes embeddings.
    """
    user_data = get_user_data(mail, password)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user_id = user_data[0]["_id"]
    current_time = int(time.time())

    # Update last_medicine_update to current time
    # This signals that embeddings are stale
    users_collection.update_one(
        {"_id": user_id},
        {"$set": {"last_medicine_update": current_time}}
    )

    # Clear from memory cache
    if mail in USER_CACHE:
        del USER_CACHE[mail]

    return {
        "status": "success",
        "message": "Cache invalidated. Embeddings will be recomputed on next query."
    }

# ---------- GET MEDICINES ----------
@app.get("/get_medicines")
def get_medicines(
    query: str,
    mail: str | None = None,
    password: str | None = None
):
    """
    Get medicines based on user query.
    Only returns medicines from the requesting user's collection.
    Automatically detects if cache needs refresh based on medicine data changes.
    """
    if not mail or not password:
        raise HTTPException(status_code=400, detail="mail and password query params are required")

    user_data = get_user_data(mail, password)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user_doc = user_data[0]
    groq_api_key = user_doc.get("groqKeyAssist")
    if not groq_api_key:
        raise HTTPException(status_code=400, detail="groqKeyAssist is required for this user")

    medicines = collect_data_for(query, mail, password)

    if not medicines:
        return jsonable_encoder(build_fallback_payload(query, [], "no matches"))

    llm_prompt = f"""
You are a medical assistant AI.
Use ONLY the provided Medicine_data.
Do NOT invent medicines.

Medicine_data: {medicines}
Patient Query: {query}

Return STRICT JSON:
{{
  "AI Response": "",
  "Medicines": [
    {{
      "S.no": 1,
      "Name of Medicine": "",
      "Batch_ID": "",
      "Description": "",
      "Quantity": "",
      "Instructions": ""
    }}
  ],
  "Score": "XX%",
  "overall instructions": ""
}}
"""

    try:
        response = llm(llm_prompt, groq_api_key)

        if not response:
            return jsonable_encoder(build_fallback_payload(query, medicines, "empty LLM"))

        response = response.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return jsonable_encoder(build_fallback_payload(query, medicines, "invalid JSON"))

        master = {m.get("Batch_ID"): m for m in medicines if m.get("Batch_ID")}
        merged = []

        for med in data.get("Medicines", []):
            bid = med.get("Batch_ID")
            merged.append({**master.get(bid, {}), **med})

        data["Medicines"] = merged
        data["fallback"] = False
        data["query"] = query

        return jsonable_encoder(data)

    except Exception as e:
        return jsonable_encoder(build_fallback_payload(query, medicines, str(e)))

# # ================= RUN =================
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

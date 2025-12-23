import uvicorn
import os
import json
import numpy as np
import pandas as pd
import dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai.errors import ClientError as GenAIClientError
from fastapi.middleware.cors import CORSMiddleware

# ================= ENV =================
dotenv.load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("API_KEY")

# ================= APP =================
app = FastAPI(title="Aushadhi 360 API")

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
llm_client = genai.Client(api_key=API_KEY)

# ================= GLOBAL CACHE =================
FAISS_INDEX = None
MEDICINE_DF = None
EMBEDDING_DIM = None

# ================= UTILS =================
def llm(prompt: str) -> str:
    response = llm_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text.strip()


def build_fallback_payload(query: str, medicines: list, reason: str):
    top = medicines[:5] if medicines else []
    fallback_meds = []
    for i, m in enumerate(top, start=1):
        name = (
            m.get("Name of Medicine")
            or m.get("Medicine Name")
            or m.get("name")
            or m.get("Product Name")
            or ""
        )
        batch_id = m.get("Batch_ID") or m.get("batch_id") or ""
        desc = (
            m.get("Description")
            or m.get("Short Description")
            or m.get("description")
            or ""
        )
        qty = m.get("Quantity") or m.get("Dosage") or ""
        item = dict(m)
        item.update(
            {
                "S.no": i,
                "Name of Medicine": name,
                "Batch_ID": batch_id,
                "Description": desc,
                "Quantity": qty,
                "Instructions": "No LLM guidance. Consult a healthcare professional.",
            }
        )
        fallback_meds.append(item)

    return {
        "AI Response": f"LLM unavailable: {reason}. Showing top matches without AI guidance.",
        "Medicines": fallback_meds,
        "Score": "N/A",
        "overall instructions": "These are suggested from your catalog using similarity only. Please consult a pharmacist/doctor before use.",
        "fallback": True,
        "query": query,
    }


def get_user_data(mail: str, password: str):
    return list(users_collection.find({"email": mail, "password": password}).limit(1))


# ================= EMBEDDING BUILDER =================
def build_embeddings_for_user(mail: str, password: str):
    global FAISS_INDEX, MEDICINE_DF, EMBEDDING_DIM

    user_data = get_user_data(mail, password)
    if not user_data:
        return

    medicines = user_data[0].get("medicines", [])
    if not medicines:
        return

    df = pd.DataFrame(medicines)
    df = df.dropna(subset=["Cover Disease", "Symptoms"], how="all")

    df["text"] = df["Cover Disease"].fillna("") + " || " + df["Symptoms"].fillna("")

    embeddings = embed_model.encode(df["text"].tolist(), show_progress_bar=False)
    df["embedding"] = embeddings.tolist()

    vectors = np.array(df["embedding"].tolist())
    dim = vectors.shape[1]

    import faiss

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # Cache
    FAISS_INDEX = index
    MEDICINE_DF = df
    EMBEDDING_DIM = dim


# ================= SEARCH =================
def collect_data_for(query: str, k: int = 3):
    if FAISS_INDEX is None or MEDICINE_DF is None:
        raise HTTPException(status_code=400, detail="AI not ready yet")

    vec = embed_model.encode(query)
    svec = np.array(vec).reshape(1, -1)

    distances, indices = FAISS_INDEX.search(svec, k)
    result = MEDICINE_DF.iloc[indices[0]]

    result = result.drop(["embedding", "text", "status_import", "Total_Quantity"], axis=1, errors="ignore")

    return result.to_dict(orient="records")


# ================= ROUTES =================
@app.get("/")
def root():
    return {"status": "running", "service": "Aushadhi 360 API"}


# ---------- LOGIN (PREPARE AI IN BACKGROUND) ----------
@app.post("/login")
def login(mail: str, password: str, background_tasks: BackgroundTasks):
    user = get_user_data(mail, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    background_tasks.add_task(build_embeddings_for_user, mail, password)

    return {"status": "success", "message": "Login successful. AI is preparing in background."}


# ---------- GET MEDICINES ----------
@app.get("/get_medicines")
def get_medicines(query: str):
    medicines = collect_data_for(query)

    llm_prompt = f"""
You are a medical assistant AI.
Use ONLY the provided Medicine_data.
Do NOT invent medicines.

Medicine_data: {medicines}
Patient Query: {query}

Return STRICTLY valid JSON:
{{
  "AI Response": "This person has {{detected problem}} issue",
  "Medicines": [
    {{
      "S.no": 1,
      "Name of Medicine": "",
      "Batch_ID": "",
      "Description": "1-line purpose",
      "Quantity": "short dosage",
      "Instructions": "doctor-style note"
    }}
  ],
  "Score": "XX%",
  "overall instructions": "2–3 line lifestyle advice"
}}
"""

    try:
        response = llm(llm_prompt)

        if not response:
            return build_fallback_payload(query, medicines, reason="empty response from LLM")

        response = response.strip()
        if response.startswith("```"):
            response = response.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return build_fallback_payload(query, medicines, reason="invalid JSON from LLM")

        master_lookup = {m.get("Batch_ID"): m for m in medicines if m.get("Batch_ID")}
        merged = []
        for med in data.get("Medicines", []):
            bid = med.get("Batch_ID")
            if bid and bid in master_lookup:
                merged.append({**med, **master_lookup[bid]})
            else:
                merged.append(med)

        data["Medicines"] = merged
        data["fallback"] = False
        data["query"] = query
        return data

    except GenAIClientError as e:
        status = getattr(e, "status_code", None)
        if status == 429 or "RESOURCE_EXHAUSTED" in str(e):
            return build_fallback_payload(query, medicines, reason="quota exceeded")
        return build_fallback_payload(query, medicines, reason=str(e))
    except Exception as e:
        return build_fallback_payload(query, medicines, reason=str(e))


# ================= RUN =================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


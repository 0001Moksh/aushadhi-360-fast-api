
import uvicorn
import os
import json
import time
import hashlib
import io
import tempfile
import numpy as np
import pandas as pd
import dotenv
import requests
from contextlib import asynccontextmanager
from functools import lru_cache
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from bson import ObjectId
from sentence_transformers import SentenceTransformer
from groq import Groq
import faiss

# ================= ENV =================
dotenv.load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
GRAPH_SCOPE = os.getenv("GRAPH_SCOPE", "https://graph.microsoft.com/.default")
GRAPH_TOKEN_URL = os.getenv("GRAPH_TOKEN_URL")
ONEDRIVE_DRIVE_ID = os.getenv("ONEDRIVE_DRIVE_ID")
ONEDRIVE_ROOT_FOLDER = os.getenv("ONEDRIVE_ROOT_FOLDER", "OCD360_Embeddings")

if not ONEDRIVE_DRIVE_ID:
    raise ValueError("ONEDRIVE_DRIVE_ID environment variable is required")

# ================= DB =================
mongo_client = MongoClient(DATABASE_URL)
db = mongo_client["aushadhi360"]
users_collection = db["users"]
medicines_collection = db["medicines"]
embedding_meta_collection = db["embedding_meta"]

# Create indexes for performance
users_collection.create_index("email")
medicines_collection.create_index("userId")
embedding_meta_collection.create_index([("userId", 1), ("embedding_version", -1)])

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

# ================= ONEDRIVE / GRAPH API (Drive-Based, App-Only Auth) =================
def get_graph_access_token() -> str:
    """Get access token for Microsoft Graph API using client credentials flow (app-only auth)."""
    try:
        data = {
            "grant_type": "client_credentials",
            "client_id": AZURE_CLIENT_ID,
            "client_secret": AZURE_CLIENT_SECRET,
            "scope": GRAPH_SCOPE
        }
        
        response = requests.post(GRAPH_TOKEN_URL, data=data)
        if response.status_code != 200:
            error_detail = response.json() if response.text else response.text
            print(f"[ERROR] Token request failed: {error_detail}")
            raise HTTPException(status_code=500, detail="Authentication failed")
        
        token = response.json().get("access_token")
        print(f"[OK] Access token acquired")
        return token
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] get_graph_access_token: {str(e)}")
        raise HTTPException(status_code=500, detail="Token acquisition failed")

def get_or_create_user_folder(access_token: str, email: str) -> str:
    """
    Creates a folder per user inside: /OCD360_Embeddings/{email}/
    Uses drive-based API: /drives/{ONEDRIVE_DRIVE_ID}/...
    Returns: folder_id of the user's email folder
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    
    try:
        # Step 1: Get root folder children to find/create OCD360_Embeddings folder
        root_url = f"https://graph.microsoft.com/v1.0/drives/{ONEDRIVE_DRIVE_ID}/root/children"
        root_resp = requests.get(root_url, headers=headers)
        
        if root_resp.status_code != 200:
            error = root_resp.json().get("error", {}) if root_resp.text else {}
            print(f"[ERROR] Failed to list root: {root_resp.status_code} - {error.get('message', 'Unknown')}")
            raise HTTPException(status_code=500, detail="Failed to access drive root")
        
        root_items = root_resp.json().get("value", [])
        root_folder_id = None
        
        # Find existing root folder
        for item in root_items:
            if item.get("name") == ONEDRIVE_ROOT_FOLDER and item.get("folder"):
                root_folder_id = item["id"]
                print(f"[OK] Found root folder: {ONEDRIVE_ROOT_FOLDER}")
                break
        
        # Create root folder if missing
        if not root_folder_id:
            print(f"[INFO] Creating root folder: {ONEDRIVE_ROOT_FOLDER}")
            create_root_url = f"https://graph.microsoft.com/v1.0/drives/{ONEDRIVE_DRIVE_ID}/root/children"
            create_root_body = {
                "name": ONEDRIVE_ROOT_FOLDER,
                "folder": {}
            }
            create_root_resp = requests.post(create_root_url, headers=headers, json=create_root_body)
            
            if create_root_resp.status_code not in [201, 200]:
                error = create_root_resp.json().get("error", {}) if create_root_resp.text else {}
                print(f"[ERROR] Failed to create root folder: {create_root_resp.status_code} - {error.get('message', 'Unknown')}")
                raise HTTPException(status_code=500, detail="Failed to create root folder")
            
            root_folder_id = create_root_resp.json()["id"]
            print(f"[OK] Created root folder")
        
        # Step 2: Find or create email-based folder inside root
        user_folder_url = f"https://graph.microsoft.com/v1.0/drives/{ONEDRIVE_DRIVE_ID}/items/{root_folder_id}/children"
        user_resp = requests.get(user_folder_url, headers=headers)
        
        if user_resp.status_code != 200:
            error = user_resp.json().get("error", {}) if user_resp.text else {}
            print(f"[ERROR] Failed to list root folder: {user_resp.status_code} - {error.get('message', 'Unknown')}")
            raise HTTPException(status_code=500, detail="Failed to access root folder")
        
        user_items = user_resp.json().get("value", [])
        user_folder_id = None
        
        # Find existing user folder
        for item in user_items:
            if item.get("name") == email and item.get("folder"):
                user_folder_id = item["id"]
                print(f"[OK] Found user folder: {email}")
                break
        
        # Create user folder if missing
        if not user_folder_id:
            print(f"[INFO] Creating user folder: {email}")
            create_user_url = f"https://graph.microsoft.com/v1.0/drives/{ONEDRIVE_DRIVE_ID}/items/{root_folder_id}/children"
            create_user_body = {
                "name": email,
                "folder": {}
            }
            create_user_resp = requests.post(create_user_url, headers=headers, json=create_user_body)
            
            if create_user_resp.status_code not in [201, 200]:
                error = create_user_resp.json().get("error", {}) if create_user_resp.text else {}
                print(f"[ERROR] Failed to create user folder: {create_user_resp.status_code} - {error.get('message', 'Unknown')}")
                raise HTTPException(status_code=500, detail="Failed to create user folder")
            
            user_folder_id = create_user_resp.json()["id"]
            print(f"[OK] Created user folder")
        
        return user_folder_id
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] get_or_create_user_folder: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to manage user folder")

def upload_faiss(access_token: str, folder_id: str, file_path: str, filename: str) -> bool:
    """
    Upload file to OneDrive using drive-based API.
    Uses: PUT /drives/{ONEDRIVE_DRIVE_ID}/items/{folder_id}:/{filename}:/content
    """
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Read file content
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Upload using drive-based endpoint
        upload_url = f"https://graph.microsoft.com/v1.0/drives/{ONEDRIVE_DRIVE_ID}/items/{folder_id}:/{filename}:/content"
        
        print(f"[INFO] Uploading {filename}...")
        upload_resp = requests.put(upload_url, headers=headers, data=file_content)
        
        if upload_resp.status_code not in [200, 201]:
            error = upload_resp.json().get("error", {}) if upload_resp.text else {}
            print(f"[ERROR] Upload {filename} failed: {upload_resp.status_code} - {error.get('message', 'Unknown')}")
            raise HTTPException(status_code=500, detail=f"Failed to upload {filename}")
        
        print(f"[OK] Uploaded {filename}")
        return True
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] upload_faiss: {str(e)}")
        raise HTTPException(status_code=500, detail="File upload failed")

def download_file(access_token: str, folder_id: str, filename: str, output_path: str) -> bool:
    """
    Download file from OneDrive using drive-based API.
    """
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # List folder contents to find file
        list_url = f"https://graph.microsoft.com/v1.0/drives/{ONEDRIVE_DRIVE_ID}/items/{folder_id}/children"
        list_resp = requests.get(list_url, headers=headers)
        
        if list_resp.status_code != 200:
            error = list_resp.json().get("error", {}) if list_resp.text else {}
            print(f"[ERROR] Failed to list folder: {list_resp.status_code} - {error.get('message', 'Unknown')}")
            return False
        
        items = list_resp.json().get("value", [])
        file_id = None
        
        for item in items:
            if item.get("name") == filename:
                file_id = item["id"]
                break
        
        if not file_id:
            print(f"[WARN] File not found: {filename}")
            return False
        
        # Download file content
        download_url = f"https://graph.microsoft.com/v1.0/drives/{ONEDRIVE_DRIVE_ID}/items/{file_id}/content"
        download_resp = requests.get(download_url, headers=headers)
        
        if download_resp.status_code != 200:
            error = download_resp.json().get("error", {}) if download_resp.text else {}
            print(f"[ERROR] Failed to download {filename}: {download_resp.status_code} - {error.get('message', 'Unknown')}")
            return False
        
        # Save to file
        with open(output_path, 'wb') as f:
            f.write(download_resp.content)
        
        print(f"[OK] Downloaded {filename}")
        return True
        
    except Exception as e:
        print(f"[ERROR] download_file: {str(e)}")
        return False

# ================= DB =================
mongo_client = MongoClient(DATABASE_URL)
db = mongo_client["aushadhi360"]
users_collection = db["users"]

# ================= MODELS =================
embed_model = SentenceTransformer("intfloat/multilingual-e5-base")

# ================= UTILITIES =================
def clean_objectid(doc):
    """
    Convert ObjectId to string in a MongoDB document for JSON serialization.
    Recursively handles nested dictionaries and lists.
    """
    if isinstance(doc, dict):
        return {k: clean_objectid(v) for k, v in doc.items()}
    elif isinstance(doc, list):
        return [clean_objectid(item) for item in doc]
    elif isinstance(doc, ObjectId):
        return str(doc)
    else:
        return doc

def get_user_medicines(user_id: str) -> list:
    """Fetch all medicines for a user."""
    meds = list(medicines_collection.find({"userId": user_id}))
    return [clean_objectid(m) for m in meds]

def compute_medicines_hash(medicines: list) -> str:
    """
    Compute SHA256 hash of medicines to detect changes.
    Serializes medicines in deterministic order.
    """
    # Sort medicines by _id for deterministic ordering
    sorted_meds = sorted(medicines, key=lambda m: str(m.get("_id", "")))
    
    # Create deterministic JSON string
    med_json = json.dumps(sorted_meds, sort_keys=True, default=str)
    
    # Return SHA256 hash
    return hashlib.sha256(med_json.encode()).hexdigest()

def get_next_embedding_version(user_id: str) -> int:
    """Get the next embedding version number for a user."""
    latest = embedding_meta_collection.find_one(
        {"userId": user_id},
        sort=[("embedding_version", -1)]
    )
    return (latest.get("embedding_version", 0) if latest else 0) + 1

# ================= GLOBAL CACHE =================
# Per-user in-memory cache (LRU: keep last 50 users)
USER_CACHE = {}
GRAPH_TOKEN_CACHE = None  # Cached Graph API token
GRAPH_TOKEN_EXPIRY = None  # Token expiry time

def get_cached_access_token():
    """Get or create cached Graph API access token with refresh logic."""
    global GRAPH_TOKEN_CACHE, GRAPH_TOKEN_EXPIRY
    current_time = time.time()
    
    # Refresh if no token or expired (with 5 min buffer)
    if GRAPH_TOKEN_CACHE is None or (GRAPH_TOKEN_EXPIRY and current_time > GRAPH_TOKEN_EXPIRY - 300):
        GRAPH_TOKEN_CACHE = get_graph_access_token()
        GRAPH_TOKEN_EXPIRY = current_time + 3600  # Tokens typically valid for 1 hour
    
    return GRAPH_TOKEN_CACHE

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
    user = users_collection.find_one({"email": mail, "password": password})
    return user

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

# ================= EMBEDDING WORKER (Background Task) =================
def embedding_worker(user_id: str):
    """
    Background worker to build/update embeddings for a user.
    - Fetches all medicines for user
    - Computes SHA256 hash for change detection
    - Skips rebuild if hash unchanged
    - Saves to OneDrive with versioning (faiss_v1.index, faiss_v2.index, etc.)
    - Updates MongoDB metadata
    """
    try:
        print(f"[INFO] Starting embedding worker for {user_id}...")
        
        # Fetch user
        user = users_collection.find_one({"email": user_id})
        if not user:
            print(f"[ERROR] User {user_id} not found")
            return
        
        # Fetch all medicines
        medicines = get_user_medicines(user_id)
        if not medicines:
            print(f"[WARN] No medicines found for {user_id}")
            users_collection.update_one(
                {"email": user_id},
                {"$set": {"embedding_status": "failed"}}
            )
            return
        
        # Compute hash of current medicines
        current_hash = compute_medicines_hash(medicines)
        
        # Check if medicines changed
        last_meta = embedding_meta_collection.find_one(
            {"userId": user_id},
            sort=[("embedding_version", -1)]
        )
        
        if last_meta and last_meta.get("hash") == current_hash:
            print(f"[OK] No changes detected for {user_id}. Skipping rebuild.")
            users_collection.update_one(
                {"email": user_id},
                {"$set": {"embedding_status": "ready"}}
            )
            return
        
        print(f"[INFO] Changes detected. Building embeddings for {user_id}...")
        
        # Build DataFrame from medicines
        df = pd.DataFrame(medicines)
        
        # Use allowed fields for embedding generation
        df["text"] = (
            df.get("Cover Disease", "").fillna("").astype(str) + " || " +
            df.get("Symptoms", "").fillna("").astype(str) + " || " +
            df.get("Side Effects", "").fillna("").astype(str) + " || " +
            df.get("Description in Hinglish", "").fillna("").astype(str)
        ).str.strip()
        
        df = df[df["text"] != ""]
        df = df[~df["text"].str.match(r'^(\|\| ?)+$')]
        
        if df.empty:
            print(f"[WARN] No valid text to embed for {user_id}")
            users_collection.update_one(
                {"email": user_id},
                {"$set": {"embedding_status": "failed"}}
            )
            return
        
        # Compute embeddings
        print(f"[INFO] Computing embeddings for {len(df)} medicines...")
        embeddings = embed_model.encode(df["text"].tolist(), show_progress_bar=False)
        embeddings = safe_numpy(embeddings)
        
        vectors = np.array(embeddings, dtype=np.float32)
        dim = vectors.shape[1]
        
        # Create FAISS index
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        
        # Get next version
        version = get_next_embedding_version(user_id)
        filename = f"faiss_v{version}.index"
        
        # Save to OneDrive with versioning
        print(f"[INFO] Uploading {filename} to OneDrive...")
        success = save_faiss_to_onedrive(
            user_id, index, df, dim, version, current_hash
        )
        
        if not success:
            print(f"[ERROR] Failed to save embeddings for {user_id}")
            users_collection.update_one(
                {"email": user_id},
                {"$set": {"embedding_status": "failed"}}
            )
            return
        
        # Update user status
        now = datetime.utcnow()
        users_collection.update_one(
            {"email": user_id},
            {
                "$set": {
                    "embedding_status": "ready",
                    "embedding_version": version,
                    "last_embedding_update": now
                }
            }
        )
        
        print(f"[OK] Embedding worker completed for {user_id} (v{version})")
        
    except Exception as e:
        print(f"[ERROR] Embedding worker failed: {str(e)}")
        users_collection.update_one(
            {"email": user_id},
            {"$set": {"embedding_status": "failed"}}
        )
        import traceback
        traceback.print_exc()

def save_faiss_to_onedrive(user_id: str, faiss_index, medicine_df, embedding_dim: int, version: int, content_hash: str):
    """
    Save FAISS index to OneDrive with versioning.
    File naming: faiss_v{version}.index
    Never overwrites - only appends new versions.
    """
    try:
        print(f"[INFO] Saving FAISS v{version} to OneDrive...")
        access_token = get_cached_access_token()
        
        # Get user folder
        folder_id = get_or_create_user_folder(access_token, user_id)
        
        # Save FAISS index to temporary file
        filename = f"faiss_v{version}.index"
        with tempfile.NamedTemporaryFile(suffix='.index', delete=False) as tmp_index:
            faiss.write_index(faiss_index, tmp_index.name)
            tmp_index_path = tmp_index.name
        
        # Upload with versioned filename
        upload_faiss(access_token, folder_id, tmp_index_path, filename)
        os.unlink(tmp_index_path)
        
        # Save metadata to MongoDB
        meta_entry = {
            "userId": user_id,
            "embedding_version": version,
            "faiss_file": filename,
            "hash": content_hash,
            "embedding_dim": embedding_dim,
            "num_medicines": len(medicine_df),
            "model": "intfloat/multilingual-e5-base",
            "createdAt": datetime.utcnow(),
            "onedrive_folder_id": folder_id
        }
        
        embedding_meta_collection.insert_one(meta_entry)
        
        print(f"[OK] Saved FAISS v{version} for {user_id}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save FAISS: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def load_user_embedding_status(user_id: str) -> dict:
    """
    Get embedding status and latest version for a user.
    Returns status, version, and path to latest FAISS file.
    """
    user = users_collection.find_one({"email": user_id})
    if not user:
        return {"status": "not_found", "version": 0}
    
    status = user.get("embedding_status", "pending")
    version = user.get("embedding_version", 0)
    
    # Get latest metadata
    meta = embedding_meta_collection.find_one(
        {"userId": user_id},
        sort=[("embedding_version", -1)]
    )
    
    return {
        "status": status,
        "version": version,
        "metadata": clean_objectid(meta) if meta else None,
        "last_update": user.get("last_embedding_update")
    }





# ================= SEARCH =================
def collect_data_for(query: str, mail: str, k: int = 3):
    """
    Search medicines for a specific user using FAISS embeddings.
    Requires embeddings to be ready (status = "ready").
    """
    # Get user
    user = users_collection.find_one({"email": mail})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    # Check embedding status
    embedding_status = load_user_embedding_status(mail)
    
    if embedding_status["status"] != "ready":
        raise HTTPException(
            status_code=503,
            detail=f"Embeddings not ready. Status: {embedding_status['status']}"
        )
    
    # Get latest metadata and load FAISS from OneDrive
    meta = embedding_status["metadata"]
    if not meta:
        raise HTTPException(status_code=404, detail="No embeddings found")
    
    version = meta["embedding_version"]
    faiss_file = meta["faiss_file"]
    
    # Download FAISS index from OneDrive
    access_token = get_cached_access_token()
    folder_id = get_or_create_user_folder(access_token, mail)
    
    with tempfile.NamedTemporaryFile(suffix='.index', delete=False) as tmp:
        tmp_path = tmp.name
    
    if not download_file(access_token, folder_id, faiss_file, tmp_path):
        raise HTTPException(status_code=500, detail="Failed to load embeddings")
    
    # Load FAISS index
    faiss_index = faiss.read_index(tmp_path)
    os.unlink(tmp_path)
    
    # Encode query and search
    vec = embed_model.encode(query)
    vec = safe_numpy(vec)
    
    svec = np.array(vec, dtype=np.float32).reshape(1, -1)
    distances, indices = faiss_index.search(svec, k)
    
    # Fetch medicines from MongoDB
    medicines = get_user_medicines(mail)
    medicine_list = [medicines[i] for i in indices[0] if i < len(medicines)]
    
    # Clean and return
    result_list = []
    for med in medicine_list:
        # Drop unnecessary fields
        med.pop("text", None)
        med.pop("embedding", None)
        med.pop("status_import", None)
        result_list.append(med)
    
    return result_list

# ================= ROUTES =================
@app.get("/")
def root():
    return {"status": "running", "service": "Aushadhi 360 API"}

# ---------- LOGIN (No embedding build - only check status) ----------
@app.post("/login")
def login(
    background_tasks: BackgroundTasks,
    mail: str = "demo@aushadhi360.com",
    password: str = "demo123"
):
    """
    Login endpoint - NO EMBEDDING BUILD.
    Only validates credentials and returns status.
    Embeddings are built by background workers.
    """
    user = users_collection.find_one({"email": mail, "password": password})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check embedding status
    embedding_status = load_user_embedding_status(mail)
    
    # Queue embedding rebuild if needed
    if embedding_status["status"] != "ready":
        background_tasks.add_task(embedding_worker, mail)
    
    return {
        "status": "success",
        "message": "Login successful",
        "embedding_status": embedding_status["status"],
        "embedding_version": embedding_status["version"]
    }

# ---------- EMBEDDING STATUS ----------
@app.get("/embeddings/status/{user_id}")
def get_embedding_status(user_id: str, password: str):
    """
    Get embedding build status for a user.
    Returns: status (pending|ready|failed), version, last_update
    """
    user = users_collection.find_one({"email": user_id, "password": password})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    status = load_user_embedding_status(user_id)
    return status

# ---------- REBUILD EMBEDDINGS (Auto on Medicine Change) ----------
@app.post("/embeddings/rebuild/{user_id}")
def rebuild_embeddings(user_id: str, background_tasks: BackgroundTasks):
    """
    Trigger embedding rebuild for a user.
    Automatically detects medicine changes via SHA256 hash.
    If medicines haven't changed: Skips rebuild (0 seconds)
    If medicines changed: Rebuilds embeddings (30-60 seconds)
    
    No admin key needed - anyone can call this.
    Safe because hash detection prevents unnecessary rebuilds.
    """
    user = users_collection.find_one({"email": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Mark as pending
    users_collection.update_one(
        {"email": user_id},
        {"$set": {"embedding_status": "pending"}}
    )
    
    # Queue background worker
    background_tasks.add_task(embedding_worker, user_id)
    
    return {
        "status": "queued",
        "message": f"Embedding rebuild queued for {user_id}. Will skip if medicines unchanged.",
        "user_id": user_id
    }


# ---------- GET MEDICINES ----------
@app.get("/get_medicines")
def get_medicines(
    query: str = "fever and cough",
    mail: str = "demo@aushadhi360.com",
    password: str = "demo123"
):
    """
    Get medicines based on user query.
    Only returns medicines from the requesting user's collection.
    Requires embeddings to be in "ready" status.
    """
    if not mail or not password:
        raise HTTPException(status_code=400, detail="mail and password query params are required")
    
    user = users_collection.find_one({"email": mail, "password": password})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    groq_api_key = user.get("groqKeyAssist")
    if not groq_api_key:
        raise HTTPException(status_code=400, detail="groqKeyAssist is required for this user")
    
    # Get similar medicines
    medicines = collect_data_for(query, mail)
    
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

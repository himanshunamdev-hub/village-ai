
import os
import re
import uuid
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, Depends, HTTPException, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, auth as firebase_auth

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("DATABASE_PATH", BASE_DIR / "village_ai.db"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
ALLOWED_IMAGE = {"image/jpeg", "image/png", "image/webp", "image/gif"}
ALLOWED_VIDEO = {"video/mp4", "video/webm", "video/quicktime"}

app = FastAPI(title="Village AI Backend", version="1.0.0")

# For local development. In production, set ALLOWED_ORIGINS to your real domains.
origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


# -----------------------------
# Firebase Admin initialization
# -----------------------------
firebase_initialized = False
service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()

try:
    if service_account_path:
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred)
        firebase_initialized = True
    else:
        print("WARNING: FIREBASE_SERVICE_ACCOUNT_JSON is not set.")
except Exception as exc:
    print("WARNING: Firebase Admin initialization failed:", exc)


# -----------------------------
# SQLite helpers
# -----------------------------
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            firebase_uid TEXT NOT NULL UNIQUE,
            email TEXT,
            display_name TEXT,
            phone TEXT,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            photo_url TEXT,
            is_premium INTEGER NOT NULL DEFAULT 0,
            blue_tick INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS private_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_uid TEXT NOT NULL,
            receiver_uid TEXT NOT NULL,
            message TEXT,
            media_url TEXT,
            media_type TEXT,
            media_name TEXT,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_messages_pair
        ON private_messages(sender_uid, receiver_uid, created_at);

        CREATE TABLE IF NOT EXISTS premium_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            firebase_uid TEXT NOT NULL,
            note TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ai_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            firebase_uid TEXT,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    # Lightweight migration for databases created by older versions.
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "phone" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN phone TEXT")
    except Exception as exc:
        print("WARNING: users table migration failed:", exc)
    conn.commit()
    conn.close()


init_db()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def safe_username(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_]", "", value)
    return value[:30]


def make_unique_username(display_name: Optional[str], uid: str) -> str:
    base = safe_username(display_name or "user")
    if len(base) < 3:
        base = "user"

    conn = db()
    try:
        candidate = base
        n = 1
        while conn.execute(
            "SELECT 1 FROM users WHERE username = ? COLLATE NOCASE",
            (candidate,),
        ).fetchone():
            suffix = str(n)
            candidate = (base[: 30 - len(suffix)] + suffix)[:30]
            n += 1
        return candidate
    finally:
        conn.close()


def user_public(row):
    if not row:
        return None
    return {
        "id": row["id"],
        "username": row["username"],
        "display_name": row["display_name"] or row["username"],
        "photo_url": row["photo_url"],
        "email": row["email"],
        "phone": row["phone"],
        "is_premium": bool(row["is_premium"]),
        "blue_tick": bool(row["blue_tick"]),
        "verified": bool(row["blue_tick"]),
    }


def get_user_by_uid(uid: str):
    conn = db()
    row = conn.execute("SELECT * FROM users WHERE firebase_uid=?", (uid,)).fetchone()
    conn.close()
    return row


def get_user_by_username(username: str):
    conn = db()
    row = conn.execute(
        "SELECT * FROM users WHERE username=? COLLATE NOCASE", (username.strip(),)
    ).fetchone()
    conn.close()
    return row


# -----------------------------
# Firebase authentication
# -----------------------------
def current_uid(authorization: Optional[str] = Header(default=None)) -> str:
    if not firebase_initialized:
        raise HTTPException(503, "Firebase Admin is not configured on the server.")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Login required.")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(401, "Invalid authentication token.")

    try:
        decoded = firebase_auth.verify_id_token(token)
        return decoded["uid"]
    except Exception:
        raise HTTPException(401, "Invalid or expired Firebase login.")


# -----------------------------
# Models
# -----------------------------
class AIRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    language: str = "hi"


class UsernameRequest(BaseModel):
    username: str = Field(min_length=3, max_length=30)


class SyncProfileRequest(BaseModel):
    display_name: Optional[str] = Field(default=None, max_length=100)
    phone: Optional[str] = Field(default=None, max_length=25)
    username: Optional[str] = Field(default=None, min_length=3, max_length=30)


class PrivateMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)


class PremiumApplicationRequest(BaseModel):
    note: str = Field(default="", max_length=2000)


# -----------------------------
# Health
# -----------------------------
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "Village AI",
        "firebase_configured": firebase_initialized,
        "ai_configured": bool(os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")),
    }


# -----------------------------
# Firebase user -> local user
# -----------------------------
@app.post("/api/auth/sync")
def sync_user(payload: SyncProfileRequest = None, uid: str = Depends(current_uid)):
    payload = payload or SyncProfileRequest()
    try:
        fb_user = firebase_auth.get_user(uid)
    except Exception:
        raise HTTPException(401, "Firebase user not found.")

    existing = get_user_by_uid(uid)
    now = now_iso()

    desired_username = safe_username(payload.username or "") if payload.username else None
    desired_name = (payload.display_name or "").strip() or None
    desired_phone = (payload.phone or "").strip() or None

    if desired_username:
        if len(desired_username) < 3:
            raise HTTPException(400, "Username must have at least 3 letters/numbers.")
        conn = db()
        conflict = conn.execute(
            "SELECT firebase_uid FROM users WHERE username=? COLLATE NOCASE AND firebase_uid<>?",
            (desired_username, uid),
        ).fetchone()
        conn.close()
        if conflict:
            raise HTTPException(409, "This username is already taken. Please choose another one.")

    if existing:
        conn = db()
        conn.execute(
            """
            UPDATE users
            SET email=?, display_name=?, phone=?, username=COALESCE(?, username), photo_url=?, updated_at=?
            WHERE firebase_uid=?
            """,
            (
                fb_user.email or existing["email"],
                desired_name or fb_user.display_name or existing["display_name"],
                desired_phone or existing["phone"],
                desired_username,
                fb_user.photo_url or existing["photo_url"],
                now,
                uid,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE firebase_uid=?", (uid,)).fetchone()
        conn.close()
        return {"user": user_public(row)}

    username = desired_username or make_unique_username(desired_name or fb_user.display_name, uid)
    conn = db()
    try:
        conn.execute(
            """
            INSERT INTO users
            (firebase_uid,email,display_name,phone,username,photo_url,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                uid,
                fb_user.email,
                desired_name or fb_user.display_name,
                desired_phone,
                username,
                fb_user.photo_url,
                now,
                now,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE firebase_uid=?", (uid,)).fetchone()
    except sqlite3.IntegrityError:
        conn.rollback()
        raise HTTPException(409, "This username is already taken. Please choose another one.")
    finally:
        conn.close()
    return {"user": user_public(row)}


# -----------------------------
# Unique username
# -----------------------------
@app.put("/api/profile/username")
def change_username(payload: UsernameRequest, uid: str = Depends(current_uid)):
    username = safe_username(payload.username)

    if len(username) < 3:
        raise HTTPException(400, "Username must have at least 3 letters/numbers.")

    conn = db()
    conflict = conn.execute(
        """
        SELECT firebase_uid FROM users
        WHERE username=? COLLATE NOCASE AND firebase_uid<>?
        """,
        (username, uid),
    ).fetchone()

    if conflict:
        conn.close()
        raise HTTPException(409, "This username is already taken.")

    conn.execute(
        "UPDATE users SET username=?, updated_at=? WHERE firebase_uid=?",
        (username, now_iso(), uid),
    )
    conn.commit()
    conn.close()
    return {"username": username}


# -----------------------------
# Username availability
# -----------------------------
@app.get("/api/users/check-username")
def check_username(username: str):
    clean = safe_username(username)
    if len(clean) < 3:
        return {"available": False, "username": clean, "message": "Username must be at least 3 characters."}

    conn = db()
    row = conn.execute(
        "SELECT 1 FROM users WHERE username=? COLLATE NOCASE", (clean,)
    ).fetchone()
    conn.close()

    if row:
        return {"available": False, "username": clean, "message": "This username is already taken."}
    return {"available": True, "username": clean, "message": "This username is available."}


# -----------------------------
# User search
# -----------------------------
@app.get("/api/users/search")
def search_users(q: str, uid: str = Depends(current_uid)):
    q = q.strip().lower()

    if len(q) < 2:
        return {"users": []}

    conn = db()

    rows = conn.execute(
        """
        SELECT *
        FROM users
        WHERE lower(username) LIKE ?
           OR lower(COALESCE(display_name, '')) LIKE ?
        ORDER BY
            CASE
                WHEN lower(username) LIKE ? THEN 0
                ELSE 1
            END,
            username
        LIMIT 20
        """,
        (
            f"%{q}%",
            f"%{q}%",
            f"{q}%",
        ),
    ).fetchall()

    conn.close()

    return {
        "users": [
            user_public(row)
            for row in rows
            if row["firebase_uid"] != uid
        ]
    }

# -----------------------------
# Private text chat
# -----------------------------
@app.post("/api/users/{username}/message")
def send_private_message(
    username: str,
    payload: PrivateMessageRequest,
    uid: str = Depends(current_uid),
):
    target = get_user_by_username(username)
    if not target:
        raise HTTPException(404, "User not found.")

    if target["firebase_uid"] == uid:
        raise HTTPException(400, "You cannot message yourself.")

    conn = db()
    conn.execute(
        """
        INSERT INTO private_messages
        (sender_uid,receiver_uid,message,media_url,media_type,media_name,created_at)
        VALUES (?,?,?,?,?,?,?)
        """,
        (uid, target["firebase_uid"], payload.message.strip(), None, None, None, now_iso()),
    )
    conn.commit()
    message_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()

    return {"ok": True, "message_id": message_id}


# -----------------------------
# Private image/video upload
# -----------------------------
@app.post("/api/users/{username}/media")
async def send_private_media(
    username: str,
    file: UploadFile = File(...),
    message: str = Form(default=""),
    uid: str = Depends(current_uid),
):
    target = get_user_by_username(username)
    if not target:
        raise HTTPException(404, "User not found.")
    if target["firebase_uid"] == uid:
        raise HTTPException(400, "You cannot send media to yourself.")

    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_IMAGE and content_type not in ALLOWED_VIDEO:
        raise HTTPException(400, "Only image and video files are allowed.")

    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")

    ext = Path(file.filename or "").suffix.lower()
    if not ext:
        ext = ".bin"

    media_id = uuid.uuid4().hex
    stored_name = f"{media_id}{ext}"
    destination = UPLOAD_DIR / stored_name
    destination.write_bytes(data)

    media_type = "image" if content_type in ALLOWED_IMAGE else "video"
    media_url = f"/uploads/{stored_name}"

    conn = db()
    conn.execute(
        """
        INSERT INTO private_messages
        (sender_uid,receiver_uid,message,media_url,media_type,media_name,created_at)
        VALUES (?,?,?,?,?,?,?)
        """,
        (
            uid,
            target["firebase_uid"],
            message.strip() or None,
            media_url,
            media_type,
            file.filename or stored_name,
            now_iso(),
        ),
    )
    conn.commit()
    message_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()

    return {
        "ok": True,
        "message_id": message_id,
        "media_url": media_url,
        "media_type": media_type,
        "media_name": file.filename,
    }


# -----------------------------
# Private message history
# -----------------------------
@app.get("/api/users/{username}/messages")
def private_messages(username: str, uid: str = Depends(current_uid)):
    target = get_user_by_username(username)
    if not target:
        raise HTTPException(404, "User not found.")

    conn = db()
    rows = conn.execute(
        """
        SELECT id,sender_uid,receiver_uid,message,media_url,media_type,media_name,created_at
        FROM private_messages
        WHERE (sender_uid=? AND receiver_uid=?)
           OR (sender_uid=? AND receiver_uid=?)
        ORDER BY id ASC
        LIMIT 500
        """,
        (uid, target["firebase_uid"], target["firebase_uid"], uid),
    ).fetchall()
    conn.close()

    return {
        "user": user_public(target),
        "messages": [dict(row) for row in rows],
    }


# -----------------------------
# Premium / Blue Tick
# -----------------------------
@app.post("/api/premium/apply")
def premium_apply(
    payload: PremiumApplicationRequest,
    uid: str = Depends(current_uid),
):
    conn = db()
    existing = conn.execute(
        """
        SELECT id FROM premium_applications
        WHERE firebase_uid=? AND status='pending'
        """,
        (uid,),
    ).fetchone()

    if existing:
        conn.close()
        return {"ok": True, "status": "already_pending"}

    conn.execute(
        """
        INSERT INTO premium_applications(firebase_uid,note,status,created_at)
        VALUES (?,?,?,?)
        """,
        (uid, payload.note.strip(), "pending", now_iso()),
    )
    conn.commit()
    conn.close()

    # Application URL intentionally remains empty for now.
    return {
        "ok": True,
        "status": "pending",
        "application_url": os.getenv("PREMIUM_APPLICATION_URL", ""),
    }


# -----------------------------
# AI
# -----------------------------
def load_local_knowledge():
    knowledge_file = BASE_DIR / "data" / "local_knowledge.txt"
    if not knowledge_file.exists():
        return ""
    return knowledge_file.read_text(encoding="utf-8")[:50000]


async def ask_gemini(message: str, language: str):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

    if not api_key:
        raise HTTPException(
            503,
            "AI is not configured. Add GEMINI_API_KEY to the backend .env file.",
        )

    local_info = load_local_knowledge()
    system = f"""
You are Village AI, a helpful AI assistant made for village communities.
Answer in the user's requested language: {language}.
Use the local information below when it is relevant.
Never invent local facts. If local information is missing, clearly say that
you do not have verified local information instead of guessing.
Be friendly, simple, practical and concise.

VERIFIED LOCAL INFORMATION:
{local_info}
""".strip()

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": message}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1200},
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, json=payload)

    if response.status_code >= 400:
        detail = response.text[:1000]
        raise HTTPException(502, f"AI provider error: {detail}")

    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(502, "AI returned an empty response.")


@app.post("/api/chat")
async def ai_chat(payload: AIRequest, uid: Optional[str] = None):
    # Authentication is intentionally optional for AI chat.
    answer = await ask_gemini(payload.message.strip(), payload.language)

    if uid:
        conn = db()
        conn.execute(
            "INSERT INTO ai_messages(firebase_uid,role,message,created_at) VALUES (?,?,?,?)",
            (uid, "user", payload.message.strip(), now_iso()),
        )
        conn.execute(
            "INSERT INTO ai_messages(firebase_uid,role,message,created_at) VALUES (?,?,?,?)",
            (uid, "assistant", answer, now_iso()),
        )
        conn.commit()
        conn.close()

    return {"response": answer}


# -----------------------------
# Simple admin endpoint for Blue Tick
# -----------------------------
def require_admin(x_admin_key: Optional[str] = Header(default=None)):
    admin_key = os.getenv("ADMIN_KEY", "").strip()
    if not admin_key or x_admin_key != admin_key:
        raise HTTPException(403, "Admin access denied.")


@app.post("/api/admin/users/{username}/blue-tick")
def set_blue_tick(
    username: str,
    enabled: bool = True,
    _: None = Depends(require_admin),
):
    conn = db()
    cur = conn.execute(
        "UPDATE users SET blue_tick=?, is_premium=?, updated_at=? WHERE username=? COLLATE NOCASE",
        (1 if enabled else 0, 1 if enabled else 0, now_iso(), username),
    )
    conn.commit()
    conn.close()

    if cur.rowcount == 0:
        raise HTTPException(404, "User not found.")

    return {"ok": True, "username": username, "blue_tick": enabled}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
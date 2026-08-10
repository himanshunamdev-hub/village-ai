
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from openai import OpenAI

import os


# =========================================================
# VILLAGE AI
# Python Backend - OpenAI Connected Version
# =========================================================

load_dotenv()


# =========================================================
# RATE LIMITING
# =========================================================

limiter = Limiter(
    key_func=get_remote_address
)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Village AI",
    version="1.0.0"
)


# Register rate-limit exception handler
app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",

        # Production me apna domain yahan add karo:
        # "https://yourdomain.com",
    ],

    allow_credentials=True,

    allow_methods=[
        "GET",
        "POST"
    ],

    allow_headers=[
        "Content-Type",
        "Authorization"
    ],
)


# =========================================================
# API KEYS
# =========================================================

# IMPORTANT:
# API key directly code me mat likhna.
# .env file me rakho:
#
# OPENAI_API_KEY=your_api_key_here
#
# Optional backup keys:
# AI_API_KEY_2=your_second_key
# AI_API_KEY_3=your_third_key
# AI_API_KEY_4=your_fourth_key

API_KEYS = [
    os.getenv("OPENAI_API_KEY", ""),
    os.getenv("AI_API_KEY_2", ""),
    os.getenv("AI_API_KEY_3", ""),
    os.getenv("AI_API_KEY_4", ""),
]


# =========================================================
# OPENAI MODEL
# =========================================================

MODEL_NAME = os.getenv(
    "OPENAI_MODEL",
    "gpt-5-mini"
)


# =========================================================
# CHAT REQUEST
# =========================================================

class ChatRequest(BaseModel):

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000
    )

    language: str = Field(
        default="en",
        pattern="^(en|hi)$"
    )


# =========================================================
# CHAT RESPONSE
# =========================================================

class ChatResponse(BaseModel):

    response: str

    success: bool

    key_used: int | None = None


# =========================================================
# AI FUNCTION
# =========================================================

def ask_ai(
    message: str,
    language: str,
    api_key: str
):

    if not api_key:
        return None


    # Create OpenAI client using the selected API key
    client = OpenAI(
        api_key=api_key
    )


    # =====================================================
    # LANGUAGE INSTRUCTION
    # =====================================================

    if language == "hi":

        instructions = (
            "You are Village AI, a helpful AI assistant "
            "for people in villages and rural communities. "
            "Answer the user in Hindi. "
            "Use simple, clear and practical Hindi. "
            "If technical English words are necessary, "
            "you may include them in English."
        )

    else:

        instructions = (
            "You are Village AI, a helpful AI assistant "
            "for people in villages and rural communities. "
            "Answer clearly, simply and practically."
        )


    # =====================================================
    # OPENAI RESPONSES API
    # =====================================================

    response = client.responses.create(

        model=MODEL_NAME,

        instructions=instructions,

        input=message
    )


    # =====================================================
    # GET TEXT RESPONSE
    # =====================================================

    answer = response.output_text


    if not answer:
        return None


    return answer.strip()


# =========================================================
# CHAT ENDPOINT
# =========================================================

@app.post(
    "/api/chat",
    response_model=ChatResponse
)
@limiter.limit("20/minute")
async def chat(
    request: Request,
    chat_request: ChatRequest
):

    # =====================================================
    # CLEAN MESSAGE
    # =====================================================

    message = chat_request.message.strip()


    # =====================================================
    # EMPTY MESSAGE
    # =====================================================

    if not message:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )


    # =====================================================
    # LANGUAGE VALIDATION
    # =====================================================

    if chat_request.language not in ["en", "hi"]:

        raise HTTPException(
            status_code=400,
            detail="Language must be 'en' or 'hi'."
        )


    # =====================================================
    # CHECK API KEYS
    # =====================================================

    if not any(API_KEYS):

        if chat_request.language == "hi":

            return ChatResponse(
                response=(
                    "नमस्ते! 🌾\n\n"
                    "Village AI backend चल रहा है, "
                    "लेकिन OpenAI API key configure नहीं की गई है।\n\n"
                    "कृपया अपनी .env file में "
                    "OPENAI_API_KEY सेट करें।"
                ),
                success=False,
                key_used=None
            )

        else:

            return ChatResponse(
                response=(
                    "Hello! 🌾\n\n"
                    "Village AI backend is running, "
                    "but no OpenAI API key has been configured.\n\n"
                    "Please add OPENAI_API_KEY to your .env file."
                ),
                success=False,
                key_used=None
            )


    # =====================================================
    # TRY API KEYS ONE BY ONE
    # =====================================================

    for index, api_key in enumerate(
        API_KEYS,
        start=1
    ):

        if not api_key:
            continue


        try:

            result = ask_ai(
                message,
                chat_request.language,
                api_key
            )


            if result:

                return ChatResponse(
                    response=result,
                    success=True,
                    key_used=index
                )


        except Exception as error:

            # API key ko kabhi log mat karo.
            print(
                f"AI API Key {index} failed: {error}"
            )

            continue


    # =====================================================
    # ALL KEYS FAILED
    # =====================================================

    raise HTTPException(
        status_code=503,
        detail=(
            "AI service is currently unavailable. "
            "Please check your API key, model name, "
            "API account/billing and server logs."
        )
    )


# =========================================================
# HOME PAGE
# =========================================================

@app.get("/")
async def home():

    if not os.path.exists("index.html"):

        return {
            "status": "ok",
            "app": "Village AI",
            "message": "Backend is running."
        }

    return FileResponse(
        "index.html"
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
async def health():

    configured_keys = sum(
        1 for key in API_KEYS if key
    )

    return {
        "status": "ok",
        "app": "Village AI",
        "ai_keys_configured": configured_keys,
        "model": MODEL_NAME
    }


# =========================================================
# SERVER
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )

# back.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import time
import logging

from openai import OpenAI

# ---- config ----
MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT")  # 可為 None
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY as an environment variable")

# 重要：gpt-4o-* 走 Responses API，且帶上 project
client = OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT)

app = FastAPI()
START_TS = time.time()

# CORS：先全部放行方便測試
allow_origins = os.getenv("CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allow_origins.split(",")] if allow_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Schemas ----------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "history_limit": 8,
        "cors": [allow_origins] if isinstance(allow_origins, str) else allow_origins,
        "project_set": bool(OPENAI_PROJECT),
        "uptime_sec": int(time.time() - START_TS),
    }

@app.post("/chat")
def chat(req: ChatRequest):
    """
    使用 Responses API 呼叫 gpt-4o-mini
    把聊天訊息串成單一文字輸入（Demo 夠用）
    """
    try:
        # 將 messages 簡單串接（也可以自己定更好的格式）
        parts = []
        for m in req.messages:
            r = m.role.strip().lower()
            if r not in ("user", "system", "assistant"):
                r = "user"
            parts.append(f"{r.upper()}: {m.content}")
        prompt = "\n".join(parts) + "\nASSISTANT:"

        resp = client.responses.create(
            model=MODEL,
            input=prompt,
            max_output_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        # 簡單取得輸出文字（OpenAI SDK 1.x 提供 output_text）
        reply_text = resp.output_text

        return {"reply": reply_text}

    except Exception as e:
        # 詳細錯誤打到 logs，前端回簡潔訊息
        logging.exception("OpenAI upstream error")
        return JSONResponse(
            status_code=502,
            content={"detail": f"Upstream error: {str(e)}"},
        )
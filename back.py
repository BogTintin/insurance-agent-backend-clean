# back.py
import os
import time
import logging
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from openai import APIConnectionError, RateLimitError

# -----------------------------
# 環境變數（請在 Render -> Environment 設定）
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY as an environment variable")

# 若你的金鑰是 sk-proj-...，建議一併提供 Project ID（OpenAI Dashboard -> Project -> ID: proj_xxx）
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT")  # 可選：proj_XXXX
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN")  # 可選：例如 https://你的前端.vercel.app

# -----------------------------
# OpenAI Client
# -----------------------------
client = OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT) if OPENAI_PROJECT else OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# FastAPI App & CORS
# -----------------------------
app = FastAPI(title="Insurance Agent MVP API")

# 上線後建議把 FRONTEND_ORIGIN 設為你的 Vercel 網域，否則就暫時放寬 *
allow_list = [FRONTEND_ORIGIN] if FRONTEND_ORIGIN else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 日誌
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

# -----------------------------
# 簡易節流（每 IP 每分鐘最多 N 次）
# -----------------------------
WINDOW_SECONDS, MAX_REQ = 60, 30
_bucket = {}
def _rate(ip: str):
    now = time.time()
    xs = [t for t in _bucket.get(ip, []) if now - t < WINDOW_SECONDS]
    if len(xs) >= MAX_REQ:
        raise HTTPException(429, "Too Many Requests. Please retry later.")
    xs.append(now)
    _bucket[ip] = xs

# -----------------------------
# 模型與生成控制（便宜好用版）
# -----------------------------
MODEL = "gpt-4o-mini"  # 便宜且夠用
MAX_TOKENS = 500       # 輸出上限（成本控制）
TEMPERATURE = 0.3
HISTORY_LIMIT = 8      # 只保留最近 N 則對話，控成本

# -----------------------------
# Schemas
# -----------------------------
class ChatMessage(BaseModel):
    role: Literal["system","user","assistant"]
    content: str

class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    reply: str

# -----------------------------
# System Prompt（已校正錯字/語意）
# -----------------------------
SYSTEM_PROMPT = (
    "你是一位壽險外勤業務員的 AI 助理，協助更快速完成工作並協助客戶規劃。\n"
    "請遵守：\n"
    "1) 先確認客戶條件是否足以回答（年齡、預算、保障需求）。\n"
    "2) 以台灣常見壽險／醫療／外幣／投資型（基金連結標的）商品的一般性說明作答，避免提供法規或理賠承諾。\n"
    "3) 回覆格式：重點結論一段 + 條列 3–5 點 + 下一步建議。\n"
    "4) 若資訊不足或不確定，請明確說明並列出需要補充的項目。\n"
    "5) 禁止捏造公司內規／條款；必要時建議參考官方文件或請專員協助。\n"
    "6) 可依市場概況與客戶條件，提供投資型保單標的配置的一般性觀點與風險揭露（非投資建議）。\n"
    "7) 可協助統整客戶既有保單之保障與理財概況，並依使用者指示提出建議。\n"
    "8) 可主動詢問是否需要與業務員進一步討論或分享建議。\n"
)

# -----------------------------
# 小工具：逾時 + 輕重試
# -----------------------------
def call_openai_with_retry(messages, retries: int = 1, timeout: int = 30):
    last_err = None
    for attempt in range(retries + 1):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=timeout,
            )
        except (APIConnectionError, RateLimitError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5)
                continue
            break
        except Exception as e:
            last_err = e
            break
    raise last_err

# -----------------------------
# Endpoints
# -----------------------------
START_AT = time.time()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "history_limit": HISTORY_LIMIT,
        "cors": allow_list,
        "project_set": bool(OPENAI_PROJECT),
        "uptime_sec": int(time.time() - START_AT),
    }

@app.post("/chat", response_model=ChatResponse)
def chat(req: Request, body: ChatRequest):
    # 節流
    ip = req.client.host if req.client else "unknown"
    _rate(ip)

    # 上下文截斷（控制成本/延遲）
    history = [m.model_dump() for m in body.messages][-HISTORY_LIMIT:]
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    try:
        r = call_openai_with_retry(msgs, retries=1, timeout=30)
        reply = (r.choices[0].message.content or "").strip()
        return ChatResponse(reply=reply or "（沒有產生內容）")
    except Exception as e:
        # 只回給前端友善字串；詳細錯誤寫到 log
        logger.exception("openai call failed: %s", e)
        raise HTTPException(status_code=503, detail="系統忙碌或供應商回應異常，請稍後再試。")
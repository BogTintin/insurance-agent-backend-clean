import os, time
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# 讀取環境變數（Render 會設）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY as an environment variable")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Insurance Agent MVP API")

# 先放寬，前端上線後再改白名單
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 極簡節流
WINDOW_SECONDS, MAX_REQ = 60, 30
_bucket = {}
def _rate(ip: str):
    now = time.time()
    xs = [t for t in _bucket.get(ip, []) if now - t < WINDOW_SECONDS]
    if len(xs) >= MAX_REQ:
        raise HTTPException(429, "Too Many Requests")
    xs.append(now); _bucket[ip] = xs

class ChatMessage(BaseModel):
    role: Literal["system","user","assistant"]
    content: str

class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    reply: str

SYSTEM_PROMPT = (
    "你是一位屬於凱基人壽的壽險外勤業務員助理 AI。請：\n"
    "協助外勤業務員能更快速的完成工作與協助客戶規劃與市場分析。\n"
    "1) 先確認客戶條件是否足以回答（年齡、預算、保障需求）。\n"
    "2) 回答以台灣壽險/醫療/外幣保單/投資型（基金連結標的）商品的一般性說明，避免提供法規或理賠承諾。\n"
    "3) 回覆格式：重點結論一段 + 條列 3–5 點 + 下一步建議。\n"
    "4) 若資訊不足或不確定，請明確說明並列出需要補充的項目。\n"
    "5) 禁止捏造公司內規/條款；必要時建議參考官方文件或請專員協助。\n"
    "6) 可依投資市場概況/標的表現等以及客戶資訊等堤關投資型保單隻基金投資組合或建議"
    "7) 協助統整客戶之際有保單之保障、理財概況、並依使用者指示提供建議"
    "8) 可以主動詢問是否需要與業務員分享與討論你的看法或建議"
)

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: Request, body: ChatRequest):
    _rate(req.client.host if req.client else "unknown")
    msgs = [{"role":"system","content":SYSTEM_PROMPT}] + [m.model_dump() for m in body.messages]
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, temperature=0.2)
        return ChatResponse(reply=r.choices[0].message.content.strip())
    except Exception as e:
        raise HTTPException(500, f"LLM error: {e}")
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from src.route import router as transcribe_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    print("🚀 Startup")

    try:
        yield
    finally:
        print("⚠️ Shutdown")


app = FastAPI(title="Phy backend", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["http://localhost:5173", "https://phy-react.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")


app.include_router(transcribe_router)


@app.get(path="/health", tags=["root"])
async def health() -> dict:
    return {"status": "ok"}

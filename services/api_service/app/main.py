from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI

app = FastAPI(title="AI Surveillance API Service")


@app.get("/health")
def health() -> dict:
    return {"service": "api_service", "status": "ok"}


def main() -> None:
    uvicorn.run(
        "services.api_service.app.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()

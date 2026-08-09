from contextlib import asynccontextmanager
from fastapi import FastAPI,Request
import time
from services.api_service.database import SQLiteDatabase
from services.api_service.routes import api_router
from services.api_service.websocket import websocket_router
from services.api_service.services.ml_client import MLClient
from shared.logging import configure_logging,get_logger
from shared.settings import ServiceSettings
settings=ServiceSettings.from_env();configure_logging(settings.log_level,"api-service");log=get_logger(__name__)
@asynccontextmanager
async def lifespan(app):
 log.info("API Service starting")
 db=SQLiteDatabase(settings.database_path);sqlite_ok=await db.connect();ml=MLClient(settings.ml_url);ml_ok=bool(await ml.health())
 app.state.database=db;app.state.ml_client=ml;app.state.dependencies={"sqlite":sqlite_ok,"ml_service":ml_ok};app.state.ml_status={"status":"unknown"};app.state.ml_metrics={}
 log.info("SQLite DB: %s",db.path);log.info("SQLite dependency: %s","available" if sqlite_ok else "unavailable");log.info("ML Service dependency: %s","available" if ml_ok else "unavailable")
 yield
 await db.close();log.info("API Service stopped")
app=FastAPI(title="AI Surveillance API",version="1.0.0",lifespan=lifespan);app.include_router(api_router,prefix="/api/v1");app.include_router(websocket_router)
@app.middleware("http")
async def access_log(request:Request,call_next):
 started=time.perf_counter();response=await call_next(request);elapsed=(time.perf_counter()-started)*1000
 if not request.url.path.startswith("/api/v1/internal/ml/") or response.status_code>=400:log.info("%s %s -> %d %.0fms",request.method,request.url.path,response.status_code,elapsed)
 return response
def main():
 import uvicorn;uvicorn.run("services.api_service.app:app",host=settings.api_host,port=settings.api_port,reload=False,access_log=False)
if __name__=="__main__":main()

from fastapi import APIRouter

from .health import router as health_router
from .cameras import router as cameras_router
from .enrollment import router as enrollment_router
from .events import router as events_router
from .heatmaps import router as heatmaps_router
from .persons import router as persons_router
from .settings import router as settings_router
from .system import router as system_router
from .internal import router as internal_router

api_router = APIRouter()
api_router.include_router(health_router)
for router in (cameras_router,enrollment_router,events_router,heatmaps_router,persons_router,settings_router,system_router):
    api_router.include_router(router)
api_router.include_router(internal_router)

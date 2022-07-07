from fastapi import APIRouter

from api_v1.endpoints import login, users, utils, test, models

api_router = APIRouter()
api_router.include_router(test.router)
api_router.include_router(login.router, tags=["login"])
api_router.include_router(models.router, tags=["model"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(utils.router, prefix="/utils", tags=["utils"])
# api_router.include_router(items.router, prefix="/items", tags=["items"])

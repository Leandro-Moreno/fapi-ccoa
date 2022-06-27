from fastapi import FastAPI, Path
from fastapi.responses import RedirectResponse
from starlette.middleware.cors import CORSMiddleware
from core.config import settings
from api import api_router

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def roots():
    return RedirectResponse("https://ccoa.leandromoreno.com")


@app.get("/hello/{name}")
async def say_hellos(name: str):
    return {"message": f"Hello {name}"}

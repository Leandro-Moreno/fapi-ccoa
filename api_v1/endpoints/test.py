from typing import Any, List

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic.networks import EmailStr
from sqlalchemy.orm import Session

import crud, models, schemas
from api_v1 import deps
from core.config import settings
from utils import send_new_account_email

router = APIRouter()


@router.get("/")
async def test():
    return {"message": "Holi"}
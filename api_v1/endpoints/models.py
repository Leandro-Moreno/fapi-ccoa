from typing import Any, List

from fastapi import APIRouter, Body, Depends, HTTPException, Form
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from fastapi.encoders import jsonable_encoder
from pydantic.networks import EmailStr
from sqlalchemy.orm import Session
from sklearn.ensemble import RandomForestClassifier

import crud, models, schemas
from api_v1 import deps
from core.config import settings
from utils import send_new_account_email
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder

router = APIRouter()


@router.post("/model")
async def models(
        LCIIU: str = Form(),
        desc_organizacion: str = Form(),
        import_export: str = Form(),
        ciudad: str = Form(),
        activos: float = Form(),
        pasivos: float = Form(),
        ingresosoperacionales: float = Form(),
        Sector: str = Form(),
        Tamaño_empresa: str = Form(),
        anio_creacion: int = Form(),
):
    # df_juridica = pd.read_excel('api_v1/endpoints/Data/df_juridica.xlsx')
    #
    # df_yX = df_juridica[['empresa_viva', 'LCIIU', 'desc_organizacion', 'import_export', 'ciudad', 'activos', 'pasivos',
    #                      'ingresosoperacionales', 'Sector', 'Tamaño_empresa', 'existencia_meses']]
    # # Drop NAs
    # df_yX = df_yX.dropna()
    #
    # # Set dtypes
    # df_yX = df_yX.astype({'empresa_viva': 'category',
    #                       'LCIIU': 'category',
    #                       'desc_organizacion': 'category',
    #                       'import_export': 'category',
    #                       'ciudad': 'category',
    #                       'Sector': 'category',
    #                       'Tamaño_empresa': 'category',
    #                       'existencia_meses': 'int'})
    anos_totales = 2022 - anio_creacion
    existencia_meses = anos_totales*12+1
    loaded = load('api_v1/endpoints/random_forest_model.joblib')
    categories_ohe = load('api_v1/endpoints/categories_ohe.joblib')
    new_data_encoder = load('api_v1/endpoints/new_data_encoder.joblib')
    df_yX = load('api_v1/endpoints/df_yX.joblib')
    new_data = pd.DataFrame(df_yX.iloc[206, 1:]).T.reset_index(drop=True)
    new_data['LCIIU'] = LCIIU
    new_data['desc_organizacion'] = desc_organizacion
    new_data['import_export'] = import_export
    new_data['ciudad'] = ciudad
    new_data['activos'] = activos
    new_data['pasivos'] = pasivos
    new_data['ingresosoperacionales'] = ingresosoperacionales
    new_data['Sector'] = Sector
    new_data['Tamaño_empresa'] = Tamaño_empresa
    new_data['existencia_meses'] = existencia_meses
    new_data['empresa_viva'] = 1
    new_data_X = pd.concat([df_yX.iloc[:, 1:], new_data])
    new_data_encoded = pd.DataFrame(new_data_encoder.toarray(),
                                    columns=categories_ohe)
    new_data_encoded = new_data_encoded.iloc[-1, :].to_frame().T
    print(new_data.head())
    new_data_ohe = pd.concat([
        new_data[['existencia_meses', 'activos', 'pasivos', 'ingresosoperacionales']].reset_index(drop=True),
        new_data_encoded.reset_index(drop=True)
    ],
        axis='columns')

    # Using Model for Predicting labels for new data
    new_data_predict = pd.DataFrame(loaded.predict(new_data_ohe), columns=['empresa_viva'])
    new_data_pred_prob = pd.DataFrame(loaded.predict_proba(new_data_ohe), columns=['PROB_0', 'PROB_1'])

    # Create dataframe with new data and results
    new_data_modeled = pd.concat([new_data, new_data_predict, new_data_pred_prob['PROB_1']], axis=1)
    print(new_data_modeled)
    return {"message": new_data_pred_prob['PROB_1']}

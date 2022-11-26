from typing import Optional
from enum import Enum

#pydantic
from pydantic import BaseModel
from pydantic import Field

#fastapi
from fastapi import FastAPI
from fastapi import Body, Query, Path, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import cv2

from tensorflow.keras.models import load_model

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sign_model = load_model("simple_cnn.h5")

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "You can find my sourcecode at: https://github.com/H0CHM31573R/pd_proyecto"}
    
@app.post("/model/sign")
async def analizar_imagen(image:UploadFile = File(...)):
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150), interpolation = cv2.INTER_AREA)
    img = img/255.0
    inp = img.reshape(1, 150, 150, 3)
    pred = sign_model.predict(inp)
    
    return {
        "prediction": pred.tolist(),
    }

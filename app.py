from typing import Optional
from enum import Enum

#pydantic
#from pydantic import BaseModel
#from pydantic import Field

#fastapi
from fastapi import FastAPI
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse 

from numpy import fromstring, uint8, reshape
from cv2 import imdecode, cvtColor, resize, IMREAD_COLOR, COLOR_BGR2RGB, INTER_AREA

from tensorflow.keras.models import load_model

app = FastAPI()
app.mount('/static', StaticFiles(directory='static',html=True))

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sign_model = load_model("simple_cnn.h5")

classes = ["A", "B", "C", "D", "del", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "nothing",
           "O", "P", "Q", "R", "S", "space", "T", "U", "V", "W", "X", "Y", "Z"]

@app.get("/", tags=["Root"])
async def read_root():
    # return {"Source": "You can find my source code at: https://github.com/H0CHM31573R/pd_proyecto", 
            # "API": "You can test my API at: http://ec2-3-82-149-136.compute-1.amazonaws.com/docs"}
    return FileResponse('index.html')
    
@app.post("/model/sign")
async def analizar_imagen(image:UploadFile = File(...)):
    contents = await image.read()
    nparr = fromstring(contents, uint8)
    
    img = imdecode(nparr, IMREAD_COLOR)
    img = cvtColor(img, COLOR_BGR2RGB)
    img = resize(img, (150, 150), interpolation = INTER_AREA)
    img = img/255.0
    inp = img.reshape(1, 150, 150, 3)
    pred = sign_model.predict(inp)
    pred_class = pred.argmax(axis=-1)
    
    return {
        "prediction": classes[pred_class],
    }

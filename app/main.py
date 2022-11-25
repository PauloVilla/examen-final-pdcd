from fastapi import Security, Depends, FastAPI, HTTPException
from fastapi.security.api_key import APIKeyQuery, APIKey
from starlette.status import HTTP_403_FORBIDDEN
from pydantic import BaseModel
import pandas as pd
import uvicorn
import pickle

API_KEY = "paulo_examen"
API_KEY_NAME = "password"

api_key_query = APIKeyQuery(name=API_KEY_NAME, auto_error=False)

app = FastAPI(title="Funding challenge",
              description="Esta API recibe un parámetro tipo body con 2",
              version="0.0.1")


def get_api_key(api_key_query: str = Security(api_key_query)):
    if api_key_query == API_KEY:
        return api_key_query
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Acceso denegado."
        )


class Inputs(BaseModel):
    Description: str
    Category: str
    Cost: float


@app.on_event("startup")
def load_model():
    global model_lr
    with open("./models/model_class.pickle", "rb") as openfile:
        model_lr = pickle.load(openfile)
    global vectorizer
    with open("./models/vectorizer.pickle", "rb") as openfile:
        vectorizer = pickle.load(openfile)
    global model_tf
    with open("./models/model_tf_idf.pickle", "rb") as openfile:
        model_tf = pickle.load(openfile)
    global transform
    transform = pd.read_excel("./models/Transform_subject.xlsx")
    transform = dict(transform.values)


@app.get("/secure_endpoint", tags=["test"])
async def get_open_api_endpoint(api_key: APIKey = Depends(get_api_key)):
    response = {'description': "You have access to this endpoint."}
    return response


@app.get("/api/v1/classify")
async def classify_found(inputs: Inputs, api_key: APIKey = Depends(get_api_key)):
    finals = str(inputs.Description)
    finals = [finals]
    bow = vectorizer.transform(finals).toarray()
    tfidf = model_tf.fit_transform(bow).toarray()
    tfidf = list(tfidf[0])
    cat = inputs.Category
    cat = transform[cat]
    tfidf.append(cat)
    tfidf.append(inputs.Cost)
    params = [tfidf]
    pred = model_lr.predict(params)
    dict_oscar = {0: 'Not founded.',
                  1: 'Founded.'}
    return {"Prediction": dict_oscar.get(pred[0]),
            'Description': "Prediction correctly executed"}


@app.get("/")
def home():
    return {"Desc": "Health Check"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

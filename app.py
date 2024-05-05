### Created by Ujjwal Solanki

import uvicorn
from fastapi import FastAPI, Request, Form

from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.components.prediction import Prediction
app = FastAPI()

# app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates/")


@app.get('/', response_class=HTMLResponse)
async def main(request: Request):
    pred = Prediction()

    data = pred.initiate_prediction()
    return templates.TemplateResponse('index.html', {'request': request, 'result': data})

# @app.get("/hello/{name}")
# async def helloworld(name):
#     return f"Hello World {name}"

#Below code is to make site more dynamic, 
#Get user input as stock name and then predict result
@app.post("/get_result")
async def get_results(request: Request, name:str = Form("...")):
    pred = Prediction()
    #change code and pass stock name to get result for perticular stock
    result = pred.initiate_prediction()
    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

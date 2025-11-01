import uvicorn
from fastapi import FastAPI, File, UploadFile
from backend.db_init import init_db
from backend.predict import predict_from_bytes

app = FastAPI(title='Skin AI')


@app.on_event('startup')
def startup_event():
    init_db()


@app.get('/health')
def health():
    return {"status": "OK"}


@app.post('/predict/')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    res = predict_from_bytes(contents)
    return res


if __name__ == '__main__':
    uvicorn.run('backend.main:app', host='0.0.0.0', port=8000, reload=True)

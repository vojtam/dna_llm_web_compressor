from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from coder import run_encoding
import logging

app = FastAPI(
    title="DNA Arithmetic Coding Compression",
    description="A simple DNA compression API",
    version="0.1"
)


origins = [
    'http://localhost:8000/encode',
    'http://localhost:8000',
    'http://127.0.0.1:8000/',
    'http://127.0.0.1:5173/' 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
	logging.error(f"{request}: {exc_str}")
	content = {'status_code': 10422, 'message': exc_str, 'data': None}
	return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

class LocusInput(BaseModel):
    locus: str

@app.post('/encode', description="Encode human GRCh38 DNA")
def encode(locus: LocusInput):
    percentage, complexity, lz_complexity = run_encoding(locus.locus)
    return {"percentage": percentage, "complexity": complexity, "lz_complexity": lz_complexity, "status_code":200}

@app.get('/')
def hello():
     return {"content": f"Hello from the API", "status_code": 200}
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from coder import run_encoding
import logging

app = FastAPI(
    title="DNA Arithmetic Coding Compression",
    description="A simple DNA compression API",
    version="0.1"
)

origins = [
    'http://localhost:3000',
    'http://localhost:5173'
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
    result = run_encoding(locus.locus)
    return {"content": f"The sequence was compressed by {result:0.3f}%", "status_code":200}



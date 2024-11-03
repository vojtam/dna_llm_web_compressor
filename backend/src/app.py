from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from coder import run_encoding
import logging
from coder import parse_locus_string
import polars as pl
import numpy as np

app = FastAPI(
    title="DNA Arithmetic Coding Compression",
    description="A simple DNA compression API",
    version="0.1"
)

bed_df = pl.read_csv("chr22_tokenized.bed", separator="\t", has_header=False, new_columns=["chr", "start", "end", "value", "index"])

def load_array_slice(filename, shape, dtype, start_row=None, end_row=None):
    """
    Load a slice of a memory-mapped numpy array.
    
    Parameters:
    filename: str - The file containing the array
    shape: tuple - The shape of the full array
    dtype: numpy.dtype - The data type of the array
    start_row: int - Starting row index (optional)
    end_row: int - Ending row index (optional)
    
    Returns:
    numpy.ndarray - The requested slice of the array
    """
    # Create memory map to file
    fp = np.memmap(filename, dtype=dtype, mode='r', shape=shape)
    
    # If no slice specified, return view of entire array
    if start_row is None and end_row is None:
        return fp
    
    # Return specific slice
    return fp[start_row:end_row].copy()

origins = [
    'http://localhost:8000/encode',
    'http://localhost:8000',
    'http://localhost:5173',
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
    chr, start, end = parse_locus_string(locus.locus)

    if chr == "22":
        filtered_df = bed_df.filter(
            (pl.col("chr") == int(chr)) &
            (pl.col("start") <= end) &  # Interval starts before or at the given end
            (pl.col("end") >= start)    # Interval ends after or at the given start
        )

        start_index = filtered_df.row(0, named=True)['index']
        end_index = filtered_df.tail(1).select("index").item() + 1

        cdfs = load_array_slice("chr22_cdfs.mmap", (8892283, 1001), np.uint16, start_index, end_index)
        input_ids = filtered_df["value"].to_list()
        percentage, complexity, lz_complexity = run_encoding(locus.locus, cdfs, input_ids)
    else:
        percentage, complexity, lz_complexity = run_encoding(locus.locus)
    return {"percentage": percentage, "complexity": complexity, "lz_complexity": lz_complexity, "status_code":200}

@app.get('/')
def hello():
     return {"content": f"Hello from the API", "status_code": 200}
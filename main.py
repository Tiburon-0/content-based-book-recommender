'''FastAPI application that loads the Production model from the Model Registry'''

import time
from pathlib import Path
from typing import List

import numpy as np
import wandb
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from db import log_retrieval, record_feedback
from recommender import TiburonBookRecommender

app = FastAPI(title="Tiburon Content-Based Book Recommender", version="1.0.0")

REGISTRY_PATH = 'wandb-registry-model/tiburon-book-recommender'
MODEL_FILE = 'tiburon_book_recommendation_model.pkl'
MODEL_ALIAS = 'production'

def load_production_model():
    artifact = wandb.Api().artifact(f'{REGISTRY_PATH}:{MODEL_ALIAS}')
    model_dir = artifact.download()
    model = TiburonBookRecommender.load(Path(model_dir) / MODEL_FILE)
    served_by = {
        'registry_version': artifact.version,
        'source_version': artifact.source_version,
        'digest': artifact.digest,
    }
    return model, served_by

model = None
served_by = None
load_error = None

try:
    model, served_by = load_production_model()
    print(f'Model loaded successfully: {served_by}')
except Exception as error:
    load_error = f'{REGISTRY_PATH}:{MODEL_ALIAS} -> {type(error).__name__}: {error}'
    print(f'{load_error}: Model is not loaded.')
    model = None

def require_model():
    '''Raises 503 when no model is loaded, so every endpoint fails identically'''
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f'Model is not loaded. {load_error}'
        )

# data validation of request body

class TextInput(BaseModel):
    text: str = Field(
        min_length=1,
        description="The query input used to retrieve the recommendations"
        )

    k: int = Field(
        default=10,
        ge = 1,
        le=50,
        description="How many recommendations to return"
    )

class BatchTextInput(BaseModel):
    text: List[str] = Field(
        min_length=1,
        description="A list of query inputs, one set of recommendations returned per entry"
    )

    k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="How many recommendations to return per query"
    )

class FeedbackInput(BaseModel):
    retrieval_id: str = Field(
        min_length=1,
        description="The retrieval_id handed back by /retrieve"
    )

    helpful: bool = Field(
        description="Thumbs up or thumbs down on that set of recommendations"
    )

@app.get("/")
def read_root():
    '''Service banner'''
    return {"message": app.title}

@app.get("/health")
def health_check():
    '''A simple endpoint to confirm that the API is running properly'''
    return {"status": "OK" if model is not None else "degraded",
            "model_loaded": model is not None,
            "served_by": served_by,
            "error": load_error
            }

def retrieve_and_log(query, k):
    '''Runs one query, times it, and logs it to DynamoDB. Returns the recs and the log id'''

    query_start = time.time()

    recs = model.recommend(query, k=k)

    latency_ms = (time.time() - query_start) * 1000

    # log_retrieval fails soft and returns None, so a dead table cannot break serving
    retrieval_id = log_retrieval(
        query=query,
        recommendations=recs,
        latency_ms=latency_ms,
        served_by=served_by,
        k=k
    )

    return recs, retrieval_id, latency_ms

@app.post("/retrieve")
def retrieve_recs(input_data: TextInput):
    '''Retrieves closest matches to a single query input'''

    require_model()

    recs, retrieval_id, latency_ms = retrieve_and_log(input_data.text, input_data.k)

    return {"retrieval_id": retrieval_id,
            "latency_ms": round(latency_ms, 1),
            "served_by": served_by,
            "recommendations": recs}

@app.post("/retrieve_from_queries")
def retrieve_batch_recs(input_data: BatchTextInput):
    '''Retrieves closest matches to multiple query inputs'''

    require_model()

    results = []

    for query in input_data.text:
        recs, retrieval_id, latency_ms = retrieve_and_log(query, input_data.k)

        results.append({
            "query": query,
            "retrieval_id": retrieval_id,
            "latency_ms": round(latency_ms, 1),
            "recommendations": recs
        })

    return {"count": len(results),
            "served_by": served_by,
            "results": results}

@app.post("/feedback")
def submit_feedback(input_data: FeedbackInput):
    '''Attaches a thumbs up/down to a retrieval that /retrieve already logged'''

    # deliberately no require_model() -- feedback is about a past retrieval,
    # so it still works when the registry pull failed

    if not record_feedback(input_data.retrieval_id, input_data.helpful):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'No logged retrieval with id {input_data.retrieval_id}'
        )

    return {"retrieval_id": input_data.retrieval_id,
            "helpful": input_data.helpful,
            "recorded": True}

@app.get("/example")
def get_training_example():
    '''Tests retrieval functionality by returning a random title from the catalog'''

    require_model()

    example_title = np.random.choice(model.catalog['Title'])

    return {
        "example_query": example_title,
        "hint": "POST this to /retrieve as {\"text\": \"<example_query>\", \"k\": 10}",
    }

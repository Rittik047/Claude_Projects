"""FastAPI service. Run: `uvicorn ecgtrust.serve.api:app` (loads ./release by default,
override with ECGTRUST_RELEASE). Research software — not a medical device."""

from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ecgtrust.serve.predictor import InputError, PredictionResult, Predictor


class SignalIn(BaseModel):
    signal: list[list[float | None]]
    fs: int


def create_app(predictor: Predictor | None = None) -> FastAPI:
    app = FastAPI(
        title="ecg-trust", version="0.1.0", description="Research prototype; not for clinical use."
    )

    @lru_cache(maxsize=1)
    def get() -> Predictor:
        return predictor or Predictor.from_release(
            Path(os.environ.get("ECGTRUST_RELEASE", "release"))
        )

    def run(x: np.ndarray, fs: int) -> PredictionResult:
        try:
            return get().predict(x, fs)
        except InputError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": get().version}

    @app.post("/predict", response_model=PredictionResult)
    def predict(body: SignalIn):
        x = np.array([[np.nan if v is None else v for v in row] for row in body.signal], float)
        return run(x, body.fs)

    @app.post("/predict/npy", response_model=PredictionResult)
    async def predict_npy(file: UploadFile = File(...), fs: int = Form(...)):
        try:
            x = np.load(io.BytesIO(await file.read()), allow_pickle=False)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"not a valid .npy array: {e}") from e
        return run(x, fs)

    return app


app = create_app()

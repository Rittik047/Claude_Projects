# ecg-trust inference service (CPU). Build from the ecg-trust/ directory:
#   docker build -t ecg-trust .
#   docker run -p 8000:8000 ecg-trust                       # FastAPI on :8000
#   docker run -p 8501:8501 ecg-trust streamlit run app/streamlit_app.py --server.port 8501
FROM python:3.11-slim
WORKDIR /srv
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"
COPY release ./release
COPY app ./app
EXPOSE 8000 8501
CMD ["uvicorn", "ecgtrust.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]

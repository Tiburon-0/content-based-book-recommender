# streamlit frontend (phase 3.1) -- app instance, port 8501
# build from the project ROOT:  docker build -f web/Dockerfile -t <name> .
# this is the user-facing app, not the monitoring dashboard

FROM python:3.11-slim

WORKDIR /app

COPY web/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pure http client of the api -- no model, no .pkl, no dynamodb
# one file and no ml stack, so this image is ~150 MB against the api's ~1 GB
COPY web/app.py .

EXPOSE 8501

# --server.address=0.0.0.0 or the container is unreachable from the public ip
# gatherUsageStats=false stops streamlit hanging on a phone-home behind a
# locked-down security group
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false"]

# monitoring dashboard (phase 3.2) -- runs ALONE on the second instance, port 8501
# build from the project ROOT:  docker build -f monitoring/Dockerfile -t <name> .
#
# no shared network and no shared volume with the api
# spec: "data will be exchanged through a Database, not JSON files"
# assignment 5 passed a json file through a named volume, but volumes do not span
# hosts -- here dynamodb is the shared state

FROM python:3.11-slim

WORKDIR /app

COPY monitoring/requirements.txt .
# --no-cache-dir takes TWO dashes
# a single dash parses as the short flags -n -o and the build fails on this line
RUN pip install --no-cache-dir -r requirements.txt

# db.py comes from the root and is shared with the api image
# one copy means the table name, region and Decimal handling are defined once
COPY monitoring/app.py db.py ./

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false"]

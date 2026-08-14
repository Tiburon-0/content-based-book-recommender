# fastapi backend -- app instance, port 8000
# build from the project ROOT:  docker build -f api/Dockerfile -t <name> .
# the trailing . is the build context, -f picks the recipe
# root context is what lets the COPY below reach recommender.py and db.py

# 3.11 not 3.13: sklearn and scipy ship prebuilt wheels for 3.11
# without them pip compiles from source -- 20+ minutes, and it can exhaust 1 GB of RAM
FROM python:3.11-slim

WORKDIR /app

# requirements before code so docker caches the pip layer
# editing main.py then rebuilds in seconds instead of reinstalling sklearn
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# recommender.py is required even though main.py never calls it directly
# joblib pickles the class path, so unpickling runs `import recommender`
# without it the registry download succeeds and the load dies on ModuleNotFoundError
COPY main.py recommender.py db.py ./

EXPOSE 8000

# --host 0.0.0.0 is mandatory
# the 127.0.0.1 default only accepts connections from inside the container,
# and it presents as a security group problem
#
# uvicorn rather than `fastapi run`: the fastapi cli ships in the [standard] extra,
# so a plain fastapi install builds fine and then fails at container start
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

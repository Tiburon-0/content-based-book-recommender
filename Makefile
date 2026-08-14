# three images, two ec2 instances
#
#   APP HOST                      MONITORING HOST
#     api  :8000  --+               dash :8501
#     web  :8501  --+ app_net           |
#           +---- writes --> DynamoDB <-+ reads
#
# no shared volume anywhere
# assignment 5 passed a json file between containers through a named volume, but
# volumes do not span hosts -- the two services here sit on different machines,
# so dynamodb is the shared state
#
# run targets are split by host because make cannot start a container on a
# machine it is not running on: run-app on instance 1, run-dash on instance 2

IMAGE_API  := tiburon-book-recommender-api
IMAGE_WEB  := tiburon-book-recommender-web
IMAGE_DASH := tiburon-book-recommender-dash

APP_NET := app_net

# must match what create_table.py provisioned
AWS_REGION     ?= us-east-1
DYNAMODB_TABLE ?= book_recommender_retrievals

# the frontend reaches the api by CONTAINER NAME over the shared network
# the ec2 public ip changes on every restart, a container name never does
API_URL ?= http://api:8000

# lab accounts usually cannot create iam roles, so the containers read the
# credentials file mounted from the host
# if you can attach an instance role instead, do that and delete this line
AWS_MOUNT := -v $(HOME)/.aws:/root/.aws:ro

.PHONY: build build-api build-web build-dash \
        run-app run-dash stop-app stop-dash \
        clean clean-app clean-dash logs-api logs-web logs-dash ps

# --------[BUILD]--------
# every build uses `-f <dir>/Dockerfile .` -- the . is the project root as context
# that is what lets the api image copy recommender.py and db.py from the root
# .dockerignore keeps the CSVs and the .pkl out of that context

build: build-api build-web build-dash

build-api:
	docker build -f api/Dockerfile -t $(IMAGE_API) .

build-web:
	docker build -f web/Dockerfile -t $(IMAGE_WEB) .

build-dash:
	docker build -f monitoring/Dockerfile -t $(IMAGE_DASH) .

# --------[RUN: APP HOST]--------
# api + web on one docker network so web can address the api as http://api:8000
# the leading - on network create lets a re-run succeed when it already exists
# WANDB_API_KEY comes from the shell at run time, never baked into the image

run-app: stop-app
	-docker network create $(APP_NET)
	docker run -d --name api \
		--network $(APP_NET) \
		-p 8000:8000 \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		-e AWS_REGION=$(AWS_REGION) \
		-e DYNAMODB_TABLE=$(DYNAMODB_TABLE) \
		$(AWS_MOUNT) \
		$(IMAGE_API)
	docker run -d --name web \
		--network $(APP_NET) \
		-p 8501:8501 \
		-e API_URL=$(API_URL) \
		$(IMAGE_WEB)
	@echo "api  -> http://localhost:8000/docs"
	@echo "web  -> http://localhost:8501"
	@echo "the api pulls a 173 MB model at startup, so /health reads 'degraded'"
	@echo "for 30-60s before it reads 'OK' -- watch it with: make logs-api"

# --------[RUN: MONITORING HOST]--------
# no network, no volume, no link to the api -- its only input is dynamodb

run-dash: stop-dash
	docker run -d --name dash \
		-p 8501:8501 \
		-e AWS_REGION=$(AWS_REGION) \
		-e DYNAMODB_TABLE=$(DYNAMODB_TABLE) \
		$(AWS_MOUNT) \
		$(IMAGE_DASH)
	@echo "dash -> http://localhost:8501"

# --------[STOP / CLEAN]--------
# stop-* removes containers only, so run-* is repeatable
# clean-* also removes the images and the network
#
# nothing here touches dynamodb -- the retrieval history survives teardown,
# instance restart, and the end of a lab session
# assignment 5's make clean destroyed the volume and every logged prediction

stop-app:
	-docker rm -f api web

stop-dash:
	-docker rm -f dash

clean-app: stop-app
	-docker network rm $(APP_NET)
	-docker rmi $(IMAGE_API) $(IMAGE_WEB)

clean-dash: stop-dash
	-docker rmi $(IMAGE_DASH)

clean: clean-app clean-dash

# --------[INSPECT]--------
# logs-api first when /health says degraded -- the registry pull prints there

logs-api:
	docker logs -f api

logs-web:
	docker logs -f web

logs-dash:
	docker logs -f dash

ps:
	docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

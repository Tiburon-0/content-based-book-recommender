# Deployment Runbook

Bringing the system up on AWS from nothing. Written to be followed top to bottom in
one lab session, because the lab session is the unit that matters — when it ends, the
instances stop, and the public IPs are gone.

**Do this once end-to-end before you need it to work.** The first run finds the
problems; the second run is the one you screenshot.

---

## The shape of what you are building

```
        YOUR BROWSER
             |
    +--------+---------+
    |                  |
  :8501              :8501
+------------+   +------------------+
| APP HOST   |   | MONITORING HOST  |     two separate EC2 instances
|            |   |                  |
|  web  :8501|   |  dash  :8501     |
|  api  :8000|   |                  |
+-----+------+   +--------+---------+
      |                   |
      |  writes           |  reads
      +------> DynamoDB <-+
              (the only thing
               they share)
```

The two hosts never talk to each other. That is the design, not a limitation — the
spec requires "data will be exchanged through a Database, not JSON files."

Inside the app host, `web` reaches `api` by container name over a Docker network
(`http://api:8000`), so nothing breaks when the public IP changes.

---

## Two phases — only the second has prerequisites

**Phase A (Steps 1–3) needs nothing.** Launch instances, set security groups, install
Docker, run `hello-world`. Pure AWS de-risking, no project code involved. Start here
even if nothing else is ready — this is where the unfamiliar failure modes live (key
pairs, security groups, ports), and none of them are your code.

**Phase B (Steps 4–7) needs three things done first,** all from your laptop, not from
EC2:

- [ ] **Push to GitHub** — Step 4 clones the repo onto each instance
- [ ] **`python create_table.py`** — the API writes to a table that must already exist
- [ ] **A version aliased `production`** — the API pulls `:production` at startup and
  503s if nothing carries that alias. Check with `python promote_model.py --list`

Also: start your AWS lab session and note how long it runs for.

---

## Step 1 — Launch two instances

|         | App host                      | Monitoring host         |
| ------- | ----------------------------- | ----------------------- |
| Name    | `recommender-app`           | `recommender-monitor` |
| AMI     | Ubuntu 22.04 LTS              | Ubuntu 22.04 LTS        |
| Type    | **t3.small** (see note) | t2.micro is fine        |
| Storage | 16 GB                         | 8 GB                    |

**The instance-type note, which is the single most likely thing to cost you an
afternoon.** The API unpickles a model whose vectorizer vocabulary alone is 1,240,874
terms, plus a 50.8 MB matrix and a 165,744-row catalog — on top of Python, scikit-learn
and scipy, inside a container. A `t2.micro` has 1 GB of RAM and will likely be killed
by the OOM reaper partway through loading. The symptom is confusing: the container
just exits, `docker logs api` ends mid-startup with no traceback.

Two ways out:

1. Use a 2 GB instance (`t3.small`) for the app host. Simplest.
2. If your lab account will not allow it, retrain with `min_df=5` — you already
   measured that this drops the vocabulary 78% and the matrix 18.6%, which brings it
   inside 1 GB. That also gives you a second registry version to promote between,
   which Phase 1.3 wants anyway.

The monitoring host has no model and no ML libraries, so `t2.micro` is genuinely fine
there.

---

## Step 2 — Security groups

This is the number one reason for "the container is running but I can't reach it."
Check it before debugging anything else.

**App host** inbound rules:

| Type       | Port | Source    |
| ---------- | ---- | --------- |
| SSH        | 22   | My IP     |
| Custom TCP | 8000 | 0.0.0.0/0 |
| Custom TCP | 8501 | 0.0.0.0/0 |

**Monitoring host** inbound rules:

| Type       | Port | Source    |
| ---------- | ---- | --------- |
| SSH        | 22   | My IP     |
| Custom TCP | 8501 | 0.0.0.0/0 |

> Screenshot both security group pages — they are good AWS Console evidence.

---

## Step 3 — Install Docker (run on BOTH instances)

```bash
ssh -i your-key.pem ubuntu@<PUBLIC_IP>

sudo apt-get update
sudo apt-get install -y docker.io make git
sudo systemctl enable --now docker

# lets you run docker without sudo
sudo usermod -aG docker ubuntu

# the group change only applies to a NEW login -- log out and back in
exit
```

Reconnect, then confirm:

```bash
docker run --rm hello-world
```

If that prints the hello-world banner, Docker is working and every later problem is
your application, not the platform. That distinction is worth the 30 seconds.

---

## Step 4 — Get the code and the credentials onto each instance

```bash
git clone https://github.com/Tiburon-0/content-based-book-recommender.git
cd content-based-book-recommender
```

The repo deliberately contains no secrets, so both must be supplied on the instance.

**AWS credentials** (needed on both hosts — the API writes to DynamoDB, the dashboard
reads from it). In a lab account these rotate every session, so copy the current
values out of the lab's credentials panel:

```bash
mkdir -p ~/.aws
cat > ~/.aws/credentials <<'EOF'
[default]
aws_access_key_id=...
aws_secret_access_key=...
aws_session_token=...
EOF
```

> If your account lets you attach an IAM instance role instead, do that — it is the
> better practice and removes credentials from disk entirely. Then delete the
> `AWS_MOUNT` line from the Makefile.

**W&B key** (app host only — the API pulls the model from the registry at startup):

```bash
export WANDB_API_KEY=<your key>
```

---

## Step 5 — App host: build and run

```bash
make build-api build-web
make run-app
```

Then watch the API come up:

```bash
make logs-api
```

**Expect a slow first start.** The API downloads a 173 MB artifact from the W&B
registry before it serves anything, so `/health` reports `degraded` for roughly 30–60
seconds and then flips to `OK`. That is normal, not a failure.

Verify:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"text": "victorian detective novels with an unreliable narrator", "k": 5}'
```

From your own browser:

- API docs — `http://<APP_HOST_IP>:8000/docs`
- Frontend — `http://<APP_HOST_IP>:8501`

Send a few real queries through the frontend and vote thumbs up/down on some of them.
**Do this before you open the dashboard** — the dashboard has nothing to show until
retrievals and feedback exist in DynamoDB.

---

## Step 6 — Monitoring host: build and run

```bash
make build-dash
make run-dash
```

Open `http://<MONITOR_HOST_IP>:8501`.

If it says "No retrievals logged yet", the frontend traffic from Step 5 did not reach
DynamoDB. Check `make logs-api` on the app host for `[db]` lines — the database layer
fails soft and prints its errors rather than breaking requests, so the explanation is
there.

---

## Step 7 — Screenshots

Take these while everything is up. Once the session ends, recreating this state costs
another full bring-up.

**AWS Console**

- [ ] EC2 instances list showing both instances running
- [ ] Security group inbound rules (both)
- [ ] DynamoDB table with items in it
- [ ] DynamoDB item detail showing one retrieval row (query, latency, feedback)

**Running application**

- [ ] `/docs` Swagger page on the app host's public IP
- [ ] `/health` response showing `"status": "OK"` and the `served_by` block
- [ ] Frontend with real recommendations, book covers visible
- [ ] Frontend after voting — the "thanks, feedback recorded" state
- [ ] Monitoring dashboard: headline metrics row
- [ ] Monitoring dashboard: latency over time chart
- [ ] Monitoring dashboard: category distribution / drift comparison
- [ ] Monitoring dashboard: feedback and live accuracy
- [ ] Terminal showing `curl` against `/retrieve` with a real response

**GitHub**

- [ ] Actions tab showing a green CI run
- [ ] A pull request showing the required checks
- [ ] Branch protection settings requiring those checks

**W&B**

- [ ] Registry page showing versions and the `production` alias
- [ ] A run's metrics page

Put the public URLs in the README as text too. They will be dead by the time anyone
reads it — that is expected for an ephemeral lab account, and worth stating plainly
rather than leaving a grader to wonder.

---

## Teardown

```bash
make clean-app     # on the app host
make clean-dash    # on the monitoring host
```

Then stop both instances in the console.

**`make clean` does not delete your data.** The retrieval history lives in DynamoDB,
not in a Docker volume, so it survives teardown, instance termination, and the end of
the lab session. Your dashboard will still have history in it next session. This is a
concrete advantage of the database-backed design over the shared-volume approach in
Assignment 5, where `make clean` destroyed every logged prediction — worth a sentence
in the README.

---

## Troubleshooting

| Symptom                           | Cause                          | Fix                                                                     |
| --------------------------------- | ------------------------------ | ----------------------------------------------------------------------- |
| Browser hangs on`<IP>:8000`     | Security group                 | Add the inbound rule — check this first, always                        |
| Container runs, port unreachable  | App bound to localhost         | Both images already bind`0.0.0.0`; check you did not override the CMD |
| `/health` stuck on `degraded` | Registry pull failed           | Read`error` in the `/health` body — it holds the real exception    |
| `/health` error mentions alias  | Nothing aliased`production`  | `python promote_model.py --version vN --alias production`             |
| API container exits silently      | Out of memory                  | See the instance-type note in Step 1                                    |
| `[db]` errors in the API log    | AWS credentials                | Session tokens expire — re-copy them from the lab panel                |
| Dashboard empty                   | No traffic yet, or wrong table | Send queries first; confirm`DYNAMODB_TABLE` matches on both hosts     |
| `make: *** missing separator`   | Makefile indented with spaces  | Recipe lines need real tabs                                             |

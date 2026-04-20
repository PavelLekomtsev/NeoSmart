# ClearML — local setup

This project uses a locally-hosted ClearML server (no cloud account).
The whole stack runs via `docker-compose.clearml.yml`: elasticsearch,
mongo, redis, apiserver (`:8008`), fileserver (`:8081`), and the web
UI (`:8080`).

## One-time setup

### 1. Start the server

```bash
docker compose -f docker-compose.clearml.yml up -d
# check all 6 containers are Up:
docker compose -f docker-compose.clearml.yml ps
```

Open http://localhost:8080 in a browser. First visit shows a login
screen — ClearML's open-source build ships without auth by default,
so any email + password creates a workspace the first time. On
subsequent logins you use the same credentials (they are stored in
the mongo container).

### 2. Generate SDK credentials

Inside the web UI:

1. Top-right avatar → **Settings** → **Workspace** → **App
   Credentials**.
2. Click **+ Create new credentials**.
3. Copy the shown `access_key` and `secret_key`. The secret is
   displayed once.

### 3. Write the local SDK config

ClearML looks for its config in this order (first hit wins):

1. `$CLEARML_CONFIG_FILE`
2. `%USERPROFILE%\clearml.conf` on Windows / `~/clearml.conf` on
   Linux/macOS
3. `./clearml.conf` in the current working directory

Copy the template shipped in the repo and fill in your credentials:

```bash
# Windows (PowerShell / Git Bash)
cp .clearml.conf.example "$USERPROFILE/clearml.conf"

# Linux / macOS
cp .clearml.conf.example ~/clearml.conf
```

Open the copy and replace the two placeholder lines inside the
`api.credentials` block:

```
api {
    web_server: http://localhost:8080
    api_server: http://localhost:8008
    files_server: http://localhost:8081
    credentials {
        "access_key" = "YOUR_ACCESS_KEY"   # from the UI
        "secret_key" = "YOUR_SECRET_KEY"
    }
}
```

The real config file (`clearml.conf` / `.clearml.conf`) is in
`.gitignore` — only the placeholder `.clearml.conf.example` is
version-controlled.

### 4. Verify connectivity

```bash
python -c "from clearml import Task; t=Task.init(project_name='NeoSmart/_preflight', task_name='smoke'); t.close(); print('OK')"
```

Expected: a new task appears under the `NeoSmart/_preflight` project
in the web UI, and the script prints `OK`. If you see `Failed
connecting to server` or `HTTP 401`, re-check that the access/secret
keys in `clearml.conf` match the ones you just generated.

The same check is wired into `Training/run_experiments.py` as a
preflight step — it runs before the first experiment and fails fast
with the same hint messages.

## Everyday use

* **Watching a training run** — open http://localhost:8080, click
  the `NeoSmart/CarDetector` project, then the task you just
  launched. Ultralytics streams scalars live, so loss / mAP / P / R
  curves update every epoch.
* **Comparing runs** — select ≥ 2 tasks in the list and press
  **Compare**. ClearML aligns scalars side-by-side and diffs the
  Configuration and Hyperparameters sections automatically.
* **Model Registry** — `NeoSmart/CarDetector` → **Models** tab. After
  `python Training/promote_best.py ...`, the final `best.pt` is
  registered as `Car_Detector@v2` with `promoted` and `v2` tags.

## Stopping / restarting

```bash
docker compose -f docker-compose.clearml.yml stop   # keep data
docker compose -f docker-compose.clearml.yml down   # stop + remove containers (volumes persist)
docker compose -f docker-compose.clearml.yml down -v  # wipe everything (dangerous)
```

Named volumes (`clearml_data_*`) hold tasks, artifacts, and the
mongo/elasticsearch state, so `down` without `-v` is always safe.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Failed connecting to server` | server not running | `docker compose ... up -d` and wait ~30s |
| `HTTP 401` on `Task.init` | wrong or missing credentials | regenerate in UI, update `clearml.conf` |
| No scalars appear during training | ClearML callback never attached | ensure `clearml` is importable **before** `ultralytics` — `Training/train.py` already does this |
| `elasticsearch` container keeps restarting | Docker host memory < 4 GB | raise Docker Desktop memory limit in Settings → Resources |
| Slow UI / large history | elasticsearch index growing | `docker compose down` then `docker compose up -d` to recycle |

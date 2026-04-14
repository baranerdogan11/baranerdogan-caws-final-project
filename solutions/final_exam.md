# Final Exam — Theoretical Questions

## 3.1 API Design

### Question 1

The `POST /diagnose` request schema has exactly **7 fields**:

1. `piece_id` — string identifier for the piece
2. `die_matrix` — integer identifying which die matrix (4974, 5052, 5090, 5091)
3. `lifetime_2nd_strike_s` — cumulative time from furnace exit to 2nd strike (seconds)
4. `lifetime_3rd_strike_s` — cumulative time from furnace exit to 3rd strike
5. `lifetime_4th_strike_s` — cumulative time from furnace exit to 4th strike (drill)
6. `lifetime_auxiliary_press_s` — cumulative time from furnace exit to auxiliary press
7. `lifetime_bath_s` — cumulative time from furnace exit to quench bath

**Why cumulative rather than pre-computed partials?** Cumulative timestamps are what the PLC/sensor system records natively — they are monotonically increasing values written at each operation milestone. Asking the client to pre-compute the 5 partial differences would push business logic to the producer, violating the single-responsibility principle. The API owns the diagnostic rules and therefore owns the derivation of partial times.

**Why is this set the minimum necessary?** The 5 partial-time formulas in §1.5 require exactly 5 distinct cumulative timestamps: `t2`, `t3`, `t4`, `ta`, `tb`. Together with `piece_id` and `die_matrix` (needed to look up reference values and construct the response), these 7 fields are both necessary and sufficient. Removing any one of them makes it impossible to compute one or more segments of the diagnosis.

### Question 2

`reference_times.json` is loaded once at startup in `app.py` via:

```python
with open(REFERENCE_FILE) as f:
    REFERENCE_TIMES = json.load(f)
```

This is correct for a containerized deployment for three reasons. First, the file is bundled inside the image (`COPY reference_times.json ./reference_times.json` in the Dockerfile) — it never changes at runtime, so reading it repeatedly would be pure waste. Second, Fargate tasks can handle thousands of requests per second; re-reading a file on every request would add disk I/O to each prediction latency. Third, loading at startup makes the failure mode explicit: if the file is missing or malformed, the container exits immediately at launch with a clear error, rather than failing silently on the first production request.

---

## 3.2 Containerization And Deployment

### Question 1

The `Dockerfile` uses a **two-stage build**:

**Stage 1 (`builder`)** starts from `python:3.13-slim` and uses `uv` to install FastAPI, Uvicorn, and Pydantic into the system Python. This stage exists purely to install dependencies — keeping the installation layer separate means Docker can cache it. If only the application code changes, the expensive `uv pip install` layer is not re-executed.

**Stage 2 (`runtime`)** starts from a fresh `python:3.13-slim`. The base image was chosen because it matches the Python version specified in `pyproject.toml` (3.13+) and `slim` minimises image size by excluding build tools not needed at runtime. The stage copies the installed packages from the builder, then copies `src/` and `reference_times.json` into the image. `reference_times.json` must be inside the image so the API can load it at startup without any external dependency. `EXPOSE 80` documents the port; the `CMD` launches Uvicorn on `0.0.0.0:80` so Fargate's network bridge can reach it.

### Question 2

**Option 1 — AWS Lambda (+ API Gateway)**

- *Advantage*: Zero cost when idle — Lambda bills per invocation, not per hour. For a low-traffic diagnostic API this would be dramatically cheaper than a running Fargate task.
- *Disadvantage*: Cold-start latency. When Lambda initialises a new container it must load the Python runtime and `reference_times.json` from scratch, adding 300–800 ms to the first request after an idle period. Fargate keeps the container warm continuously.

**Option 2 — Amazon EC2 (t3.micro)**

- *Advantage*: Full control over the OS, networking, and instance configuration. Useful if the API needs to be combined with other processes on the same machine or requires a specific kernel feature.
- *Disadvantage*: You manage patching, scaling, and availability yourself. Fargate is serverless — AWS handles the underlying infrastructure, there is no OS to patch, and the task can be restarted automatically on failure.

---

## 3.3 Testing And Extensibility

### Question 1

The exam asks for `diagnose()` to be tested as a pure function (not through HTTP) for three reasons.

First, **speed and isolation**: calling `diagnose()` directly runs in microseconds with no network stack, no Uvicorn process, and no event loop to manage. The 24 unit tests + 10 golden tests complete in under a second; an equivalent HTTP test suite would take 10–100× longer and require a running server.

Second, **debuggability**: when a pure-function test fails, the traceback points directly to the logic in `diagnose.py`. An HTTP test failure points to a status code, requiring additional steps to locate the actual logic error.

Third, **portability**: `diagnose()` can be imported into a SageMaker pipeline, a batch script, or a different web framework without any FastAPI dependency. If the logic were embedded in the route handler, it could not be tested or reused independently.

### Question 2

To support a new die matrix `6001`, every layer of the system needs one targeted change:

**1. Data files** — Add a `"6001"` key to `reference_times.json` with the 5 median partial times computed from the new matrix's production data. No other data file changes.

**2. Code** — No code changes are required. `diagnose.py` and `app.py` look up the matrix key dynamically from `REFERENCE_TIMES`; they do not hardcode the list of valid matrices. The unknown-matrix guard (`if key not in reference_times: raise ValueError(...)`) automatically accepts `6001` once it appears in the JSON.

**3. Tests** — Add `6001` to the `MATRICES` list at the top of `test_diagnose.py`. This automatically generates 6 new parametrised tests (all-OK + one per segment). Also add one row to `validation_pieces.csv` and regenerate `validation_expected.json` using the generation script to extend the golden test.

**4. Redeployment** — Rebuild the Docker image (so the updated `reference_times.json` is bundled inside), push the new image to ECR, update the ECS task definition to reference the new image tag, and trigger a new Fargate deployment. The endpoint URL does not change.

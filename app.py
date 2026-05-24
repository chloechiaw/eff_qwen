"""Dev harness for the adaptfm Qwen3.5-4B competition on Modal.

Per competition spec, the container must serve on port 8080:
    GET  /ping                  — 200 once the model is loaded
    POST /invocations           — inference (legacy SageMaker shape)
    POST /v1/completions        — raw text, used for the LATENCY benchmark
    POST /v1/chat/completions   — chat template, used for QUALITY benchmarks

Submission images bake weights at /opt/ml/model/ — we mount the volume
there in dev so paths match production exactly.

NOTE: a long-running serve() web function previously lived in this file but
was removed: any `modal run` of *any* function in an app also auto-starts
the app's web functions, which kept crash-looping and swamping logs.
benchmark() is self-contained — it starts the server inside its own
container, measures, and tears down. Use `download.py` for weights.

Usage:
    modal run app.py::check_gpu     # verify the GPU is reachable
    modal run app.py::benchmark     # reproduce competition latency
    modal run app.py::quality_mini  # 5-min MMLU-Pro + IFEval sanity check
"""

import modal

image = (
    modal.Image.from_registry(
        "adaptfm/adaptfm-base:latest",
        setup_dockerfile_commands=[
            # The base image's ENTRYPOINT is /opt/program/serve.py, which
            # runs as PID 1 of every container and blocks waiting for model
            # weights forever — Modal's runner never gets a chance to invoke
            # our function. Clearing it lets Modal take over container
            # startup. We invoke the server explicitly via SERVE_CMD when we
            # actually want it (in benchmark()).
            "ENTRYPOINT []",
            "CMD []",
            # Base image ships python3 but no `python` on PATH; Modal's
            # builder runs `python -m pip install …` and fails without this.
            "RUN ln -sf $(command -v python3) /usr/local/bin/python",
        ],
    )
    # Modal's client install bumps pydantic_core to a version that needs
    # typing_extensions>=4.13 (for `Sentinel`); the base image ships older.
    .pip_install("typing_extensions>=4.13")
    # Quality eval harness — matches what the competition's
    # run_quality_local.py uses (lm-eval==0.4.11 + langdetect + immutabledict).
    .pip_install("lm-eval==0.4.11", "langdetect", "immutabledict")
)

weights = modal.Volume.from_name("adaptfm-weights", create_if_missing=True)

# lm-eval downloads benchmark datasets from HF on first run; cache them so
# subsequent quality_mini runs skip the download.
hf_cache = modal.Volume.from_name("adaptfm-hf-cache", create_if_missing=True)

WEIGHTS_DIR = "/opt/ml/model"
PORT = 8080
HF_REPO = "Qwen/Qwen3.5-4B"

# Base image's entrypoint script — observed via the traceback when the
# ENTRYPOINT was still firing on container start.
SERVE_CMD = ["python3", "/opt/program/serve_default.py"]

# Competition baseline (g5.xlarge / A10G, unoptimized).
#   name          prompt_tok  output_tok  baseline_ms
CATEGORIES = [
    ("Short",        64,  128, 2582),
    ("Medium",     2048,  256, 5441),
    ("Long",       8192,  256, 6576),
]

app = modal.App("adaptfm-dev", image=image)


# ---------- diagnostic: is the GPU visible? ----------

@app.function(gpu="A10G", timeout=5 * 60)
def check_gpu():
    """Confirm Modal's A10G is actually reachable from inside the base image."""
    import subprocess

    print("=== nvidia-smi ===")
    r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(r.stdout or r.stderr)

    print("=== torch ===")
    import torch
    print(f"torch.__version__         = {torch.__version__}")
    print(f"torch.version.cuda        = {torch.version.cuda}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"device_name(0)            = {torch.cuda.get_device_name(0)}")
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x
        print(f"matmul OK; result shape {tuple(y.shape)}, sum {y.sum().item():.2f}")


# ---------- benchmark: reproduces the competition's latency eval ----------

@app.function(
    gpu="A10G",
    volumes={WEIGHTS_DIR: weights},
    timeout=60 * 60,
)
def benchmark(warmup: int = 5, measure: int = 10):
    """Start the server inside this container and time /v1/completions.

    Per competition spec the latency benchmark is:
      - endpoint: /v1/completions  (raw text, no chat template, no thinking)
      - 5 warmup + 50 measurement runs per category
      - 3 categories: Short / Medium / Long (see CATEGORIES above)

    Default measure=10 for a quick sanity check; bump to 50 for the
    official count once you trust the numbers.
    """
    import subprocess
    import time

    print(f"starting server: {' '.join(SERVE_CMD)}")
    proc = subprocess.Popen(SERVE_CMD)
    try:
        _wait_local_ping(timeout_s=600)
        print("server ready; building prompts…")

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(WEIGHTS_DIR)

        results = []
        for name, prompt_tok, output_tok, baseline_ms in CATEGORIES:
            prompt = _make_prompt(tok, prompt_tok)
            req = {
                "prompt": prompt,
                "max_tokens": output_tok,
                "temperature": 0.0,
            }

            for _ in range(warmup):
                _post_local("/v1/completions", req)

            samples_ms = []
            for _ in range(measure):
                t0 = time.perf_counter()
                _post_local("/v1/completions", req)
                samples_ms.append((time.perf_counter() - t0) * 1000)

            avg = sum(samples_ms) / len(samples_ms)
            results.append((name, avg, baseline_ms, min(samples_ms), max(samples_ms)))
            print(
                f"  {name:6s}  avg {avg:7.0f} ms   "
                f"min {min(samples_ms):7.0f}   max {max(samples_ms):7.0f}   "
                f"baseline {baseline_ms} ms"
            )

        print()
        print(f"{'Category':8s} {'Avg ms':>8s} {'Baseline':>9s} {'Speedup':>9s}")
        total_avg = total_base = 0.0
        for name, avg, baseline_ms, _mn, _mx in results:
            ratio = baseline_ms / avg
            total_avg += avg
            total_base += baseline_ms
            print(f"{name:8s} {avg:8.0f} {baseline_ms:9d} {ratio:8.2f}x")
        mean_avg = total_avg / len(results)
        mean_base = total_base / len(results)
        print(
            f"{'Average':8s} {mean_avg:8.0f} {mean_base:9.0f} "
            f"{mean_base / mean_avg:8.2f}x"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------- quality mini test ----------

# Competition thresholds: (baseline, minimum to pass).
QUALITY_THRESHOLDS = {
    "mmlu_pro": (0.690, 0.621),
    "ifeval":   (0.857, 0.814),
    # gpqa_diamond: (0.700, 0.630) — skipped in mini test (see docstring)
}


@app.function(
    gpu="A10G",
    volumes={
        WEIGHTS_DIR: weights,
        "/root/.cache/huggingface": hf_cache,
    },
    timeout=30 * 60,
)
def quality_mini(limit: int = 20):
    """5-min quality sanity check: MMLU-Pro + IFEval on ~`limit` questions each.

    Uses lm-eval==0.4.11 (same harness as the competition's
    run_quality_local.py), but with --limit so it finishes in minutes
    instead of an hour.

    Skips GPQA-Diamond: it needs thinking-mode (max_tokens=12288) + streaming,
    and each question takes 30-90s — even 5 questions blow the 5-min budget.
    Run the competition's full eval for that gate before submitting.

    Use this as a regression check: run after each optimization to confirm
    you haven't tanked quality. Not authoritative — the small sample has
    high variance. A failing mini test means re-check with a larger limit.
    """
    import subprocess

    proc = subprocess.Popen(SERVE_CMD)
    try:
        _wait_local_ping(timeout_s=600)
        print(f"server ready; running lm-eval (limit={limit})…")

        import lm_eval

        results = lm_eval.simple_evaluate(
            model="local-chat-completions",
            model_args=(
                f"base_url=http://localhost:{PORT}/v1/chat/completions,"
                f"model={HF_REPO},"
                f"num_concurrent=4,"
                f"tokenizer_backend=huggingface,"
                f"tokenizer={WEIGHTS_DIR}"
            ),
            tasks=list(QUALITY_THRESHOLDS.keys()),
            limit=limit,
            apply_chat_template=True,
        )

        print()
        print(
            f"{'Task':12s} {'Score':>8s} {'Baseline':>10s} "
            f"{'Threshold':>11s} {'Status':>8s}"
        )
        for task, metrics in results["results"].items():
            score = next(
                (
                    v for k, v in metrics.items()
                    if (("acc" in k) or ("exact_match" in k) or ("prompt_level" in k))
                    and isinstance(v, (int, float)) and "stderr" not in k
                ),
                None,
            )
            if score is None:
                print(f"{task:12s} (no scalar metric found in {list(metrics.keys())})")
                continue
            baseline, threshold = QUALITY_THRESHOLDS[task]
            status = "OK" if score >= threshold else "FAIL"
            print(
                f"{task:12s} {score:8.3f} {baseline:10.3f} "
                f"{threshold:11.3f} {status:>8s}"
            )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------- helpers ----------

def _make_prompt(tok, n_tokens: int) -> str:
    seed = "The quick brown fox jumps over the lazy dog. "
    seed_ids = tok.encode(seed, add_special_tokens=False)
    n_copies = (n_tokens // len(seed_ids)) + 1
    ids = (seed_ids * n_copies)[:n_tokens]
    return tok.decode(ids)


def _wait_local_ping(timeout_s: int) -> None:
    import time
    import urllib.error
    import urllib.request

    url = f"http://localhost:{PORT}/ping"
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print(f"  /ping OK after {int(time.time() - start)} s")
                    return
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass
        time.sleep(2)
    raise RuntimeError(f"server did not become ready in {timeout_s} s")


def _post_local(path: str, payload: dict) -> bytes:
    import json
    import urllib.request

    req = urllib.request.Request(
        f"http://localhost:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=120).read()

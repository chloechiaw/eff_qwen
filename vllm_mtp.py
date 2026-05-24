"""vLLM with native MTP speculative decoding for Qwen3.5-VL-4B.

Same benchmark setup as app.py — only difference is we use a newer vLLM
that knows how to invoke the model's built-in MTP heads for speculative
decoding. Run alongside app.py::benchmark to attribute the speedup.

Usage:
    modal run vllm_mtp.py::benchmark                  # MTP enabled (default)
    modal run vllm_mtp.py::benchmark -- --no-mtp      # same vLLM, MTP off
                                                      # (isolates "newer vLLM"
                                                      #  vs "MTP" contributions)

Reuses the adaptfm-weights volume from download.py; no re-download.
"""

import modal

# vLLM's official container — has matching CUDA, torch, vLLM versions.
VLLM_IMAGE = "vllm/vllm-openai:latest"

image = (
    modal.Image.from_registry(
        VLLM_IMAGE,
        setup_dockerfile_commands=[
            # Same playbook as trtllm.py / app.py:
            # 1) clear ENTRYPOINT so Modal's runner takes over
            # 2) strip Debian pip/wheel/setuptools so Modal's upgrade works
            # 3) bootstrap fresh pip+wheel+setuptools
            # 4) symlink python -> python3 for Modal's builder
            "ENTRYPOINT []",
            "CMD []",
            "RUN apt-get update "
            "&& apt-get -y install --no-install-recommends curl ca-certificates "
            "&& apt-get clean",
            "RUN apt-get -y remove --purge "
            "python3-pip python3-wheel python3-setuptools || true",
            "RUN curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py "
            "&& python3 /tmp/get-pip.py "
            "&& rm /tmp/get-pip.py "
            "&& python3 -m pip install --upgrade setuptools wheel",
            "RUN ln -sf $(command -v python3) /usr/local/bin/python",
        ],
    )
    # Same fix as app.py: Modal's client install pins typing_extensions to
    # an older version that's missing `Sentinel`, breaking pydantic_core.
    .pip_install("typing_extensions>=4.13", "huggingface_hub", "transformers")
)

weights = modal.Volume.from_name("adaptfm-weights", create_if_missing=True)

WEIGHTS_DIR = "/opt/ml/model"
PORT = 8080
HF_REPO = "Qwen/Qwen3.5-4B"

# Competition baseline (vLLM 0.19 in adaptfm-base, no MTP).
#   name          prompt_tok  output_tok  baseline_ms
CATEGORIES = [
    ("Short",        64,  128, 2582),
    ("Medium",     2048,  256, 5441),
    ("Long",       8192,  256, 6576),
]

app = modal.App("adaptfm-vllm-mtp", image=image)


@app.function(
    gpu="A10G",
    volumes={WEIGHTS_DIR: weights},
    timeout=60 * 60,
)
def benchmark(
    warmup: int = 5,
    measure: int = 10,
    mtp: bool = True,
    num_speculative_tokens: int = 1,
):
    """Boot vLLM (optionally with MTP), run the 3-category latency benchmark.

    `mtp=True`  → enables native MTP speculative decoding via the model's
                  built-in MTP heads (the model has `mtp_num_hidden_layers: 1`
                  in its config).
    `mtp=False` → same vLLM, MTP disabled. Use this to compare and isolate
                  what "newer vLLM" contributes vs what "MTP" contributes.
    """
    import subprocess
    import time

    cmd = [
        "vllm", "serve", WEIGHTS_DIR,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--served-model-name", HF_REPO,
        "--trust-remote-code",
        # Latency mode: single-request, no batching reservations.
        "--max-num-seqs", "1",
        # Skip vision tower load — purely text benchmark.
        "--limit-mm-per-prompt", '{"image": 0, "video": 0}',
    ]
    if mtp:
        # Native MTP speculative decoding using the model's built-in heads.
        # Syntax in vLLM ≥ 0.7; format may shift across versions.
        cmd += [
            "--speculative-config",
            f'{{"method":"mtp","num_speculative_tokens":{num_speculative_tokens}}}',
        ]

    label = f"mtp={'on' if mtp else 'off'}"
    print(f"[{label}] starting: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    try:
        _wait_local_ready(timeout_s=600)
        print("server ready; building prompts…")

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(WEIGHTS_DIR, trust_remote_code=True)

        results = []
        for name, prompt_tok, output_tok, baseline_ms in CATEGORIES:
            prompt = _make_prompt(tok, prompt_tok)
            req = {
                "model": HF_REPO,
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
            results.append((name, avg, baseline_ms))
            print(
                f"  {name:6s}  avg {avg:7.0f} ms   "
                f"min {min(samples_ms):7.0f}   max {max(samples_ms):7.0f}   "
                f"baseline {baseline_ms} ms"
            )

        print()
        print(f"=== {label} ===")
        print(f"{'Category':8s} {'Avg ms':>8s} {'Baseline':>9s} {'Speedup':>9s}")
        total_avg = total_base = 0.0
        for name, avg, baseline_ms in results:
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


# ---------- helpers ----------

def _make_prompt(tok, n_tokens: int) -> str:
    seed = "The quick brown fox jumps over the lazy dog. "
    seed_ids = tok.encode(seed, add_special_tokens=False)
    n_copies = (n_tokens // len(seed_ids)) + 1
    ids = (seed_ids * n_copies)[:n_tokens]
    return tok.decode(ids)


def _wait_local_ready(timeout_s: int) -> None:
    """Poll vLLM's /v1/models until 200."""
    import time
    import urllib.error
    import urllib.request

    url = f"http://localhost:{PORT}/v1/models"
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print(f"  ready after {int(time.time() - start)} s")
                    return
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass
        time.sleep(2)
    raise RuntimeError(f"vLLM not ready in {timeout_s} s")


def _post_local(path: str, payload: dict) -> bytes:
    import json
    import urllib.request

    req = urllib.request.Request(
        f"http://localhost:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=120).read()

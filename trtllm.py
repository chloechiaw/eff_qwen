"""TensorRT-LLM serving stack: Qwen3.5-4B + INT4 AWQ + CUDA graphs.

Pipeline:
    modal run trtllm.py::build_engine    # one-time, ~20-30 min
    modal run trtllm.py::benchmark       # repeat as you tune

The engine is single-batch (max_batch_size=1) with CUDA graphs enabled —
configured specifically for the competition's single-request latency
benchmark. NOT optimized for throughput / concurrent requests.

Reuses the existing `adaptfm-weights` Modal volume populated by
download.py (fp16 weights at /opt/ml/model). The TRT-LLM checkpoint and
compiled engine live in a separate volume so we don't re-build every run.

If anything fails:
  - The exact path to convert_checkpoint.py varies by TRT-LLM version;
    check /opt/tensorrt_llm/examples/qwen/ inside the container.
  - `--use_cuda_graph` may have been renamed in newer trtllm-build CLIs.
  - Qwen3.5 may need a fresh TRT-LLM version (>=0.14 likely); older
    versions only support Qwen2.
  - First AWQ quantization downloads a calibration dataset from HF —
    needs internet (Modal containers have it by default).
"""

import modal

# Official NVIDIA TRT-LLM image — has CUDA + TRT + TRT-LLM pre-aligned.
# 24.10 has TRT-LLM 0.14 / transformers 4.45, both of which predate Qwen3.
# Bumping to a 2025 tag so transformers knows qwen3_5 and TRT-LLM has the
# Qwen3 model definition.
TRTLLM_IMAGE = "nvcr.io/nvidia/tritonserver:25.10-trtllm-python-py3"

image = (
    modal.Image.from_registry(
        TRTLLM_IMAGE,
        setup_dockerfile_commands=[
            # Same trick as app.py: clear ENTRYPOINT so Modal's runner
            # takes over container startup.
            "ENTRYPOINT []",
            "CMD []",
            # The 25.10 NGC image installs pip via apt (Debian), which
            # doesn't ship the RECORD file pip needs to uninstall itself.
            # Modal's builder runs `pip install --upgrade pip` in a later
            # step, which fails with "Cannot uninstall pip … RECORD file
            # not found". Strip Debian's pip and reinstall via the official
            # bootstrap so it has a real RECORD.
            "RUN apt-get update "
            "&& apt-get -y install --no-install-recommends curl ca-certificates "
            "&& apt-get clean",
            # Modal's builder also upgrades wheel and uv; debian's wheel /
            # setuptools have the same no-RECORD problem, so strip them all.
            "RUN apt-get -y remove --purge "
            "python3-pip python3-wheel python3-setuptools || true",
            "RUN curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py "
            "&& python3 /tmp/get-pip.py "
            "&& rm /tmp/get-pip.py "
            "&& python3 -m pip install --upgrade setuptools wheel",
            "RUN ln -sf $(command -v python3) /usr/local/bin/python",
        ],
    )
    # setuptools is needed by torch.utils.cpp_extension, which modelopt's
    # AWQ quantization imports — without it, quantize.py fails with the
    # misleading "Please install optional [torch] dependencies".
    .pip_install("huggingface_hub", "transformers", "setuptools")
)

weights = modal.Volume.from_name("adaptfm-weights", create_if_missing=True)
engine_vol = modal.Volume.from_name("adaptfm-trt-engine", create_if_missing=True)

WEIGHTS_DIR = "/opt/ml/model"
ENGINE_DIR = "/opt/ml/engine"
CKPT_DIR = "/tmp/trt_ckpt"
PORT = 8080
HF_REPO = "Qwen/Qwen3.5-4B"

CATEGORIES = [
    ("Short",        64,  128, 2582),
    ("Medium",     2048,  256, 5441),
    ("Long",       8192,  256, 6576),
]

app = modal.App("adaptfm-trtllm", image=image)


# ---------- one-time: convert HF weights -> AWQ-INT4 TRT-LLM engine ----------

@app.function(
    gpu="A10G",
    volumes={WEIGHTS_DIR: weights, ENGINE_DIR: engine_vol},
    timeout=60 * 60,
)
def build_engine():
    """Convert + build the TRT-LLM engine. Run once per config change."""
    import os
    import subprocess

    if os.path.exists(f"{ENGINE_DIR}/rank0.engine") or any(
        f.endswith(".engine") for f in os.listdir(ENGINE_DIR)
        if os.path.isdir(ENGINE_DIR)
    ):
        print(f"engine already at {ENGINE_DIR}; delete the volume to rebuild")
        return

    quantize_script = _find_script("quantize.py", path_contains="quantization")
    print(f"using quantize script: {quantize_script}")

    # AWQ in TRT-LLM 0.14+ goes through the modelopt-based quantize.py,
    # not the per-model convert_checkpoint.py (which only supports int8 /
    # int4 / int4_gptq). The output is still a TRT-LLM checkpoint that
    # trtllm-build accepts.
    print("step 1/2: HF -> TRT-LLM checkpoint (INT4 AWQ via modelopt)")
    _run_and_show(
        [
            "python", quantize_script,
            "--model_dir", WEIGHTS_DIR,
            "--output_dir", CKPT_DIR,
            "--dtype", "float16",
            "--qformat", "int4_awq",
            "--calib_size", "32",
            "--tp_size", "1",
        ]
    )

    print("step 2/2: build TRT engine with CUDA graphs (latency mode)")
    _run_and_show(
        [
            "trtllm-build",
            "--checkpoint_dir", CKPT_DIR,
            "--output_dir", ENGINE_DIR,
            "--gemm_plugin", "float16",
            "--use_cuda_graph",
            "--max_batch_size", "1",
            "--max_input_len", "8200",
            "--max_seq_len", "8500",
        ]
    )
    engine_vol.commit()
    print(f"engine saved to {ENGINE_DIR}")


# ---------- benchmark: same 3-category latency eval as app.py ----------

@app.function(
    gpu="A10G",
    volumes={WEIGHTS_DIR: weights, ENGINE_DIR: engine_vol},
    timeout=60 * 60,
)
def benchmark(warmup: int = 5, measure: int = 10):
    """Start trtllm-serve, hit /v1/completions, report avg per category."""
    import subprocess
    import time

    print("starting trtllm-serve (OpenAI-compatible API on :%d)" % PORT)
    proc = subprocess.Popen(
        [
            "trtllm-serve", "serve",
            ENGINE_DIR,
            "--tokenizer", WEIGHTS_DIR,
            "--port", str(PORT),
            "--host", "0.0.0.0",
        ]
    )
    try:
        _wait_local_ready(timeout_s=300)
        print("server ready; building prompts…")

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(WEIGHTS_DIR)

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

def _run_and_show(cmd: list[str]) -> None:
    """subprocess.run with check=True, but always echo stdout and stderr.

    The default subprocess.run with check=True raises CalledProcessError on
    non-zero exit but the actual stderr can get lost in Modal's log capture.
    This explicitly prints both streams before raising, so we always see why
    a subprocess failed.
    """
    import subprocess

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print("--- stdout ---")
        print(result.stdout)
    if result.stderr:
        print("--- stderr ---")
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"command exited {result.returncode}: {' '.join(cmd[:3])}…"
        )

def _find_script(filename: str, path_contains: str | None = None) -> str:
    """Find a script inside the TRT-LLM repo matching the installed version.

    Clones the matching tag (lazily) on first call, then walks the tree for
    `filename`, optionally filtering to paths containing `path_contains`.
    """
    import os
    import subprocess

    import tensorrt_llm

    version = tensorrt_llm.__version__
    repo_dir = "/tmp/TensorRT-LLM"

    if not os.path.exists(repo_dir):
        print(f"installed TRT-LLM version: {version}")
        tag = f"v{version}"
        print(f"cloning TRT-LLM at {tag}")
        result = subprocess.run(
            [
                "git", "clone", "--depth", "1", "--branch", tag,
                "https://github.com/NVIDIA/TensorRT-LLM.git", repo_dir,
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  tag {tag} not found; falling back to main")
            print(f"  git stderr: {result.stderr.strip()}")
            subprocess.run(
                [
                    "git", "clone", "--depth", "1",
                    "https://github.com/NVIDIA/TensorRT-LLM.git", repo_dir,
                ],
                check=True,
            )

    candidates = []
    for root, _dirs, files in os.walk(repo_dir):
        if filename in files and (
            path_contains is None or path_contains in root.lower()
        ):
            candidates.append(os.path.join(root, filename))

    if not candidates:
        examples_dir = os.path.join(repo_dir, "examples")
        print(f"\n{filename} not found. {examples_dir} contains:")
        if os.path.isdir(examples_dir):
            for entry in sorted(os.listdir(examples_dir)):
                print(f"  {entry}")
        raise RuntimeError(f"{filename} not found in cloned repo")

    print(f"found {len(candidates)} candidate(s); using: {candidates[0]}")
    return candidates[0]


def _make_prompt(tok, n_tokens: int) -> str:
    seed = "The quick brown fox jumps over the lazy dog. "
    seed_ids = tok.encode(seed, add_special_tokens=False)
    n_copies = (n_tokens // len(seed_ids)) + 1
    ids = (seed_ids * n_copies)[:n_tokens]
    return tok.decode(ids)


def _wait_local_ready(timeout_s: int) -> None:
    """Poll trtllm-serve's /v1/models until it returns 200."""
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
    raise RuntimeError(f"trtllm-serve not ready in {timeout_s} s")


def _post_local(path: str, payload: dict) -> bytes:
    import json
    import urllib.request

    req = urllib.request.Request(
        f"http://localhost:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=120).read()

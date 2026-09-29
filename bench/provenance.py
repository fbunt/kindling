"""Run fingerprint (what a benchmark number depends on) and resume drift checks.

run_meta.json records a fingerprint when a run dir is created. Resuming into
that dir recomputes it and refuses if any DRIFT_KEYS value changed (unless
--allow-drift), so one run's traces never silently mix two prompts, question
sets, datasets, images or SDKs. A change of an IDENTITY_KEYS value (model,
dataset) is refused even with --allow-drift: one run dir is one model on one
dataset.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata

from app.config import LITE_MODEL
from app.genai_client import use_vertex
from bench.questions import Question

# Keys whose change between the first and a resumed session is drift.
# question_shas is compared per overlapping id (see check_drift).
DRIFT_KEYS = (
    "model",
    "parquet_identity",
    "prompt_sha",
    "max_rounds",
    "git_sha",
    "git_dirty",
    "backend",
    "guard_model",
    "versions",
    "worker_image",
    "sandbox",
    "http_options",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def prompt_sha(system_instruction: str, tool) -> str:
    """sha256 over the system prompt plus the tool declarations actually sent."""
    tool_json = json.dumps(
        tool.model_dump(mode="json", exclude_none=True), sort_keys=True
    )
    return _sha(system_instruction + "\n" + tool_json)[:16]


def question_sha(q: Question) -> str:
    """What the model sees and how it is judged (reference code has its own
    reference_sha, tied to ground truth)."""
    payload = {
        "id": q.id,
        "text": q.text,
        "answer_kind": q.answer_kind,
        "criterion": q.criterion,
        "tolerance_rel": q.tolerance_rel,
        "tolerance_abs": q.tolerance_abs,
    }
    return _sha(json.dumps(payload, sort_keys=True))[:12]


def question_set_sha(shas: dict[str, str]) -> str:
    return _sha(json.dumps(shas, sort_keys=True))[:12]


def git_info(cwd: str | None = None) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout

    try:
        sha = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain", "--untracked-files=no").strip())
        diff = git("diff", "HEAD") if dirty else ""
    except (OSError, subprocess.CalledProcessError) as e:
        return {"git_sha": None, "git_dirty": None, "git_error": str(e)}
    return {
        "git_sha": sha,
        "git_dirty": dirty,
        "git_diff_sha": _sha(diff)[:12] if dirty else None,
    }


def package_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("google-genai", "polars", "pyarrow", "httpx"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _parse_pids(value: str | None) -> int | None:
    if value is None:
        return 8192
    if value.strip().lower() in ("0", "none", "unlimited"):
        return None
    return int(value)


def sandbox_settings_from_env(environ=None) -> dict:
    """SandboxPool kwargs from the KINDLING_* env, mirroring app/main.py's
    lifespan (same knobs, same defaults) so the bench measures the production
    sandbox. Orphan reaping stays off: a bench must never kill a running app's
    warm workers."""
    env = os.environ if environ is None else environ
    cpus = env.get("KINDLING_SANDBOX_CPUS") or None
    return {
        "size": int(env.get("KINDLING_POOL_SIZE", "2")),
        "max_total": int(env.get("KINDLING_SANDBOX_MAX_TOTAL", "3")),
        "image": env.get("KINDLING_SANDBOX_IMAGE", "kindling-worker:latest"),
        "memory": env.get("KINDLING_SANDBOX_MEM", "110g"),
        "cpus": cpus,
        "pids": _parse_pids(env.get("KINDLING_SANDBOX_PIDS")),
        "max_threads": max(1, int(float(cpus))) if cpus else None,
        "worker_parquet_path": env.get("KINDLING_WORKER_PARQUET_PATH"),
    }


def effective_sandbox(pool, settings: dict) -> dict:
    """What the pool will actually run with (env settings + pool defaults)."""
    return {
        **settings,
        "runtime": pool.runtime,
        "hard_timeout": pool.hard_timeout,
        "checkout_timeout": pool.checkout_timeout,
        "worker_parquet_path": pool.worker_parquet_path,
    }


def image_id(runtime: str, image: str) -> str | None:
    """The image's content id, or None if the host image store lacks it."""
    try:
        out = subprocess.run(
            [runtime, "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


_WORKER_VERSIONS_PY = (
    "import json, platform, polars, pyarrow, numpy; "
    "print(json.dumps({'python': platform.python_version(), "
    "'polars': polars.__version__, 'pyarrow': pyarrow.__version__, "
    "'numpy': numpy.__version__}))"
)


def worker_versions(runtime: str, image: str) -> dict:
    """Library versions inside the worker image (one throwaway container)."""
    try:
        out = subprocess.run(
            [
                runtime,
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "python",
                image,
                "-c",
                _WORKER_VERSIONS_PY,
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as e:
        return {"error": f"{type(e).__name__}: {e}"}


def build_fingerprint(
    *,
    model: str,
    parquet_identity: str,
    prompt: str,
    question_shas: dict[str, str],
    max_rounds: int,
    sandbox: dict,
    worker_image: dict,
    http_options: dict,
) -> dict:
    return {
        "model": model,
        "parquet_identity": parquet_identity,
        "prompt_sha": prompt,
        "question_shas": dict(question_shas),
        "question_set_sha": question_set_sha(question_shas),
        "max_rounds": max_rounds,
        **git_info(),
        "backend": "vertex" if use_vertex() else "developer",
        "guard_model": LITE_MODEL,
        "versions": package_versions(),
        "worker_image": worker_image,
        "sandbox": sandbox,
        "http_options": http_options,
        "argv": sys.argv[1:],
    }


# Keys --allow-drift can never waive: resume skips existing traces without
# checking their model, and their expected values come from that dataset.
IDENTITY_KEYS = ("model", "parquet_identity")


def identity_drift(existing: dict, current: dict) -> list[str]:
    """Differences in IDENTITY_KEYS (a missing key counts as a change)."""
    return [
        f"{key}: {existing.get(key)!r} -> {current.get(key)!r}"
        for key in IDENTITY_KEYS
        if existing.get(key) != current.get(key)
    ]


def check_drift(existing: dict, current: dict) -> list[str]:
    """Human-readable differences that make resuming into `existing` unsafe."""
    diffs = []
    for key in DRIFT_KEYS:
        if key not in existing:  # pre-fingerprint run_meta: can't compare
            diffs.append(f"{key}: missing from run_meta (now {current.get(key)!r})")
        elif existing[key] != current.get(key):
            diffs.append(f"{key}: {existing[key]!r} -> {current.get(key)!r}")
    old_q = existing.get("question_shas") or {}
    for qid, sha in current.get("question_shas", {}).items():
        if qid in old_q and old_q[qid] != sha:
            diffs.append(f"question {qid}: {old_q[qid]} -> {sha}")
    return diffs

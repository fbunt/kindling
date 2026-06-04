# Sandbox worker image for kindling query execution.
# Build: podman build -t kindling-worker:latest -f Containerfile .
#
# polars/numpy/matplotlib/seaborn/pyarrow are pinned from uv.lock so core query
# semantics match the host's schema reads. pandas/scipy/scikit-learn are the
# model's extra analysis toolkit (container-only — the host never executes
# queries). Python is pinned to the host venv patch version (3.12.13).
FROM python:3.12.13-slim

RUN pip install --no-cache-dir \
        polars==1.38.1 \
        numpy==2.4.2 \
        matplotlib==3.10.8 \
        seaborn==0.13.2 \
        pyarrow==23.0.1 \
        pandas==3.0.3 \
        scipy==1.17.1 \
        scikit-learn==1.9.0 \
        tabulate==0.9.0

ENV MPLBACKEND=Agg \
    XDG_CACHE_HOME=/tmp \
    POLARS_TEMP_DIR=/tmp \
    PYTHONUNBUFFERED=1
# POLARS_MAX_THREADS is intentionally NOT pinned here: the pool passes it via -e
# only when a CPU cap is configured (KINDLING_SANDBOX_CPUS); otherwise polars
# auto-detects and uses all host cores.

# Pre-compile stdlib/site so first-query latency under the read-only root
# filesystem stays low. (matplotlib's MPLCONFIGDIR is chosen at runtime by the
# worker, since the runtime tmpfs can't reuse a baked, root-owned cache dir.)
RUN python -m compileall -q /usr/local/lib/python3.12

COPY app/sandbox/worker.py /worker.py

# Run as a non-root user as a second line of defense behind the container itself.
RUN useradd -u 1000 -m runner
USER runner

# Isolated mode: ignore PYTHON* env injection and user site-packages.
ENTRYPOINT ["python", "-I", "/worker.py"]

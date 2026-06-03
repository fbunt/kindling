# Sandbox worker image for kindling query execution.
# Build: podman build -t kindling-worker:latest -f Containerfile .
#
# Versions are pinned VERBATIM from uv.lock so in-container query semantics match
# what the eval suite validates. Python is pinned to the host venv's patch version
# (3.12.13) so to_dicts() float/repr formatting matches the in-process path.
FROM python:3.12.13-slim

RUN pip install --no-cache-dir \
        polars==1.38.1 \
        numpy==2.4.2 \
        matplotlib==3.10.8 \
        seaborn==0.13.2 \
        pyarrow==23.0.1

ENV MPLBACKEND=Agg \
    XDG_CACHE_HOME=/tmp \
    POLARS_TEMP_DIR=/tmp \
    POLARS_MAX_THREADS=4 \
    PYTHONUNBUFFERED=1

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

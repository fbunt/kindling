# kindling app (server) image — Option A.
#
# Runs the FastAPI orchestrator. It does NOT execute queries; it spawns sibling
# worker containers on the HOST runtime via a mounted socket (CONTAINER_HOST).
# So this image only needs the web stack + google-genai + polars/pyarrow (for
# schema reads) + a runtime *client*. The heavy analysis libs live in the worker
# image (Containerfile), never here.
#
# Build:  podman build -t kindling-app:latest -f Containerfile.app .
# Run:    see compose.yaml / deploy/kindling.container / the README.
FROM fedora:41

# podman-remote: a lightweight client that drives the host's Podman over the
# socket. For Docker hosts, also install a docker CLI (e.g. drop the static
# docker binary into /usr/local/bin) and run with KINDLING_CONTAINER_RUNTIME=docker.
RUN dnf install -y --setopt=install_weak_deps=False \
        python3 python3-pip podman-remote \
    && dnf clean all
# The package ships the binary as `podman-remote`; the app shells out to `podman`
# (and detect_runtime / build_run_argv key off the name "podman" for --userns=keep-id).
# As the remote-only client it forwards to the host via CONTAINER_HOST.
RUN ln -s /usr/bin/podman-remote /usr/local/bin/podman

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml ./
COPY app ./app
# Pin polars/pyarrow to the worker versions so the app's schema reads match the
# semantics of query execution in the worker.
RUN python3 -m pip install . polars==1.38.1 pyarrow==23.0.1

EXPOSE 8000

# The parquet is mounted at /data/dataset.parquet (app's view, for schema reads);
# KINDLING_WORKER_PARQUET_PATH gives the HOST path bind-mounted into sibling
# workers. Bind 0.0.0.0 so the published port is reachable.
CMD ["kindling", "/data/dataset.parquet", "--host", "0.0.0.0"]

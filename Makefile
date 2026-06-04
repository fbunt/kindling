# kindling — build/run helpers. Override RUNTIME=docker to use Docker.
RUNTIME      ?= podman
IMAGE_WORKER ?= kindling-worker:latest
IMAGE_APP    ?= kindling-app:latest
PARQUET      ?= $(shell readlink -f data/mtbs_pix_data.parquet)

.PHONY: build build-worker build-app socket run stop logs compose-up compose-down

build: build-worker build-app

build-worker:
	$(RUNTIME) build -t $(IMAGE_WORKER) -f Containerfile .

build-app:
	$(RUNTIME) build -t $(IMAGE_APP) -f Containerfile.app .

socket:   ## enable the rootless podman socket (one-time, podman only)
	systemctl --user enable --now podman.socket

run:      ## run the app; workers spawn as siblings on the host runtime
	$(RUNTIME) run -d --name kindling -p 8000:8000 \
	  -v $(XDG_RUNTIME_DIR)/podman/podman.sock:/run/podman/podman.sock \
	  -e CONTAINER_HOST=unix:///run/podman/podman.sock \
	  -v $(PARQUET):/data/dataset.parquet:ro \
	  -e KINDLING_WORKER_PARQUET_PATH=$(PARQUET) \
	  -e KINDLING_SANDBOX_IMAGE=$(IMAGE_WORKER) \
	  -e GEMINI_API_KEY=$(GEMINI_API_KEY) \
	  -e PYTHONUNBUFFERED=1 \
	  --security-opt label=disable \
	  $(IMAGE_APP) kindling /data/dataset.parquet --host 0.0.0.0

stop:
	-$(RUNTIME) rm -f kindling

logs:
	$(RUNTIME) logs -f kindling

compose-up:
	KINDLING_PARQUET=$(PARQUET) $(RUNTIME) compose up --build

compose-down:
	$(RUNTIME) compose down

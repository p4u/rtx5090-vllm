#!/usr/bin/env bash
# Check that this host can actually run the stack, BEFORE anything long starts.
#
# `make up` runs this first so a missing dependency fails in two seconds with
# an instruction, instead of twenty minutes into a CUDA build. Run it on its
# own any time: ./preflight.sh (or `make preflight`).
#
# Exit 0 = good to go (warnings may still be printed). Exit 1 = something
# required is missing; the output says exactly what to install or set.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Disk needed for a first run, per location (see the disk section): weights for
# the biggest model ~22 GB; images = llama.cpp build ~10 GB + vLLM ~9 GB.
MIN_WEIGHTS_GB="${MIN_WEIGHTS_GB:-25}"
MIN_IMAGES_GB="${MIN_IMAGES_GB:-20}"

FAILED=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; FAILED=1; }
fix()  { printf '      → %s\n' "$1"; }

echo "preflight — checking this host can serve the stack"
echo ""

# ─── command-line tools ─────────────────────────────────────────────────────
for tool in docker curl git python3; do
  if command -v "$tool" >/dev/null 2>&1; then
    ok "$tool found"
  else
    bad "$tool is not installed"
    case "$tool" in
      docker)  fix "install Docker Engine: https://docs.docker.com/engine/install/" ;;
      git)     fix "needed to clone the llama.cpp fork — apt install git" ;;
      *)       fix "apt install $tool" ;;
    esac
  fi
done

# Optional: the scripts degrade gracefully without these.
command -v jq >/dev/null 2>&1 \
  && ok "jq found" \
  || warn "jq missing — test-chat.sh / bench-ctx.sh need it (apt install jq)"
command -v hf >/dev/null 2>&1 \
  && ok "hf CLI found (fast weight downloads)" \
  || warn "hf CLI missing — downloads fall back to a slower Docker path (pip install 'huggingface_hub[hf_xet]')"

# ─── docker daemon ──────────────────────────────────────────────────────────
if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    ok "docker daemon reachable"
  else
    bad "cannot talk to the docker daemon"
    if ! docker info 2>&1 | grep -qi 'permission denied'; then
      fix "start it: sudo systemctl start docker"
    else
      fix "add yourself to the docker group: sudo usermod -aG docker $USER"
      fix "then log out and back in (or: newgrp docker)"
    fi
  fi
fi

# ─── GPU ────────────────────────────────────────────────────────────────────
# Host driver first: a driver/library mismatch (package upgraded, host not
# rebooted) breaks every NEW GPU container while already-running ones keep
# working — so `docker ps` looks healthy and only new launches fail.
if command -v nvidia-smi >/dev/null 2>&1; then
  if smi_out=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1); then
    ok "GPU: ${smi_out%%,*} ($(printf '%s' "$smi_out" | cut -d, -f2 | tr -d ' '))"
  else
    bad "nvidia-smi failed: $(printf '%s' "$smi_out" | head -1)"
    if printf '%s' "$smi_out" | grep -qi 'version mismatch'; then
      fix "the driver was upgraded without a reboot — reboot to load the new kernel module"
    else
      fix "check the NVIDIA driver install (CUDA 12.8+ is needed for Blackwell sm_120)"
    fi
  fi
else
  bad "nvidia-smi not found — no NVIDIA driver?"
  fix "install a driver with CUDA 12.8+ support (Blackwell sm_120)"
fi

# Container toolkit: prove it by running a GPU container, but only with an
# image that is already local — pulling one just to check would be rude.
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  probe=""
  for img in bonsai-llamacpp:prism vllm/vllm-openai:latest; do
    docker image inspect "$img" >/dev/null 2>&1 && { probe="$img"; break; }
  done
  if [[ -n "$probe" ]]; then
    if err=$(docker run --rm --gpus all --entrypoint true "$probe" 2>&1); then
      ok "docker can start GPU containers"
    else
      bad "docker cannot start a GPU container"
      printf '      %s\n' "$(printf '%s' "$err" | tail -1)"
      if printf '%s' "$err" | grep -q 'nvidia-persistenced'; then
        fix "nvidia-persistenced is down (usually a driver upgrade without a reboot) — reboot"
      else
        fix "install/configure the NVIDIA Container Toolkit, then: sudo systemctl restart docker"
        fix "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
      fi
    fi
  elif docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
    ok "nvidia container runtime registered (first launch will prove it)"
  else
    bad "the nvidia runtime is not registered with docker"
    fix "install the NVIDIA Container Toolkit, then: sudo nvidia-ctk runtime configure --runtime=docker"
    fix "and restart docker: sudo systemctl restart docker"
  fi
fi

# ─── configuration ──────────────────────────────────────────────────────────
# UI_PASSWORD gates the web UI; run-ui.sh refuses to start without it. A real
# environment variable wins over .env, same as everywhere else in the repo.
env_pw=$(sed -n 's/^[[:space:]]*\(export \)\?UI_PASSWORD=//p' .env 2>/dev/null | tail -1 | tr -d '"'"'"'' )
if [[ -n "${UI_PASSWORD:-}" || -n "$env_pw" ]]; then
  ok "UI_PASSWORD is set"
else
  bad "UI_PASSWORD is not set — the web UI will not start without it"
  [[ -f .env ]] || fix "create your config: cp .env.example .env"
  fix "then set UI_PASSWORD=<something real> in .env"
fi

# ─── disk ───────────────────────────────────────────────────────────────────
# Two places fill up: cache/ (weights) and Docker's data root (images — vLLM
# ~9 GB, the llama.cpp build ~10 GB). They are often the same disk, sometimes
# not, so check each where it actually lives. Low space is only FATAL on a host
# with nothing provisioned yet: once the UI image and some weights exist, `make`
# mostly re-checks a running stack, and refusing that over free space would
# break re-runs. run.sh/download-model.sh still fail loudly if a later download
# really does not fit.
disk_free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
disk_dev()     { df --output=source "$1" 2>/dev/null | tail -1; }
provisioned=""
if command -v docker >/dev/null 2>&1 && docker image inspect vllm-ui >/dev/null 2>&1 \
   && compgen -G "cache/models--*" >/dev/null; then
  provisioned=1
fi
low_disk() {
  if [[ -n "$provisioned" ]]; then
    warn "$1 (fine for what is already here; a NEW model download may not fit)"
  else
    bad "$1"
    fix "free space, or move it: symlink cache/ to a bigger disk (ln -s /big/disk cache),"
    fix "or relocate Docker's data-root (/etc/docker/daemon.json)"
  fi
}
cache_path="cache"; [[ -e "$cache_path" ]] || cache_path="."
docker_root="$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || true)"
cache_free=$(disk_free_gb "$cache_path")
if [[ -n "$docker_root" && "$(disk_dev "$docker_root")" != "$(disk_dev "$cache_path")" ]]; then
  docker_free=$(disk_free_gb "$docker_root")
  if [[ -n "$cache_free" ]]; then
    (( cache_free >= MIN_WEIGHTS_GB )) && ok "disk (weights, $cache_path): ${cache_free} GB free" \
      || low_disk "only ${cache_free} GB free for weights in $cache_path — need about ${MIN_WEIGHTS_GB} GB"
  fi
  if [[ -n "$docker_free" ]]; then
    (( docker_free >= MIN_IMAGES_GB )) && ok "disk (images, $docker_root): ${docker_free} GB free" \
      || low_disk "only ${docker_free} GB free for images in $docker_root — need about ${MIN_IMAGES_GB} GB"
  fi
elif [[ -n "$cache_free" ]]; then
  need=$(( MIN_WEIGHTS_GB + MIN_IMAGES_GB ))
  (( cache_free >= need )) && ok "disk: ${cache_free} GB free (weights + images share it)" \
    || low_disk "only ${cache_free} GB free — need about ${need} GB for weights + images"
fi

echo ""
if (( FAILED )); then
  echo "preflight FAILED — fix the ✗ items above, then run it again." >&2
  exit 1
fi
echo "preflight passed."

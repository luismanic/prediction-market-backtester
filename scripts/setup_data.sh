#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-data}"
DATA_ARCHIVE="${DATA_ARCHIVE:-data.tar.zst}"
DATA_URL="${DATA_URL:-}"
DATA_SHA256="${DATA_SHA256:-}"
SETUP_MARKER="${DATA_DIR}/.setup_complete"

if [[ -f "$SETUP_MARKER" ]]; then
  echo "[setup] dataset already initialized at ${DATA_DIR} (marker found)"
  exit 0
fi

if [[ ! -f "$DATA_ARCHIVE" ]]; then
  if [[ -z "$DATA_URL" ]]; then
    echo "[setup] DATA_URL is required when ${DATA_ARCHIVE} is missing"
    echo "[setup] set DATA_URL in .env or environment"
    exit 1
  fi

  tmp_archive="${DATA_ARCHIVE}.part"
  echo "[setup] downloading dataset archive from ${DATA_URL}"

  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 5 --retry-delay 2 --output "$tmp_archive" "$DATA_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=5 --wait=2 --output-document="$tmp_archive" "$DATA_URL"
  else
    echo "[setup] curl or wget is required"
    exit 1
  fi

  mv "$tmp_archive" "$DATA_ARCHIVE"
fi

if [[ -n "$DATA_SHA256" ]]; then
  echo "[setup] verifying sha256"
  echo "${DATA_SHA256}  ${DATA_ARCHIVE}" | sha256sum --check --status
fi

echo "[setup] validating archive paths"
if tar --zstd -tf "$DATA_ARCHIVE" | grep -Eq '(^/|(^|/)\.\.(/|$))'; then
  echo "[setup] archive contains unsafe paths"
  exit 1
fi

echo "[setup] extracting ${DATA_ARCHIVE}"
mkdir -p "$DATA_DIR"
tar --zstd -xf "$DATA_ARCHIVE" -C .

touch "$SETUP_MARKER"
echo "[setup] complete"

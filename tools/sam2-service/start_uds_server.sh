#!/usr/bin/env bash
# Start the resident EfficientViT segmentation and Efficient-SAM2 tracking service.
#
# Optional overrides:
#   SAM2_SERVICE_SOCKET=/run/user/$UID/sam2-service.sock
#   SAM2_SERVICE_BACKEND=sam2
#   SAM2_SERVICE_QUEUE_CAPACITY=8
#   SAM2_SERVICE_REQUEST_TIMEOUT_S=90
#   SAM2_SERVICE_SOCKET_MODE=660

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
LEGACY_PYTHON="${SCRIPT_DIR}/../../tool/sam2-service/.venv/bin/python"
if [[ -x "${LOCAL_PYTHON}" ]]; then
    PYTHON_SERVICE="${LOCAL_PYTHON}"
elif [[ -x "${LEGACY_PYTHON}" ]]; then
    # Transitional interpreter reuse only: PYTHONPATH below forces all SAM2
    # service code and checkpoints to come from tools/sam2-service.
    PYTHON_SERVICE="${LEGACY_PYTHON}"
else
    echo "sam2-service Python environment is unavailable" >&2
    echo "Run: cd ${SCRIPT_DIR} && uv sync --python 3.10" >&2
    exit 1
fi

# Do not use /tmp here. The GCS process runs in a restricted identity and a
# socket it creates in sticky /tmp cannot later be removed by dainanjun.
# This directory is non-sticky so the GCS service identity and dainanjun can
# both replace a stale socket. It is dedicated to this service, not /tmp.
RUNTIME_DIR="${SAM2_SERVICE_RUNTIME_DIR:-${SCRIPT_DIR}/runtime}"
SOCKET_PATH="${SAM2_SERVICE_SOCKET:-${RUNTIME_DIR}/sam2-service.sock}"
PID_FILE="${SAM2_SERVICE_PID_FILE:-${RUNTIME_DIR}/sam2-service.pid}"
BACKEND="${SAM2_SERVICE_BACKEND:-efficientvit}"
QUEUE_CAPACITY="${SAM2_SERVICE_QUEUE_CAPACITY:-4}"
REQUEST_TIMEOUT_S="${SAM2_SERVICE_REQUEST_TIMEOUT_S:-60}"
SOCKET_MODE="${SAM2_SERVICE_SOCKET_MODE:-666}"

# PID 文件由服务在加载模型前写入。这个检查兼容升级前已经运行、尚未持有
# 单实例锁的服务，避免新版 GCS 第一次启动时再复制一套模型。
if [[ -r "${PID_FILE}" ]]; then
    EXISTING_PID="$(head -n 1 "${PID_FILE}" 2>/dev/null || true)"
    if [[ "${EXISTING_PID}" =~ ^[0-9]+$ ]] \
       && kill -0 "${EXISTING_PID}" 2>/dev/null \
       && tr '\0' ' ' < "/proc/${EXISTING_PID}/cmdline" 2>/dev/null | grep -q 'sam2-service'; then
        echo "[sam2-service] refusing duplicate start: PID ${EXISTING_PID} is already running" >&2
        exit 75
    fi
fi

echo "[sam2-service] starting resident segmentation + tracking service"
echo "[sam2-service] python: ${PYTHON_SERVICE}"
echo "[sam2-service] backend: ${BACKEND}"
echo "[sam2-service] socket: ${SOCKET_PATH} (mode ${SOCKET_MODE})"
echo "[sam2-service] segmentation queue capacity: ${QUEUE_CAPACITY}"
echo "[sam2-service] request timeout: ${REQUEST_TIMEOUT_S}s"
echo "[sam2-service] loading models; this can take a few minutes on first start..."

if [[ ! -d "${RUNTIME_DIR}" ]]; then
    mkdir -p "${RUNTIME_DIR}" || true
    chmod 777 "${RUNTIME_DIR}" 2>/dev/null || true
fi
if [[ ! -d "${RUNTIME_DIR}" || ! -w "${RUNTIME_DIR}" ]]; then
    echo "SAM2 runtime directory is unavailable or not writable: ${RUNTIME_DIR}" >&2
    echo "Initialize it once with: mkdir -p ${RUNTIME_DIR} && chmod 777 ${RUNTIME_DIR}" >&2
    exit 1
fi

ARGS=(
    --socket "${SOCKET_PATH}" \
    --backend "${BACKEND}" \
    --queue-capacity "${QUEUE_CAPACITY}" \
    --request-timeout-s "${REQUEST_TIMEOUT_S}" \
    --socket-mode "${SOCKET_MODE}" \
    --pid-file "${PID_FILE}"
)

export PYTHONPATH="${SCRIPT_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_SERVICE}" -m sam2_service.uds_http_service "${ARGS[@]}"

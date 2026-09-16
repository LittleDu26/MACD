#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_DIR="${REPO_ROOT}/simulator"
BUILD_DIR="${MACD_SIMULATOR_BUILD_DIR:-${REPO_ROOT}/build/simulator}"
OUTPUT_DIR="${MACD_SIMULATOR_OUTPUT_DIR:-${REPO_ROOT}/evogym}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v cmake >/dev/null 2>&1; then
    echo "Error: cmake is required to build the simulator." >&2
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Error: Python executable '${PYTHON_BIN}' was not found." >&2
    exit 1
fi

PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
PYTHON_VERSION="$(${PYTHON_BIN} -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))')"
if [[ "${PYTHON_VERSION}" != "3.7" ]]; then
    echo "Error: this repository requires CPython 3.7; found ${PYTHON_VERSION}." >&2
    echo "Activate the documented Conda environment and run this script again." >&2
    exit 1
fi

if [[ ! -f "${SOURCE_DIR}/CMakeLists.txt" ]]; then
    echo "Error: bundled simulator sources are missing from ${SOURCE_DIR}." >&2
    exit 1
fi

if [[ -n "${MACD_BUILD_JOBS:-}" ]]; then
    BUILD_JOBS="${MACD_BUILD_JOBS}"
elif command -v nproc >/dev/null 2>&1; then
    BUILD_JOBS="$(nproc)"
else
    BUILD_JOBS=1
fi

mkdir -p "${BUILD_DIR}" "${OUTPUT_DIR}"

cmake \
    -S "${SOURCE_DIR}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${OUTPUT_DIR}" \
    -DPYTHON_EXECUTABLE="${PYTHON_BIN}"

cmake --build "${BUILD_DIR}" --config Release -- -j"${BUILD_JOBS}"

EXT_SUFFIX="$(${PYTHON_BIN} -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")')"
OUTPUT_FILE="${OUTPUT_DIR}/simulator_cpp${EXT_SUFFIX}"

if [[ ! -f "${OUTPUT_FILE}" ]]; then
    echo "Error: build completed but ${OUTPUT_FILE} was not produced." >&2
    exit 1
fi

echo "Built simulator extension: ${OUTPUT_FILE}"

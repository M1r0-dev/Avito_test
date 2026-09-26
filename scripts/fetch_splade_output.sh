#!/usr/bin/env bash
set -euo pipefail

KERNEL_REF="m1r0tvorxc/avito-russian-splade-candidate-retrieval"
OUTPUT_DIR="artifacts/splade_kaggle"
mkdir -p "$OUTPUT_DIR"
kaggle kernels output "$KERNEL_REF" -p "$OUTPUT_DIR" --force

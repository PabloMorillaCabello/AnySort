#!/bin/bash
# =============================================================================
# Download model weights for the GraspGen pipeline.
# Run inside the Docker container or on host with huggingface-cli installed.
#
# Usage:
#   export HF_TOKEN=hf_your_token_here
#   ./scripts/download_models.sh
#
# Prerequisites:
#   - Request access to facebook/sam3 on HuggingFace
#   - Request access to adithyamurali/GraspGenModels on HuggingFace
# =============================================================================
set -e

MODELS_DIR="${MODELS_DIR:-/opt/models}"

echo "============================================"
echo "  GraspGen Thesis - Model Downloader"
echo "  Target dir: ${MODELS_DIR}"
echo "============================================"

# --- Check HF Token ---
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "ERROR: HF_TOKEN is not set."
    echo "  1. Go to https://huggingface.co/settings/tokens"
    echo "  2. Create a token with 'read' access"
    echo "  3. export HF_TOKEN=hf_your_token_here"
    echo "  4. Re-run this script"
    exit 1
fi

# Login to HuggingFace
echo ""
echo "Authenticating with HuggingFace..."
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true

# --- SAM3 ---
echo ""
echo "[1/2] Downloading SAM3 model (facebook/sam3)..."
SAM3_DIR="${MODELS_DIR}/sam3"
mkdir -p "$SAM3_DIR"
huggingface-cli download facebook/sam3 \
    --local-dir "$SAM3_DIR" \
    --token "$HF_TOKEN"
echo "  -> SAM3 weights saved to: ${SAM3_DIR}"

# --- GraspGen ---
echo ""
echo "[2/2] Downloading GraspGen models (adithyamurali/GraspGenModels)..."
GRASPGEN_DIR="${MODELS_DIR}/graspgen"
mkdir -p "$GRASPGEN_DIR"
huggingface-cli download adithyamurali/GraspGenModels \
    --local-dir "$GRASPGEN_DIR" \
    --token "$HF_TOKEN"
echo "  -> GraspGen weights saved to: ${GRASPGEN_DIR}"

echo ""
echo "============================================"
echo "  All models downloaded successfully!"
echo ""
echo "  SAM3:     ${SAM3_DIR}"
echo "  GraspGen: ${GRASPGEN_DIR}"
echo ""
echo "  Available GraspGen gripper configs:"
ls -1 "${GRASPGEN_DIR}"/*.yml 2>/dev/null || echo "    (check subdirectories)"
echo "============================================"

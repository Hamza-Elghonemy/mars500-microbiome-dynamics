#!/bin/bash
# ==============================================================================
# M8: PICRUSt2 Automated Functional Prediction Environment (Bash)
# MARS500 Temporal Gut Microbiome Dynamics
# ==============================================================================
# Resolves the complex PICRUSt2 dependencies by automatically instantiating a
# strictly pure miniconda envelope, processing the biological sequences,
# and yielding MetaCyc pathway profiles.
# ==============================================================================

set -e

echo "Bootstrapping PICRUSt2 Analytical Environment..."

# Detect if conda is available
if ! command -v conda &> /dev/null
then
    echo "[ERROR] Conda could not be found. Please install Anaconda or Miniconda first."
    exit 1
fi

ENV_NAME="picrust2_env"

# Check if environment already exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "Found existing $ENV_NAME environment."
else
    echo "Creating isolated $ENV_NAME environment (using x86 emulation for Mac ARM)..."
    # PICRUSt2 relies heavily on bioconda which is missing gappa binaries for arm64
    CONDA_SUBDIR=osx-64 conda create -y -n $ENV_NAME -c bioconda -c conda-forge picrust2=2.5.2
    conda run -n $ENV_NAME conda config --env --set subdir osx-64
fi

echo "Activating $ENV_NAME..."
# Using dynamic eval to ensure conda initialize script works in raw bash
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "Executing Python Wrapper (05_functional_prediction.py)..."
python pipelines/05_functional_prediction.py

echo "PICRUSt2 functional predictions have successfully resolved!"
echo "Check the 'results/picrust2' directory for the Metacyc pathway outputs."
conda deactivate

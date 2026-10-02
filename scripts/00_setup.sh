#!/usr/bin/env bash
# One-time environment setup. Run from the vla-study/ folder.
set -euo pipefail
conda create -y -n vla python=3.12   # LeRobot requires Python >= 3.12 (check its install guide if this changes)
eval "$(conda shell.bash hook)" && conda activate vla
conda install -y -c conda-forge ffmpeg
git clone https://github.com/huggingface/lerobot.git ../lerobot
pip install -e "../lerobot[smolvla,libero]"
pip install -e .                       # this study's package (vla_study)
pip install matplotlib
( cd ../lerobot && echo "LeRobot commit: $(git rev-parse HEAD)" ) | tee LEROBOT_VERSION.txt
python -c "import lerobot, libero; print('lerobot + libero import OK')"
echo "Done. Put LEROBOT_VERSION.txt in your README so others can reproduce your results."

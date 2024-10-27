#!/bin/bash

module load python/3
source /home/u2hussai/dgcnn/.venv/bin/activate

python3 /home/u2hussai/dgcnn/scripts/training_script.py $1

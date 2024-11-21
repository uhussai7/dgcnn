#!/bin/bash

module load python/3
source /home/u2hussai/dgcnn/.venv/bin/activate

python3 /home/u2hussai/dgcnn/jobHandling/dtifit/dtifit_predictions.py -c $1 -v $2 -t $3

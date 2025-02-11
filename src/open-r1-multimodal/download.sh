#!/bin/bash
srun --gres=gpu:4 --partition=AI4Phys python download_model_and_datasets.py

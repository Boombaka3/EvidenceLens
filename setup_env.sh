#!/usr/bin/env bash
set -e
conda create -n evidencelens python=3.11 -y
conda activate evidencelens
pip install -r requirements.txt
cp .env.example .env
echo "Edit .env and add your ANTHROPIC_API_KEY before running the pipeline."

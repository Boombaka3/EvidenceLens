@echo off
conda create -n evidencelens python=3.11 -y
call conda activate evidencelens
pip install -r requirements.txt
copy .env.example .env
echo Edit .env and add your ANTHROPIC_API_KEY before running the pipeline.

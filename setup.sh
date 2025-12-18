mkdir log
sudo apt install python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install uv
uv pip install fastapi uvicorn transformers torch accelerate math_verify datasets openai verify
uv pip install vllm
cd affinetes
uv pip install -e .
cd ../
chmod +x eval_cde.sh
chmod +x eval_lgc.sh

git clone https://github.com/dashi-superai/RL-evaluation.git
cd RL-evaluation
mkdir log
sudo apt install python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install uv
uv pip install fastapi uvicorn transformers torch accelerate math_verify datasets
cd affinetes
uv pip install -e .
uv pip install openai
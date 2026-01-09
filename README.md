### 🛠️Setup
chmod +x setup.sh

./setup.sh

source .venv/bin/activate

### 📡Run LLM Server
➖Local server

python server.py --hug_url Sota26/Affine_M016 --local_dir model

➖vllm server

vllm serve model_name(huggingface_url) --max-model-len 10000 --gpu-memory-utilization 0.9 --host 0.0.0.0

### 📐Math evaluation
➖Local server

python eval_math.py

➖vllm server

./eval_math.sh

### 🔍Science evaluation
➖Local server

python eval_science.py

➖vllm server

./eval_sci.sh

### 📋Logic evaluation
./eval_lgc.sh

### 🧑‍💻Code evaluation
./eval_cde.sh

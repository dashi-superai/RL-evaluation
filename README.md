### 🛠️Setup
chmod +x setup.sh

./setup.sh

source .venv/bin/activate
### 📡Run LLM Server
python server.py --hug_url Sota26/Affine_M016 --local_dir model

### 📐Math evaluation
python eval_math.py

### 🔍Science evaluation
python eval_science.py

### 📋Logic evaluation
./eval_lgc.sh

### 🧑‍💻Code evaluation
./eval_cde.sh

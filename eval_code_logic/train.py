from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

MODEL_PATH = "./models2"
DATASET_NAME = "PrimeIntellect/INTELLECT-3-SFT"
DATASET_SUBSET = "openreasoning_science"
DIFFICULTY_KEY = "avg@8_qwen3_4b_thinking_2507"
DATASET_SPLIT = "train"

EVAL_DATASET_SIZE = 1000

# INSTRUCTION_PROMPT = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}."

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    dtype=torch.float32,
    device_map="auto",
)

dataset = load_dataset(DATASET_NAME, DATASET_SUBSET, split=DATASET_SPLIT)

def preprocess_function(example):
    # return {
    #     "prompt": [{"role": "user", "content": example["question"]}],
    #     "completion": [{"role": "assistant", "content": f"{example['answer']}"}],
    # }
    return example


dataset = dataset.map(preprocess_function, remove_columns=["source"])


n = len(dataset)
n = n // 16

import time
dataset.shuffle(seed=int(time.time()))

train_dataset = dataset.select(range(0, n - EVAL_DATASET_SIZE))
eval_dataset = dataset.select(range(n - EVAL_DATASET_SIZE, n))

train_dataset.save_to_disk("./train_dataset_sci")
eval_dataset.save_to_disk("./eval_dataset_sci")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.bos_token_id = tokenizer.pad_token_id

training_args = SFTConfig(
    output_dir="./sft_model",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    packing=True,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.25},
    load_best_model_at_end=True,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=200,
    save_steps=200,
    learning_rate=5e-6,
    optim="adamw_torch_fused",
    logging_steps=5,
    save_only_model=True,
    gradient_checkpointing=True,
    gradient_accumulation_steps=4,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    processing_class=tokenizer,
)

trainer.train(resume_from_checkpoint=True)

model.save_pretrained("./sft_model/model")
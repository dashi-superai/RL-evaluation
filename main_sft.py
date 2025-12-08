from datasets import load_dataset
from math_verify import parse, verify
async def llm_chat(prompt, temperature):
        """Call LLM API with specified API key and optional seed"""
        # Unset SSL_CERT_FILE to avoid certificate path issues in container
        # Let httpx/certifi use default certificate bundle
        # Prepare API call parameters
        import requests
        # Generate text
        # response = requests.post(
        #     "http://localhost:5000/generate",
        #     json={
        #         "prompt": prompt,
        #         "max_length": 100000,
        #         "temperature": temperature
        #     }
        # )
        params = {
            "prompt": prompt,
            "max_length": 30000,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(
            "http://localhost:5001/chat",
            json=params
        )
        return response
INSTRUCTION_PROMPT = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}."
async def math_main():
    datasets = load_dataset("PrimeIntellect/INTELLECT-3-SFT", "openreasoning_math")['train']
    print(len(datasets))
    import time
    import random
    random.seed(int(time.time()))
    for index in range(50):
        rand = random.randint(0, len(datasets))
        print(f"Processing data index : {rand}")
        data = datasets[rand]
        # prompt = data['question']
        # print('+' * 60)
        # print(prompt)
        # print('+' * 60)
        # print('+' * 60)
        prompt = datasets[rand]['prompt'][0]['content']
        truth = datasets[rand]['completion'][0]['content']
        # truth = data['answer']
        print(truth)
        print('+' * 60)
        truth_answer = parse(f"\\boxed{{{truth}}}", parsing_timeout=5)
        print(f"Ground Truth : {truth_answer}")
        print('+' * 60)
        score = 0.0
        resp = await llm_chat(INSTRUCTION_PROMPT + prompt, 0.7)
        data = resp.json()
        content = data["response"]
        content_answer = parse(f"\\boxed{{{content}}}", parsing_timeout=5)
        print(f"Model Answer : {content_answer}")
        print('<>' * 60)
        score = float(
            verify(
                parse(f"\\boxed{{{truth}}}", parsing_timeout=5),
                parse(f"\\boxed{{{content}}}", parsing_timeout=5),
                timeout_seconds=5,
            )
        )
        print('|' * 60)
        print(f"Score: {score}")
        print('=' * 60)
        # content = "1007"
        with open(f'./log/output-{index}.txt', 'w') as f:
            res = ""
            res += truth
            res += '+' * 60
            res += content
            res += '+' * 60
            res += f"Score : {score}"
            f.write(res)
if __name__ == "__main__":
    import asyncio
    asyncio.run(math_main())

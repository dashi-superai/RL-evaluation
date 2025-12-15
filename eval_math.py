from datasets import load_dataset
from math_verify import parse, verify
INSTRUCTION_PROMPT = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}."

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
            "max_length": 3000,
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


async def math_main():
    datasets = load_dataset("PrimeIntellect/INTELLECT-3-RL", "math")['train']
    print(len(datasets))

    import time
    import random
    random.seed(int(time.time()))
    idx = 5000
    cnt = 0
    
    for index in range(idx, idx + 1000, 5):
        # rand = random.randint(0, len(datasets))
        print(f"Processing data index : {index}")
        
        data = datasets[index]

        prompt =INSTRUCTION_PROMPT + data['question']

        # print('+' * 60)
        # print(prompt)
        # print('+' * 60)

        # print('+' * 60)
        # print(datasets[rand]['completion'][0]['content'])
        
        truth = data['answer']

        print(truth)
        print('+' * 60)
        truth_answer = parse(f"\\boxed{{{truth}}}", parsing_timeout=5)
        print(f"Ground Truth : {truth_answer}")
        print('+' * 60)
        
        score = 0.0
        
        resp = await llm_chat(prompt, 0.7)
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
        if score:
            cnt += 1
        # content = "1007"

        with open(f'./log/output-{index}.txt', 'w') as f:
            res = ""
            res += truth
            res += '+' * 60
            res += content
            res += '+' * 60
            res += f"Score : {score}"

            f.write(res)
    print(f"correct num: {cnt}/{len(range(idx, idx + 1000, 5))}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(math_main())

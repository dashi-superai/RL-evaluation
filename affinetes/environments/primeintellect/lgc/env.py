"""Logic Environment Actor"""

import os
import time
import gc
import httpx
# import openai
import sys
import random

# Add /app to path to import local modules
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from logic_task import LogicTask


class Actor:
    """Logic task evaluation actor"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Actor with API key
        
        Args:
            api_key: API key for LLM service. If not provided, will use CHUTES_API_KEY env var
        """
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        
        # Initialize logic task instance
        self.logic_task = LogicTask()
    
    async def _llm_chat(self, prompt, temperature):
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
            "max_length": 1000,
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
    
    async def evaluate(
        self,
        model="deepseek-ai/DeepSeek-V3",
        base_url="https://llm.chutes.ai/v1",
        timeout=600,
        temperature=0.7,
        api_key: str = None,
        seed: int = None,
        task_id: int = None
    ):
        """
        Run evaluation on a single logic task
        
        Args:
            model: Model name to use for evaluation
            base_url: Base URL for LLM API
            timeout: Timeout for LLM API calls
            temperature: Temperature for LLM generation
            api_key: Override API key for this evaluation. If not provided, uses instance api_key
            seed: Random seed for LLM generation. Used to ensure reproducible results. If not provided, a random seed will be generated.
            task_id: Optional task ID for deterministic task selection.
                     If provided, used as index into dataset.
                     If not provided, random sample is selected.
        """
        # Generate random seed if not provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Allow per-call api_key override
        current_api_key = api_key or self.api_key
        
        start = time.time()
        
        # Generate challenge
        challenge = await self.logic_task.generate(task_id=task_id)
        
        # Call LLM
        try:
            resp = await self._llm_chat(challenge.prompt, temperature)
            error = None
        except Exception as e:
            import traceback
            resp = None
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
        data = resp.json()
        print(data["response"])
        # Evaluate
        score = 0.0
        if resp:
            try:
                score = await self.logic_task.evaluate(data["response"], challenge)
            except Exception as e:
                import traceback
                error = f"Evaluation error: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        conversation = [
            {"role": "user", "content": challenge.prompt},
            {"role": "assistant", "content": resp}
        ]

        result = {
            "task_name": "logic:intellect-3-rl",
            "score": score,
            "success": score > 0,
            "time_taken": time.time() - start,
            "extra": {
                "conversation": conversation,
                "seed": seed,
                "task_type": challenge.extra.get("task_type", ""),
                "dataset_index": challenge.extra.get("dataset_index")
            }
        }
        
        # Add error info if present
        if error:
            result["error"] = error
            result["error_type"] = "llm_failure"

        # Force garbage collection to free memory immediately
        gc.collect()


        return result
    
    
async def main():
    actor = Actor()
    cnt = 0
    id = 8000
    for i in range(id, id + 1000, 5):
        result = await actor.evaluate(task_id = i)
        print(f"task id : {i} result: {result['score']}")
        if result['score']:
            cnt += 1
    print(f"correct num: {cnt}")
            
if __name__  == '__main__':
    import asyncio
    asyncio.run(main())
        

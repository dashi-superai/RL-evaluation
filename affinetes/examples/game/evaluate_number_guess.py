#!/usr/bin/env python3
"""
Evaluate Number Guessing interactive environment

Usage:
    python evaluate_number_guess.py
"""

import asyncio
import json
import sys
import os
import affinetes as af
from dotenv import load_dotenv

load_dotenv(override=True)

async def main():
    api_key = os.getenv("CHUTES_API_KEY")
    if not api_key:
        print("Error: CHUTES_API_KEY not set")
        sys.exit(1)

    image_tag = af.build_image_from_env(
        env_path="environments/game/number_guess",
        image_tag="game:guess",
        quiet=False
    )
    
    print("Loading environment...")
    env = af.load_env(
        image=image_tag,
        mode="docker",
        env_vars={"CHUTES_API_KEY": api_key},
    )
    print("Environment loaded\n")
    
    result = await env.evaluate(
        model="deepseek-ai/DeepSeek-V3",
        base_url="https://llm.chutes.ai/v1",
        task_id=100
    )
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
    

if __name__ == "__main__":
    asyncio.run(main())
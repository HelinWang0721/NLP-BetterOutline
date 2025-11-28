"""
Baseline Outline Generator - Minimal Prompting
Generates outlines without criteria-aware prompts for comparison
"""

import numpy as np
from typing import List, Tuple, Dict
from string import Template
import yaml
import openrouter_client

# Load configuration
with open("./config.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Simple baseline prompt (no criteria)
BASELINE_PROMPT = """请根据以下关键词，创作一个故事大纲：

必须包含的关键词：${mandatory_words}

可选的关键词：${optional_words}

要求：
- 至少${n_characters}个角色
- 至少${n_scenes}个场景  
- 至少${n_events}个事件

请包含以下部分：
## 故事氛围
## 故事背景
## 人设
## 场景
## 目的
## 高潮和结局
## 事件大纲
"""

DETAIL_PROMPT = """请为上述大纲补充详细内容，用多个段落详细描述每个事件。"""


def get_rand_n(min_n: int, max_n: int, percentage: float) -> int:
    """Get a random number from normal distribution"""
    mu = percentage * (max_n - min_n) + min_n
    sigma = (max_n - min_n) / 6
    n = np.random.normal(mu, sigma)
    n = np.clip(n, min_n, max_n)
    return int(n)


def generate_baseline_outline(
    mandatory_words: str,
    optional_words: str,
    max_retries: int = 3
) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Generate outline with minimal prompting (baseline)
    
    Args:
        mandatory_words: Mandatory keywords
        optional_words: Optional keywords
        max_retries: Maximum retries
        
    Returns:
        Tuple of (prompts, messages, responses)
    """
    # Generate random parameters
    rand_percentage = np.random.rand()
    n_characters = get_rand_n(
        config['generation']['min_characters'],
        config['generation']['max_characters'],
        rand_percentage
    )
    n_scenes = get_rand_n(
        config['generation']['min_scenes'],
        config['generation']['max_scenes'],
        rand_percentage
    )
    n_events = get_rand_n(
        config['generation']['min_events'],
        config['generation']['max_events'],
        rand_percentage
    )
    
    # Create simple prompt
    prompt = Template(BASELINE_PROMPT).substitute(
        mandatory_words=mandatory_words,
        optional_words=optional_words,
        n_characters=n_characters,
        n_scenes=n_scenes,
        n_events=n_events
    )
    
    # Get client
    client = openrouter_client.get_client()
    
    # Generate initial outline
    messages = [{"role": "user", "content": prompt}]
    responses = []
    
    print("Generating baseline outline (minimal prompting)...")
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.chat(messages)
            messages.append({"role": "assistant", "content": response})
            responses.append(response)
            print("✓ Baseline outline generated")
            break
        except Exception as e:
            print(f"Error: {e}")
            if attempt == max_retries - 1:
                raise
            attempt += 1
    
    # Generate details
    messages.append({"role": "user", "content": DETAIL_PROMPT})
    detail_response = client.chat(messages)
    messages.append({"role": "assistant", "content": detail_response})
    responses.append(detail_response)
    print("✓ Baseline details generated")
    
    prompts = [prompt, DETAIL_PROMPT]
    return prompts, messages, responses


if __name__ == "__main__":
    """Test baseline generation with random task from eval_tasks.json"""
    import json
    import random
    
    print("\n=== Testing Baseline Generation ===\n")
    
    # Load tasks from eval_tasks.json
    try:
        with open("./eval_tasks.json", 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        
        # Pick random task
        task = random.choice(tasks)
        m_words = task["m_words"]
        o_words = task["o_words"]
        
        print(f"Loaded random task from eval_tasks.json:")
        print(f"  Mandatory: {m_words}")
        print(f"  Optional: {o_words[:80]}...")
    except FileNotFoundError:
        print("eval_tasks.json not found, using default keywords")
    
    prompts, messages, responses = generate_baseline_outline(m_words, o_words)
    
    print("\n=== Baseline Outline (First 500 chars) ===\n")
    print(responses[0][:500] + "...\n")
    
    print("✓ Test completed!")

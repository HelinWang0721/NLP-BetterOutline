"""
Outline Generation with Criteria Hints
Uses criteria in prompts but NO self-evaluation or refinement
"""

import numpy as np
from typing import List, Tuple
from string import Template
import yaml
import logging
import openrouter_client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load prompts (contains criteria hints)
with open("./gen-prompt.md", 'r', encoding="utf-8") as f:
    prompt_content = f.read()
    parts = prompt_content.split("<EOF>")
    PROMPT_INITIAL = parts[0].strip() + "\n"
    PROMPT_DETAIL = parts[1].strip() + "\n" if len(parts) > 1 else ""

# Load configuration
with open("./config.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Generation ranges
RANGE_CHARACTERS = [
    config['generation']['min_characters'],
    config['generation']['max_characters']
]
RANGE_SCENES = [
    config['generation']['min_scenes'],
    config['generation']['max_scenes']
]
RANGE_EVENTS = [
    config['generation']['min_events'],
    config['generation']['max_events']
]


def get_rand_n(min_n: int, max_n: int, percentage: float) -> int:
    """Get random number from normal distribution"""
    mu = percentage * (max_n - min_n) + min_n
    sigma = (max_n - min_n) / 6
    n = np.random.normal(mu, sigma)
    n = np.clip(n, min_n, max_n)
    return int(n)


def get_prompt(mandatory_words: str, optional_words: str) -> Tuple[str, str]:
    """
    Generate prompts for outline creation (with criteria hints embedded)
    
    Args:
        mandatory_words: Mandatory keywords
        optional_words: Optional keywords
        
    Returns:
        Tuple of (initial_prompt, detail_prompt)
    """
    rand_percentage = np.random.rand()
    n_characters = get_rand_n(RANGE_CHARACTERS[0], RANGE_CHARACTERS[1], rand_percentage)
    n_scenes = get_rand_n(RANGE_SCENES[0], RANGE_SCENES[1], rand_percentage)
    n_events = get_rand_n(RANGE_EVENTS[0], RANGE_EVENTS[1], rand_percentage)
    
    initial_prompt = Template(PROMPT_INITIAL).substitute(
        mandatory_words=mandatory_words,
        optional_words=optional_words,
        n_characters=n_characters,
        n_scenes=n_scenes,
        n_events=n_events
    )
    
    return initial_prompt, PROMPT_DETAIL


def check_outline_completeness(outline: str) -> bool:
    """Check if outline contains all required sections"""
    required_sections = [
        "故事氛围", "故事背景", "人设", "场景", "目的", "高潮和结局", "事件大纲"
    ]
    
    for section in required_sections:
        if not (f"## {section}" in outline or f"#### {section}" in outline):
            logger.warning(f"Missing section: {section}")
            return False
    
    return True


def generate_outline_with_criteria(
    mandatory_words: str,
    optional_words: str,
    max_retries: int = 3
) -> Tuple[List[str], List[dict], List[str]]:
    """
    Generate outline with criteria hints in prompts
    (NO self-evaluation or refinement - just generation)
    
    Args:
        mandatory_words: Mandatory keywords
        optional_words: Optional keywords
        max_retries: Maximum retries
        
    Returns:
        Tuple of (prompts, messages, responses)
    """
    # Get prompts (contains criteria hints)
    initial_prompt, detail_prompt = get_prompt(mandatory_words, optional_words)
    
    # Get OpenRouter client
    client = openrouter_client.get_client()
    
    # Initialize conversation
    messages = []
    responses = []
    
    # Step 1: Generate initial outline with criteria hints
    logger.info("Generating outline with criteria hints...")
    messages.append({"role": "user", "content": initial_prompt})
    
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.chat(messages)
            
            if check_outline_completeness(response):
                logger.info("✓ Outline generated successfully")
                messages.append({"role": "assistant", "content": response})
                responses.append(response)
                break
            else:
                logger.warning(f"Outline incomplete, retrying ({attempt + 1}/{max_retries})...")
                attempt += 1
                if attempt == max_retries:
                    raise Exception("Failed to generate complete outline after retries")
        except Exception as e:
            logger.error(f"Error generating outline: {e}")
            if attempt == max_retries - 1:
                raise
            attempt += 1
    
    # Step 2: Generate detailed outline
    logger.info("Generating detailed outline...")
    messages.append({"role": "user", "content": detail_prompt})
    
    detail_response = client.chat(messages)
    messages.append({"role": "assistant", "content": detail_response})
    responses.append(detail_response)
    logger.info("✓ Detailed outline generated")
    
    prompts = [initial_prompt, detail_prompt]
    return prompts, messages, responses


# Backward compatibility
def generate_outline(mandatory_words: str, optional_words: str):
    """Legacy function for backward compatibility"""
    return generate_outline_with_criteria(mandatory_words, optional_words)


if __name__ == "__main__":
    """Test outline generation"""
    import json
    import random
    
    print("\n=== Testing Outline Generation  with Criteria Hints ===\n")
    
    # Load random task
    try:
        with open("./eval_tasks.json", 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        task = random.choice(tasks)
        m_words = task["m_words"]
        o_words = task["o_words"]
        print(f"Loaded random task:")
    except:
        print(f"Failed loading tasks")
    
    print(f"  Mandatory: {m_words}")
    print(f"  Optional: {o_words[:80]}...\n")
    
    prompts, messages, responses = generate_outline_with_criteria(m_words, o_words)
    
    print("\n=== Generated Outline (First 500 chars) ===\n")
    print(responses[0][:500] + "...\n")
    
    print("✓ Test completed successfully!")

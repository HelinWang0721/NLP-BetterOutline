from openai import OpenAI
import yaml
from typing import Dict, Optional, List, Tuple

# Load configuration
with open("./config.yaml", 'r') as f:
    config = yaml.safe_load(f)
    api_key = config['api']['api_key']
    MODEL_CONFIGS = config['evaluation']['voting_models']

# Load voting system instruction
with open("./vote-sys-inst.md", 'r', encoding='utf-8') as f:
    VOTE_SYS_INST = f.read()

# Global client
client = None


def chat(**kwargs) -> str:
    """Chat with OpenRouter API"""
    kwargs["stream"] = False
    return client.chat.completions.create(**kwargs).choices[0].message.content


def parse_score(response) -> Optional[List[float]]:
    """Parse evaluation scores from model response"""
    details = response.split("## ")
    details = [detail.strip() for detail in details if detail.strip()]

    if len(details) != 15:
        print(f"Expected 15 details, got {len(details)}")
        return None

    titles = [
        "1. 合理性", "2. 新颖程度", "3. 悬念", "4. 反转和惊喜", "5. 期待感",
        "6. 目标", "7. 读者偏好", "8. 设定复杂性", "9. 情节复杂性", "10. 代入感",
        "11. 情感波动", "12. 一致性", "13. 相关度", "14. 结局", "15. 情节分配"
    ]

    # Check title order
    for i in range(len(titles)):
        if titles[i] not in details[i]:
            print(f"Title {titles[i]} not found in details[{i}]")
            return None

    # Parse scores
    ret = []
    for i in range(len(details)):
        lines = details[i].split("\n")
        lines = [line.strip() for line in lines]
        for line in lines:
            if line.startswith("score:"):
                score = line.split(":")[-1].strip()
                try:
                    ret.append(float(score))
                except ValueError:
                    print(f"Failed to parse score: {score} for title: {titles[i]}")
                    return None

    return ret


def vote_one_model(outline: str, model_config: dict) -> Dict[str, Tuple[List[float], str]]:
    """
    Evaluate outline with one model
    
    Args:
        outline: Story outline text
        model_config: Model configuration dict
        
    Returns:
        Dict mapping model name to (scores, response)
    """
    global client
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    
    MAX_RETRY = 3
    messages = [
        {"role": "system", "content": VOTE_SYS_INST},
        {"role": "user", "content": outline}
    ]
    model_config["messages"] = messages

    try_cnt = 0
    scores = None
    while try_cnt < MAX_RETRY:
        response = chat(**model_config)
        scores = parse_score(response)
        if scores is not None:
            break
        else:
            print(f"Failed to parse scores for model: {model_config['model']}")
            print(response)
        try_cnt += 1

    if scores is None:
        raise Exception("Failed to parse scores")

    return {model_config["model"]: (scores, response)}


def vote_all(outline: str) -> Dict[str, Tuple[List[float], str]]:
    """
    Evaluate outline with all configured models
    
    Args:
        outline: Story outline text
        
    Returns:
        Dict mapping model names to (scores, response)
    """
    ret = {}
    for model in MODEL_CONFIGS:
        try:
            scores = vote_one_model(outline, model)
            ret.update(scores)
        except Exception as e:
            print(f"Failed to vote for model: {model['model']}")
            print(e)

    return ret


def process_file(input_file: str, output_file: str):
    """Process an outline file and save evaluations"""
    import json
    import os
    
    print(f"Processing {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return

    results = []
    total = len(data)
    
    for i, item in enumerate(data):
        print(f"Evaluating outline {i+1}/{total}...")
        
        # Get the outline text (usually the last response)
        responses = item.get("responses", [])
        if not responses:
            print(f"Skipping item {i}: No responses found")
            continue
            
        outline_text = responses[-1]
        
        # Evaluate
        eval_scores = vote_all(outline_text)
        
        # Create result entry
        result_entry = item.copy()
        result_entry["evaluation"] = eval_scores
        results.append(result_entry)
        
        # Save incrementally
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
    print(f"Done! Saved evaluations to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate outlines using LLM voting")
    parser.add_argument("input_file", help="Input JSON file containing outlines")
    parser.add_argument("--output", help="Output JSON file for evaluations (default: input_eval.json)")
    
    args = parser.parse_args()
    
    output_file = args.output
    if not output_file:
        base, ext = os.path.splitext(args.input_file)
        output_file = f"{base}_eval{ext}"
        
    process_file(args.input_file, output_file)

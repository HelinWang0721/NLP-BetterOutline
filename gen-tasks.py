"""
Batch Outline Generation for Baseline vs Criteria Comparison
Generates outlines using both baseline (minimal) and criteria-integrated prompts
"""

import json
import random
import baseline_gen
import outline_gen
import yaml

# Load configuration
with open("./config.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


def load_tasks(task_file: str = "./eval_tasks.json"):
    """Load evaluation tasks"""
    with open(task_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_results(data: list, filename: str):
    """Save results to JSON"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_comparison_batch(
    num_samples: int = 10,
    baseline_output: str = "./baseline_outlines.json",
    improved_output: str = "./improved_outlines.json"
):
    """
    Generate outlines for comparison
    
    Args:
        num_samples: Number of outlines to generate
        baseline_output: Output file for baseline outlines
        improved_output: Output file for improved outlines
    """
    
    # Load tasks
    try:
        tasks = load_tasks("./tasks.json")
        print(f"Loaded {len(tasks)} tasks from tasks.json")
    except:
        try:
            tasks = load_tasks("./eval_tasks.json")
            print(f"Loaded {len(tasks)} tasks from eval_tasks.json")
        except Exception as e:
            print(f"Error: Could not load tasks")
            print(f"Error message: {e}")
            return
    
    # Randomly shuffle tasks
    random.shuffle(tasks)
    print(f"Randomly shuffled tasks\n")
    
    baseline_results = []
    improved_results = []
    
    for i, task in enumerate(tasks[:num_samples]):
        print(f"\n{'='*60}")
        print(f"Generating outline {i+1}/{num_samples}")
        print(f"{'='*60}")
        
        m_words = task["m_words"]
        o_words = task["o_words"]
        
        print(f"\nKeywords:")
        print(f"  Mandatory: {m_words}")
        print(f"  Optional: {o_words[:50]}...")
        
        try:
            # Generate baseline (minimal prompting)
            print("\n[1/2] Generating BASELINE outline...")
            baseline_prompts, baseline_msgs, baseline_resps = baseline_gen.generate_baseline_outline(
                m_words, o_words
            )
            
            baseline_entry = {
                "index": i,
                "m_words": m_words,
                "o_words": o_words,
                "prompts": baseline_prompts,
                "responses": baseline_resps
            }
            baseline_results.append(baseline_entry)
            
            # Generate improved (criteria hints)
            print("\n[2/2] Generating IMPROVED outline (with criteria hints)...")
            improved_prompts, improved_msgs, improved_resps = outline_gen.generate_outline_with_criteria(
                m_words, o_words
            )
            
            improved_entry = {
                "index": i,
                "m_words": m_words,
                "o_words": o_words,
                "prompts": improved_prompts,
                "responses": improved_resps
            }
            improved_results.append(improved_entry)
            
            # Save incrementally
            save_results(baseline_results, baseline_output)
            save_results(improved_results, improved_output)
            
            print(f"\n✓ Outline {i+1} completed and saved")
            
        except Exception as e:
            print(f"\n✗ Error generating outline {i+1}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Baseline outlines: {baseline_output} ({len(baseline_results)} samples)")
    print(f"Improved outlines: {improved_output} ({len(improved_results)} samples)")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comparison batch")
    parser.add_argument("--num", type=int, default=10, help="Number of outlines to generate")
    parser.add_argument("--baseline", default="./baseline_outlines.json", help="Baseline output file")
    parser.add_argument("--improved", default="./improved_outlines.json", help="Improved output file")
    
    args = parser.parse_args()
    
    generate_comparison_batch(args.num, args.baseline, args.improved)
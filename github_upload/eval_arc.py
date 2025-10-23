import json
import os
import time
import argparse

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def format_example(item, include_answer=True):
    """Format a single example like in MMLU."""
    question = item["question"]["stem"]
    choices = item["question"]["choices"]
    
    prompt = question + "\n"
    for choice in choices:
        prompt += f"{choice['label']}. {choice['text']}\n"
    
    prompt += "Answer:"
    if include_answer:
        prompt += f" {item['answerKey']}\n\n"
    
    return prompt

def gen_prompt(train_items, test_item, k=-1):
    """Generate prompt with few-shot examples."""
    prompt = "The following are multiple choice questions (with answers).\n\n"
    
    # Add few-shot examples if k > 0
    if k > 0:
        for i in range(min(k, len(train_items))):
            prompt += format_example(train_items[i], include_answer=True)
    
    # Add the test question without answer
    prompt += format_example(test_item, include_answer=False)
    
    return prompt

def load_dataset(file_path):
    """Load dataset from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

@torch.no_grad()
def evaluate(model, tokenizer, test_items, train_items=None, ntrain=0):
    """Evaluate model on test items."""
    if train_items is None:
        train_items = []
    
    cors = []
    all_probs = []
    results = []
    
    for i in tqdm(range(len(test_items))):
        item = test_items[i]
        
        # Get prompt and make sure it fits
        k = ntrain
        prompt = gen_prompt(train_items, item, k)
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        
        # Ensure input fits within model's context window
        while input_ids.shape[-1] > 2048 and k > 0:
            k -= 1
            prompt = gen_prompt(train_items, item, k)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        
        # Get the correct answer
        label = item["answerKey"]
        
        # Get logits for the last token
        logits = model(input_ids=input_ids).logits[0, -1]
        
        # Get probabilities for each option (A, B, C, D)
        option_tokens = []
        for choice in ["A", "B", "C", "D"]:
            option_tokens.append(tokenizer(choice).input_ids[-1])
        
        # Handle case where some options might not be available
        available_options = ["A", "B", "C", "D"][:len(item["question"]["choices"])]
        
        # Calculate probabilities for available options
        probs = torch.nn.functional.softmax(
            torch.tensor([logits[tokenizer(opt).input_ids[-1]] for opt in available_options]).float(),
            dim=0
        ).detach().cpu().numpy()
        
        # Get prediction
        pred = available_options[np.argmax(probs)]
        
        # Check if correct
        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)
        
        # Save result for this item
        results.append({
            "id": item["id"],
            "question": item["question"]["stem"],
            "correct_answer": label,
            "predicted_answer": pred,
            "is_correct": bool(cor),
            "probabilities": {opt: float(probs[i]) for i, opt in enumerate(available_options)}
        })
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(test_items)} questions. Current accuracy: {np.mean(cors):.3f}")
    
    # Calculate accuracy
    acc = np.mean(cors)
    print(f"Final accuracy: {acc:.3f}")
    
    return cors, acc, all_probs, results

def main():
    parser = argparse.ArgumentParser(description='Evaluate models on ARC datasets')
    parser.add_argument('--model_name_or_path', type=str, required=True, 
                        help='Path to model checkpoint')
    parser.add_argument('--benchmark', type=str, required=True, 
                        choices=['arc_easy', 'arc_challenge'],
                        help='Which ARC benchmark to evaluate')
    parser.add_argument('--model_family', type=str, required=True,
                        choices=['llama2', 'qwen2_5'],
                        help='Model family for proper handling')
    parser.add_argument('--torch_dtype', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Torch dtype for model')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save results (default: auto-generated)')
    parser.add_argument('--ntrain', type=int, default=0,
                        help='Number of few-shot examples (0 for zero-shot)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set dataset paths based on benchmark
    if args.benchmark == 'arc_easy':
        dataset_path = "utility_data/ARC-Easy/ARC-Easy-Test.jsonl"
        dev_dataset_path = "utility_data/ARC-Easy/ARC-Easy-Dev.jsonl"
        default_result_file = "arc_easy_results.json"
    else:  # arc_challenge
        dataset_path = "utility_data/ARC-Challenge/ARC-Challenge-Test.jsonl"
        dev_dataset_path = "utility_data/ARC-Challenge/ARC-Challenge-Dev.jsonl"
        default_result_file = "arc_challenge_results.json"
    
    result_file = args.save_path if args.save_path else default_result_file
    
    # Set torch dtype
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32
    }
    torch_dtype = dtype_map[args.torch_dtype]
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model.eval()
    
    # Load test data
    print(f"Loading test dataset: {dataset_path}")
    test_items = load_dataset(dataset_path)
    print(f"Loaded {len(test_items)} test questions")
    
    # Load dev data for few-shot if needed
    train_items = []
    if args.ntrain > 0:
        print(f"Loading dev dataset for {args.ntrain}-shot evaluation: {dev_dataset_path}")
        train_items = load_dataset(dev_dataset_path)
        print(f"Loaded {len(train_items)} dev questions")
    
    # Start evaluation
    print("Starting evaluation...")
    start_time = time.time()
    cors, acc, probs, detailed_results = evaluate(model, tokenizer, test_items, train_items, args.ntrain)
    end_time = time.time()
    
    # Save results
    results = {
        "model": args.model_name_or_path,
        "model_family": args.model_family,
        "benchmark": args.benchmark,
        "dataset": dataset_path,
        "ntrain": args.ntrain,
        "accuracy": float(acc),
        "time_taken": end_time - start_time,
        "num_correct": int(sum(cors)),
        "num_total": len(cors),
        "detailed_results": detailed_results
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(result_file) or '.', exist_ok=True)
    
    # Save results to file
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation complete! Accuracy: {acc:.3f}")
    print(f"Results saved to: {result_file}")

if __name__ == "__main__":
    main()


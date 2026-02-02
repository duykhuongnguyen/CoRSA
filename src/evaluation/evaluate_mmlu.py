import argparse
from datasets import load_dataset, get_dataset_config_names
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm
import argparse


# Standard mapping for MMLU choices
CHOICES = ["A", "B", "C", "D"]


def format_example(example, include_answer=True):
    """Formats a single Q&A example into text."""
    prompt = f"Question: {example['question']}\n"
    for i, choice in enumerate(example['choices']):
        prompt += f"{CHOICES[i]}. {choice}\n"
    
    prompt += "Answer:"
    if include_answer:
        prompt += f" {CHOICES[example['answer']]}\n\n"
    return prompt


def generate_prompt(dev_set, test_example, n_shot=5):
    """Creates a 5-shot prompt using examples from the dev set."""
    prompt = "The following are multiple choice questions (with answers).\n\n"
    for i in range(min(n_shot, len(dev_set))):
        prompt += format_example(dev_set[i], include_answer=True)
    prompt += format_example(test_example, include_answer=False)
    return prompt


def main(model_name, lora_path=None, n_shot=5):
    print(f"Fetching MMLU subject list...")
    # Dynamically get all subject names (configs) from the Hugging Face hub
    subjects = get_dataset_config_names("hails/mmlu_no_train")
    
    # Filter out 'all' if present to avoid duplication, though usually configs are just subjects
    subjects = [s for s in subjects if s != 'all']
    
    print(f"Found {len(subjects)} subjects. Initializing vLLM model: {model_name}...")
    
    # Initialize vLLM once (loading the model is the heaviest step)
    llm = LLM(model=model_name, max_model_len=8192, enable_lora=bool(lora_path), max_lora_rank=32, trust_remote_code=True)
    lora_request = LoRARequest("adapter", 1, lora_path) if lora_path else None
    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    global_correct = 0
    global_total = 0

    print("Starting evaluation across all subjects...")
    
    # Iterate through every subject
    for subject in tqdm(subjects, desc="Evaluating Subjects"):
        # Load specific subject data
        dataset = load_dataset("hails/mmlu_no_train", subject, split="test")
        dev_dataset = load_dataset("hails/mmlu_no_train", subject, split="dev")

        # Prepare prompts
        prompts = [generate_prompt(dev_dataset, example, n_shot) for example in dataset]
        ground_truths = [CHOICES[example['answer']] for example in dataset]

        # Batch generation for this subject
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request, use_tqdm=False)

        # check answers
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip().upper()
            if generated_text.startswith(ground_truths[i]):
                global_correct += 1
            global_total += 1

    # Final Calculation
    if global_total > 0:
        final_accuracy = global_correct / global_total
        print("\n" + "="*40)
        print(f"Model: {model_name}")
        print(f"Total Questions Evaluated: {global_total}")
        print(f"Global MMLU Accuracy (5-shot): {final_accuracy:.2%}")
        print("="*40)
    else:
        print("Error: No data was evaluated.")


if __name__ == "__main__":
    # Replace with your local model path or HF model ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Base model name/path for vLLM")
    ap.add_argument("--lora_path", type=str, default="", help="Path to LoRA adapter directory/file")
    args = ap.parse_args()
    
    main(args.model, lora_path=args.lora_path, n_shot=5)

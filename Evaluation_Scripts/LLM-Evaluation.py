# %%
# Install required packages
!pip install -q transformers accelerate evaluate bert-score rouge-score nltk detoxify torch datasets
!pip install -q sentencepiece protobuf

# %%
# Download NLTK data for METEOR
import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# 2. Imports and Configuration

import json
import torch
import gc
import warnings
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import numpy as np
from collections import Counter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from evaluate import load
from bert_score import score as bert_score_func
from detoxify import Detoxify
import re
import os

warnings.filterwarnings('ignore')

# %%
# Clear GPU cache
torch.cuda.empty_cache()
gc.collect()

# %%
# 3. Configuration

# CHANGE THIS TO SWITCH MODELS
# MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_NAME = "ShivomH/Elixir-MentalHealth-3B"
# MODEL_NAME = "google/gemma-2-2b-it"
# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# %%
# Configuration parameters
CONFIG = {
    "model_name": MODEL_NAME,
    "dataset_path": "/content/MH_final_test.jsonl",
    "max_new_tokens": 512,
    "temperature": 0.5,
    "top_p": 0.85,
    "batch_size": 1,  # Keep at 1 for memory efficiency
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_half_precision": True,  # Use float16 for memory efficiency
    "seed": 42
}

# Set seed for reproducibility
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

print(f"Configuration:")
print(f"  Model: {CONFIG['model_name']}")
print(f"  Device: {CONFIG['device']}")
print(f"  Dataset: {CONFIG['dataset_path']}")

# %%
# 4. Helper Functions

def load_dataset(path: str) -> List[Dict]:
    """Load the test dataset from JSON or JSONL file."""
    try:
        data = []

        # Check file extension to determine format
        if path.endswith('.jsonl'):
            # Load JSONL format (one JSON object per line)
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        data.append(json.loads(line))
            print(f"✓ Loaded {len(data)} examples from {path} (JSONL format)")
        else:
            # Load standard JSON format
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ Loaded {len(data)} examples from {path} (JSON format)")

        # Verify dataset structure
        multi_turn = sum(1 for item in data if len(item['messages']) > 2)
        single_turn = len(data) - multi_turn
        print(f"  - Multi-turn conversations: {multi_turn}")
        print(f"  - Single-turn conversations: {single_turn}")

        return data
    except FileNotFoundError:
        print(f"❌ Dataset not found at {path}")
        print("Please upload your test_dataset.json or test_dataset.jsonl file to /content/")
        raise
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        print("Please check your file format (JSON or JSONL)")
        raise
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise

def prepare_input(messages: List[Dict], tokenizer) -> str:
    """Prepare input for the model using chat template."""
    # Remove the last assistant message (the one we want to predict)
    input_messages = messages[:-1]

    # Apply chat template
    try:
        input_text = tokenizer.apply_chat_template(
            input_messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except:
        # Fallback for models without chat template
        input_text = ""
        for msg in input_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                input_text += f"User: {content}\n"
            elif role == "assistant":
                input_text += f"Assistant: {content}\n"
        input_text += "Assistant: "

    return input_text

def calculate_distinct_n(texts: List[str], n: int = 1) -> float:
    """Calculate Distinct-n metric for diversity."""
    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        all_ngrams.extend(ngrams)

    if len(all_ngrams) == 0:
        return 0.0

    return len(set(all_ngrams)) / len(all_ngrams)

def calculate_self_bleu(texts: List[str], n: int = 3) -> float:
    """Calculate Self-BLEU for measuring repetitiveness."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    if len(texts) < 2:
        return 0.0

    smoothing = SmoothingFunction().method1
    scores = []

    for i, text in enumerate(texts[:100]):  # Limit to 100 for efficiency
        references = [t.split() for j, t in enumerate(texts[:100]) if i != j]
        hypothesis = text.split()
        if hypothesis and references:
            score = sentence_bleu(references, hypothesis,
                                 weights=(1/n,)*n,
                                 smoothing_function=smoothing)
            scores.append(score)

    return np.mean(scores) if scores else 0.0

def calculate_perplexity(model, tokenizer, texts: List[str], device: str) -> float:
    """Calculate perplexity as a measure of fluency."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in tqdm(texts[:50], desc="Calculating perplexity", leave=False):  # Sample for efficiency
            inputs = tokenizer(text, return_tensors="pt",
                              truncation=True, max_length=512).to(device)

            if inputs.input_ids.shape[1] <= 1:
                continue

            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss

            if loss is not None:
                total_loss += loss.item() * inputs.input_ids.shape[1]
                total_tokens += inputs.input_ids.shape[1]

    if total_tokens == 0:
        return float('inf')

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity if perplexity < 1000 else 1000.0


# %%
# 5. Load Model and Tokenizer

def load_model_and_tokenizer(model_name: str, use_half: bool = True):
    """Load model and tokenizer with memory-efficient settings."""
    print(f"\n📥 Loading model: {model_name}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with half precision for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_half else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        model.eval()
        print(f"✓ Model loaded successfully!")

        # Print memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU Memory allocated: {allocated:.2f} GB")

        return model, tokenizer

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

# Load the model
model, tokenizer = load_model_and_tokenizer(CONFIG["model_name"], CONFIG["use_half_precision"])

# %%
# 6. Generate Predictions

def generate_predictions(model, tokenizer, dataset: List[Dict], config: Dict) -> Tuple[List[str], List[str]]:
    """Generate predictions for the dataset."""
    predictions = []
    references = []

    print(f"\n🔮 Generating predictions...")

    for item in tqdm(dataset, desc="Generating"):
        messages = item["messages"]

        # Get reference (last assistant message)
        reference = messages[-1]["content"]
        references.append(reference)

        # Prepare input
        input_text = prepare_input(messages, tokenizer)

        # Tokenize
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(config["device"])

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode prediction
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        predictions.append(prediction)

        # Clear cache periodically
        if len(predictions) % 50 == 0:
            torch.cuda.empty_cache()

    print(f"✓ Generated {len(predictions)} predictions")

    return predictions, references

# Generate predictions
predictions, references = generate_predictions(model, tokenizer,
                                              load_dataset(CONFIG["dataset_path"]),
                                              CONFIG)

# %%
# 7. Evaluate Metrics

def evaluate_all_metrics(predictions: List[str], references: List[str], model, tokenizer, device: str) -> Dict:
    """Calculate all evaluation metrics."""
    print("\n📊 Computing evaluation metrics...")
    results = {}

    # 1. BERTScore
    print("  • Computing BERTScore...")
    try:
        P, R, F1 = bert_score_func(predictions, references,
                                   lang="en", verbose=False,
                                   device=device, batch_size=8)
        results["bertscore"] = {
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F1.mean())
        }
    except Exception as e:
        print(f"    Warning: BERTScore failed - {e}")
        results["bertscore"] = {"precision": 0, "recall": 0, "f1": 0}

    # 2. ROUGE-L
    print("  • Computing ROUGE-L...")
    try:
        rouge = load("rouge")
        rouge_results = rouge.compute(predictions=predictions, references=references)
        results["rouge_l"] = float(rouge_results["rougeL"])
    except Exception as e:
        print(f"    Warning: ROUGE-L failed - {e}")
        results["rouge_l"] = 0.0

    # 3. METEOR
    print("  • Computing METEOR...")
    try:
        meteor = load("meteor")
        meteor_results = meteor.compute(predictions=predictions, references=references)
        results["meteor"] = float(meteor_results["meteor"])
    except Exception as e:
        print(f"    Warning: METEOR failed - {e}")
        results["meteor"] = 0.0

    # 4. Distinct-1 and Distinct-2
    print("  • Computing Distinct-n...")
    results["distinct_1"] = calculate_distinct_n(predictions, n=1)
    results["distinct_2"] = calculate_distinct_n(predictions, n=2)

    # 5. Self-BLEU
    print("  • Computing Self-BLEU...")
    results["self_bleu"] = calculate_self_bleu(predictions, n=3)

    # 6. Toxicity
    print("  • Computing Toxicity scores...")
    try:
        toxicity_model = Detoxify('original')
        toxicity_scores = []
        for pred in tqdm(predictions, desc="    Analyzing toxicity", leave=False):
            scores = toxicity_model.predict(pred)
            toxicity_scores.append(scores['toxicity'])
        results["toxicity"] = {
            "mean": float(np.mean(toxicity_scores)),
            "max": float(np.max(toxicity_scores)),
            "min": float(np.min(toxicity_scores))
        }
    except Exception as e:
        print(f"    Warning: Toxicity detection failed - {e}")
        results["toxicity"] = {"mean": 0, "max": 0, "min": 0}

    # 7. Perplexity
    print("  • Computing Perplexity...")
    try:
        perplexity = calculate_perplexity(model, tokenizer, predictions, device)
        results["perplexity"] = float(perplexity)
    except Exception as e:
        print(f"    Warning: Perplexity calculation failed - {e}")
        results["perplexity"] = 0.0

    # Add metadata
    results["metadata"] = {
        "num_samples": len(predictions),
        "avg_prediction_length": np.mean([len(p.split()) for p in predictions]),
        "avg_reference_length": np.mean([len(r.split()) for r in references])
    }

    print("✓ All metrics computed successfully!")

    return results

# %%
# Evaluate
results = evaluate_all_metrics(predictions, references, model, tokenizer, CONFIG["device"])

# %%
# 8. Display Results and Sample Outputs

# Display sample predictions
print("\n" + "="*80)
print("📝 SAMPLE PREDICTIONS (First 3)")
print("="*80)

for i in range(min(3, len(predictions))):
    print(f"\n--- Example {i+1} ---")
    print(f"Reference: {references[i][:200]}...")
    print(f"Prediction: {predictions[i][:200]}...")

# Display final results
print("\n" + "="*80)
print("📊 EVALUATION RESULTS")
print("="*80)

# Pretty print results
import json
print(json.dumps(results, indent=2))

# %%
# 9. Save Results

# Create output filename based on model name
model_name_clean = CONFIG["model_name"].replace("/", "-").replace(".", "_")
output_filename = f"{model_name_clean}_eval_results.json"

# Prepare final results dictionary
final_results = {
    "model": CONFIG["model_name"],
    "configuration": {
        "max_new_tokens": CONFIG["max_new_tokens"],
        "temperature": CONFIG["temperature"],
        "top_p": CONFIG["top_p"],
        "device": CONFIG["device"]
    },
    "metrics": results,
    "sample_outputs": [
        {
            "reference": ref[:500],
            "prediction": pred[:500]
        }
        for ref, pred in zip(references[:5], predictions[:5])
    ]
}

# Save to JSON file
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Results saved to: {output_filename}")

# %%
# Clear GPU memory
del model
del tokenizer
torch.cuda.empty_cache()
gc.collect()

print("\n🧹 Cleaned up GPU memory")
print("✨ Evaluation complete!")
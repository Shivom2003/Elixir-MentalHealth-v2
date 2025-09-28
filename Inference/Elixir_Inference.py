# %%
#Install dependencies
#!pip install transformer torch accelerate bitsandbytes

# %%
#Import dependencies
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# %%
# Hugging Face login (required for Llama models)
from huggingface_hub import login

# Option 1: Use token from environment
os.environ["HF_TOKEN"] = "your_token_id"
login(token=os.environ["HF_TOKEN"])

# Option 2: Interactive login
# login()

# %%
# Name/Paths of the model/tokenizer
model_path = "ShivomH/Elixir-MentalHealth-3B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model (quantization if needed)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# %%
def generate_response(
    model,
    tokenizer,
    messages,
    max_new_tokens=256,
    # temperature=0.5,
    # top_p=0.8,
    do_sample=True
):
    """
    Generate response from the model given messages.

    Args:
        messages: List of message dicts with 'role' and 'content'

    Returns:
        Generated response string
    """
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # temperature=temperature,
            do_sample=do_sample,
            # top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant response
    response = generated.split("assistant\n")[-1].strip()

    return response

print("Inference function ready!")

# %%
# Interactive chat interface
print("="*60)
print("💬 Mental Health Support Chat")
print("Type 'quit' to exit, 'reset' to start a new conversation")
print("="*60)

# System prompt
system_message = {
    "role": "system",
    "content": """You a supportive and empathetic AI assistant trained to provide mental health and medical guidance.
    Your responses should be warm, professional, and safe.
    - Prioritize active listening, validation, and evidence-based strategies.
    - Do not attempt to diagnose or prescribe medications.
    - Always remind users that they should consult a licensed professional for medical or crisis support.
    - If the user expresses self-harm or crisis thoughts, encourage them to reach out to a helpline or emergency services immediately."""
}

# Conversation history
conversation = [system_message]

while True:
    # Get user input
    user_input = input("\nYou: ").strip()

    if user_input.lower() == 'quit':
        print("Goodbye! Take care.")
        break

    if user_input.lower() == 'reset':
        conversation = [system_message]
        print("Conversation reset.")
        continue

    if not user_input:
        continue

    # Add user message
    conversation.append({"role": "user", "content": user_input})

    # Generate assistant reply
    reply = generate_response(model, tokenizer, conversation)

    # Add assistant message to history
    conversation.append({"role": "assistant", "content": reply})

    # Print assistant response
    print(f"\nAssistant: {reply}")
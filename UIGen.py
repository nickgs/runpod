import runpod
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Tesslate/UIGEN-T1.5-7B")
    model = AutoModelForCausalLM.from_pretrained(
        "Tesslate/UIGEN-T1.5-7B",
        device_map="auto",         # Use this if you're on a GPU-enabled Runpod runtime
        torch_dtype="auto"         # Helps with large models (mixed precision)
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

def text_generation_handler(event):
    global model

    if "model" not in globals():
        model = load_model()

    prompt = event["input"].get("prompt")
    if not prompt:
        return {"error": "No prompt provided."}

    output = model(prompt)
    return {"generated_text": output[0]["generated_text"]}

runpod.serverless.start({"handler": text_generation_handler})


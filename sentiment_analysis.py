import runpod
from transformers import pipeline

def load_model():
    return pipeline(
        "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
    )


def sentiment_analysis_handler(event):
    global model

    # Ensure the model is loaded
    if "model" not in globals():
        model = load_model()

    # Get the input text from the event
    text = event["input"].get("text")

    # Validate input
    if not text:
        return {"error": "No text provided for analysis."}

    # Perform sentiment analysis
    result = model(text)[0]

    return {"sentiment": result["label"], "score": float(result["score"])}

runpod.serverless.start({"handler": sentiment_analysis_handler})

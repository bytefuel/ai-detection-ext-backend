from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Initialize the Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load the BART model for zero-shot classification (already done)
bart_classifier = pipeline("zero-shot-classification",
                           model="facebook/bart-large-mnli")

# Load DistilBERT model for AI detection (second model)
distilbert_model_name = 'distilbert-base-uncased'
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(
    distilbert_model_name)
distilbert_model = DistilBertForSequenceClassification.from_pretrained(
    distilbert_model_name)

# Define a route for AI detection

@app.route('/detect-ai', methods=['POST'])
def detect_ai():
    # Extract the text from the request body
    data = request.get_json()
    text = data.get('text')

    # Check if text is provided
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Define candidate labels for classification
    candidate_labels = ['AI', 'Human']

    try:
        # Run BART model for zero-shot classification
        bart_result = bart_classifier(text, candidate_labels=candidate_labels)
        bart_ai_score = bart_result['scores'][bart_result['labels'].index(
            'AI')]
        bart_human_score = bart_result['scores'][bart_result['labels'].index(
            'Human')]

        # Run DistilBERT model for AI detection (second model)
        inputs = distilbert_tokenizer(
            text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
            distilbert_scores = outputs.logits.softmax(dim=-1)
            # Assuming index 1 is AI
            distilbert_ai_score = distilbert_scores[0][1].item()

        # Combine the results (simple average or weighted average)
        ai_score = (bart_ai_score + distilbert_ai_score) / 2
        human_score = 1 - ai_score  

        return jsonify({
            "ai_score": ai_score,
            "human_score": human_score,
            "result": "AI" if ai_score > human_score else "Human"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(port=3000)

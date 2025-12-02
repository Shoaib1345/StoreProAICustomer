from flask import Flask, request, jsonify, render_template
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# -------------------------
# Load Fine-Tuned Model
# -------------------------
model_path = "fine_tuned_sbert"
print("🔥 Loading fine-tuned SBERT model...")
model = SentenceTransformer(model_path)
print("✅ Model loaded!")

# -------------------------
# Load Dataset + Prepare Embeddings
# -------------------------
dataset_file = "dataset.json"
print("🔍 Loading dataset...")
data = json.load(open(dataset_file, "r", encoding="utf-8"))

questions = [item["input"] for item in data]
answers = [item["output"] for item in data]

print(f"📄 Total dataset pairs: {len(questions)}")
print("🔄 Generating embeddings using fine-tuned model...")
embeddings = model.encode(questions, convert_to_numpy=True)
print("✅ Embeddings ready!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/gui")
def gui():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "")

    if not user_msg.strip():
        return jsonify({"answer": "Please type a message."})

    # Encode user message
    user_emb = model.encode([user_msg], convert_to_numpy=True)

    # Similarity with dataset embeddings
    sims = cosine_similarity(user_emb, embeddings)[0]

    # Find best match
    best_idx = np.argmax(sims)
    response = answers[best_idx]

    return jsonify({
        "answer": response,
        "matched_question": questions[best_idx],
        "similarity_score": float(sims[best_idx])
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": "loaded"})


if __name__ == "__main__":
    app.run(debug=True)
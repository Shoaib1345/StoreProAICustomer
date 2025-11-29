from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "your-secret-key"

class SmartSBERTChatbot:
    def __init__(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

        # Load questions, answers, embeddings
        with open("ai_models/questions.pkl", "rb") as f:
            self.questions = pickle.load(f)

        with open("ai_models/answers.pkl", "rb") as f:
            self.answers = pickle.load(f)

        self.embeddings = np.load("ai_models/embeddings.npy")

        print(f"✅ SBERT Chatbot loaded. Total Q/A pairs: {len(self.questions)}")

    def get_response(self, user_question):
        try:
            user_emb = self.model.encode([user_question], convert_to_numpy=True)
            sims = cosine_similarity(user_emb, self.embeddings)[0]

            best_idx = np.argmax(sims)
            best_score = sims[best_idx]

            if best_score >= 0.65:
                return self.answers[best_idx]
            else:
                return "🤔 Sorry, I didn't understand that. Please ask something else."
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return "😔 Sorry, I encountered an error processing your question."

try:
    chatbot = SmartSBERTChatbot()
except Exception as e:
    print(f"Failed to initialize chatbot: {str(e)}")
    chatbot = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    if chatbot is None:
        return jsonify({"answer": "Chatbot is not initialized properly"}), 500
        
    data = request.get_json()
    user_question = data.get("question", "")
    answer = chatbot.get_response(user_question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
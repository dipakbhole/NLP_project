from flask import Flask, request, jsonify, render_template
import pickle
import nltk
import numpy as np
from nltk.corpus import stopwords

app = Flask(__name__)

stop_words = set(stopwords.words("english"))

#Load the trained model
with open(r"artifacts/semantic_similarity_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open(r"artifacts/glove_embeddings.pkl", "rb") as embeddings_file:
    glove_embeddings = pickle.load(embeddings_file)


#Text Preprocessing and Vectorization functions
def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(words)

def sentence_to_vector(sentence, embeddings):
    vectors = [embeddings[word] for word in sentence.split() if word in embeddings]
    return np.mean(vectors, axis=0) if vectors else np.zeros_like(embeddings["a"])

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norm_product if norm_product != 0 else 0.0
   
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_similarity', methods=['POST','GET'])
def predict_similarity():
    if request.method == 'POST':
        try:
            data = request.get_json()
            text1 = preprocess_text(data["text1"])
            text2 = preprocess_text(data["text2"])

            vector1 = sentence_to_vector(text1, glove_embeddings)
            vector2 = sentence_to_vector(text2, glove_embeddings)

            similarity_score = model.predict([[cosine_similarity(vector1, vector2)]])[0]

            # Convert the float32 to a standard Python float
            similarity_score = float(similarity_score)

            response = {"similarity_score": similarity_score}
            return jsonify(response)

        except Exception as e:
            return jsonify({"error": str(e)})
    
    else:
        return jsonify({"error": "Method not allowed. Use POST request."})
    
if __name__ == "__main__":
    app.run(debug= False, host="0.0.0.0", port= 8000)
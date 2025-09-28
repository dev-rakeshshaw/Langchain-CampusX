# hf_inference_similarity_docs.py
# Uses Hugging Face Inference API (InferenceClient) to compute similarity between
# one query and many documents. No local model download.

import os, sys
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import numpy as np

load_dotenv()  # loads HF_TOKEN from .env if present

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    print("ERROR: set HF_TOKEN in .env or environment (HF_TOKEN or HUGGINGFACEHUB_API_TOKEN).")
    sys.exit(1)

# create client (uses hf-inference provider by default)
client = InferenceClient(api_key=HF_TOKEN)

# Your query and documents
query = "tell me about bumrah"
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# Model to use on HF Inference — a common, small sentence embedding model
model_id = "sentence-transformers/all-MiniLM-L6-v2"

try:
    # Single API call: anchor + list of docs -> returns similarity scores (one per document)
    scores = client.sentence_similarity(query, other_sentences=documents, model=model_id)
except Exception as e:
    print("Inference API request failed:", e)
    sys.exit(1)

# scores is normally a list of floats (one per document)
# Convert to numpy for easy operations and sorting
scores_arr = np.array([float(s) for s in scores])

# Print all scores, sorted (descending)
print("Query:", query, "\n")
print("Document similarities (highest → lowest):")
sorted_idx = np.argsort(-scores_arr)  # negative for descending
for i in sorted_idx:
    print(f"  score={scores_arr[i]:.4f}  ->  {documents[i]}")

# Best match
best_idx = int(np.argmax(scores_arr))
print("\nBest matching document:")
print(documents[best_idx])
print("Similarity score:", f"{scores_arr[best_idx]:.4f}")

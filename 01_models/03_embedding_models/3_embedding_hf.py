from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

#Local Enbedding
# embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# documents = [
#     "Delhi is the capital of India",
#     "Kolkata is the capital of West Bengal",
#     "Paris is the capital of France"
# ]

# vector = embedding.embed_documents(documents)

# print(str(vector))


#Using inference API
import os
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

result = client.sentence_similarity("That is a happy person",
    other_sentences=[
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day"
    ],
    model="sentence-transformers/all-MiniLM-L6-v2",
)

print(result)
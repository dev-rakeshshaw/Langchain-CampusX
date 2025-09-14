from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")  # <-- Pass token here
# )

# model = ChatHuggingFace(llm=llm)

# result = model.invoke("What is the capital of India")

# print(result.content)



import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
)

completion = client.chat.completions.create(
    model="baidu/ERNIE-4.5-21B-A3B-Thinking",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)

print(completion.choices[0].message)
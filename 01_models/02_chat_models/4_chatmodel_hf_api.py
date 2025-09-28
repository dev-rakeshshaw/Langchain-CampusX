import os
from openai import OpenAI
from dotenv import load_dotenv  

# Load environment variables from the .env file
# (This allows you to keep secrets like API keys outside of the code)
load_dotenv()

# Create an OpenAI client but pointing to Hugging Face's router endpoint
# - base_url: tells the client to use Hugging Face's API instead of OpenAI
# - api_key: retrieves Hugging Face token from environment variable "HF_TOKEN"
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

# Create a chat completion request (sends prompt to the model)
# - model: specifies the Hugging Face model you want to query
# - messages: chat history in OpenAI format (system, user, assistant)
completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3.1-Terminus",
    messages=[
        {
            "role": "user",  # role can be "system", "user", or "assistant"
            "content": "On how many parameters you have been trained?"  # user query
        }
    ],
)

# Print the assistant's reply (first choice from response)
print(completion.choices[0].message)

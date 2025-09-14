# https://platform.openai.com/api-keys

# Import necessary modules
from langchain_openai import OpenAI  # LangChain wrapper for OpenAI's models
from dotenv import load_dotenv       # To load environment variables from a .env file
import os                            # For interacting with the operating system

# Load environment variables from a .env file (e.g., to access API keys securely)
load_dotenv()

# Print the loaded OpenAI API key (for debugging purposes only; remove in production)
print("Loaded API key", os.getenv("OPENAI_API_KEY"))

# Initialize the OpenAI language model using LangChain with a specific model
# 'gpt-3.5-turbo-instruct' is an instruction-following model suitable for prompts
llm = OpenAI(model='gpt-3.5-turbo-instruct')

# Use the model to generate a response for a simple question
result = llm.invoke("What is the capital of India")

# Output the result
print(result)

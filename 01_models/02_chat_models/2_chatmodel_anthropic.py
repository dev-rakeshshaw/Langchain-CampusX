# Import necessary modules
from langchain_anthropic import ChatAnthropic  # LangChain wrapper for Anthropic's Claude chat models
from dotenv import load_dotenv                 # For loading environment variables from a .env file

# Load environment variables from the .env file (typically used to load API keys securely)
load_dotenv()

# Initialize the Claude model from Anthropic via LangChain
# model='claude-3-5-sonnet-20241022' → Specific Claude model version (Sonnet, Oct 2024 release)
model = ChatAnthropic(model='claude-3-5-sonnet-20241022')

# Invoke the model with a simple prompt
# This sends the prompt to Claude and gets a response
result = model.invoke('What is the capital of India')

# Print only the text content of the model's response
print(result.content)

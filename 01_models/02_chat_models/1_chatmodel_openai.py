# https://platform.openai.com/api-keys

# Import necessary modules
from langchain_openai import ChatOpenAI  # LangChain wrapper for OpenAI's chat models
from dotenv import load_dotenv           # To load environment variables from a .env file

# Load environment variables from the .env file (e.g., to access the OpenAI API key securely)
load_dotenv()

# Initialize the ChatOpenAI model with specific configuration
# model='gpt-4'              → Uses GPT-4 model
# temperature=1.5            → High creativity in responses (more randomness)
# max_completion_tokens=10   → Limit the response to approx. 10 tokens (very short output)
model = ChatOpenAI(
    model='gpt-4',
    temperature=1.5,
    max_completion_tokens=10
)

# Invoke the model with a prompt to generate a response
# Since the max tokens are very low (10), the output will likely be cut off or short
result = model.invoke("Write a 5 line poem on cricket")

# Print only the content of the model's response
print(result.content)

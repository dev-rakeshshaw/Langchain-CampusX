#ai.google.dev/gemini-api/docs

# Import necessary modules
from langchain_google_genai import ChatGoogleGenerativeAI  # LangChain wrapper for Google Gemini models
from dotenv import load_dotenv                             # Used to load environment variables from a .env file

# Load environment variables from the .env file
# This is typically used to securely load your Google API key or credentials
load_dotenv()

# Initialize the Gemini model from Google via LangChain
# model='gemini-2.5-flash-lite' → A lightweight, fast version of the Gemini 2.5 model
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

# Send a prompt to the model and get the generated response
result = model.invoke('What is the capital of India')

# Print only the content (text) of the model's response
print(result.content)

# option_a_pipeline_direct.py
import os
from dotenv import load_dotenv

# Put big files on E: so C: doesn't fill up
os.environ['HF_HOME'] = 'E:/huggingface_cache'

load_dotenv()

from langchain_huggingface import HuggingFacePipeline

# Create the HF-backed pipeline wrapper (LangChain helper)
# Note: do not pass 'device' in pipeline_kwargs to avoid conflicts with internal calls
llm = HuggingFacePipeline.from_model_id(
    model_id="distilgpt2",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=64,
        do_sample=True,
        num_return_sequences=1,
    )
)

# Access the underlying transformers.pipeline object
pipeline_obj = llm.pipeline

prompt = "What is the capital of India?"

# Call the pipeline directly (it returns a list of dicts with 'generated_text')
out = pipeline_obj(prompt, max_new_tokens=64, do_sample=True, temperature=0.5, num_return_sequences=1)

print(out[0], type(out[0]['generated_text']))

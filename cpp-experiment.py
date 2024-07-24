from llama_cpp import Llama

llm = Llama(
      model_path="./llama-3.1-8B-Instruct.q8.gguf",
      n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # n_ctx=2048, # Uncomment to increase the context window
)
output = llm(
      "Who is Edna Mode?", # Prompt
      max_tokens=768, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["<|eot_id|>"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output)
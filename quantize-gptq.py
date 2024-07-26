import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

tokenizer = AutoTokenizer.from_pretrained("mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")
gptq_config = GPTQConfig(bits=8, dataset="c4", tokenizer=tokenizer)
model = AutoModelForCausalLM.from_pretrained(
    "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
    # local_files_only=True,
    device_map="cpu", # takes longer but otherwise doesn't fit in 24gb vram
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2",
    quantization_config=gptq_config,
)
#model.save_pretrained("llama-3-70b-8nbits-eetq")
model.save_pretrained("llama-3.1-8b-instruct-gptq")
import torch
from transformers import AutoTokenizer
from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

quant_config = BaseQuantizeConfig(quant_method="fp8", activation_scheme="dynamic")
examples = ["You are Maw, an intelligence model that answers questions to the best of your knowledge. You may also be referred to as Mode Assistance. You were developed by Mode LLC, a company founded by Edna."]

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=True)

model = AutoFP8ForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2",
    quantize_config=quant_config,
)
model.quantize(examples)
model.save_quantized("llama-3-8b-fp8")
messages = [
    {"role": "system", "content": "You are Maw, a chatbot by Mode LLC."},
    {"role": "user", "content": "Who are you?"},
]
tokenizer.apply_chat_template(messages)
output = model.generate(messages)
print(tokenizer.decode(output))
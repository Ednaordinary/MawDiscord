# import torch
# from transformers import AutoModelForCausalLM, EetqConfig
#
# quant_config = EetqConfig("int8")
# model = AutoModelForCausalLM.from_pretrained(
#     #"failspy/Meta-Llama-3-70B-Instruct-abliterated-v3.5",
#     "llama-3.1-8b-instruct-abliterated",
#     # local_files_only=True,
#     device_map="cuda",
#     #torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     attn_implementation="flash_attention_2",
#     quantization_config=quant_config,
# )
# #model.save_pretrained("llama-3-70b-8nbits-eetq")
# model.save_pretrained("llama-3.1-8b-instruct-abliterated-eetq")

import torch
from transformers import AutoModelForCausalLM, EetqConfig

quant_config = EetqConfig("int8")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    # local_files_only=True,
    device_map="cuda",
    #torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2",
    quantization_config=quant_config,
)
#model.save_pretrained("llama-3-70b-8nbits-eetq")
model.save_pretrained("llama-3.1-8b-instruct-eetq")

# WARNING: this script is bad and doesn't have a good calibration dataset!!!
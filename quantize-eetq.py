import torch
from transformers import AutoModelForCausalLM, EetqConfig

#quant_config = HqqConfig(nbits=2, axis=1, group_size=64, quant_zero=True, quant_scale=True, offload_meta=True,
#                         compute_dtype=torch.bfloat16)
quant_config = EetqConfig("int8")
model = AutoModelForCausalLM.from_pretrained(
    # "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3",
    "failspy/llama-3-70B-Instruct-abliterated",
    offload_meta=True,
    # local_files_only=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2",
    quantization_config=quant_config,
)
model.save_pretrained("llama-3-70b-8nbits-eetq")
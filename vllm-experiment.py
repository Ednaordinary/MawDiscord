from vllm import LLM, SamplingParams
import time

sampling_params = SamplingParams(temperature=0.6, top_p=0.9)
#failspy/Meta-Llama-3-8B-Instruct-abliterated-v3
model = LLM(model="failspy/Meta-Llama-3-70B-Instruct-abliterated-v3.5", speculative_model="[ngram]", num_speculative_tokens=15, ngram_prompt_lookup_max=4, quantization="fp8", use_v2_block_manager=True, cpu_offload_gb=55)
start_time = time.time()
outputs = model.generate("Who is Edna Mode?", sampling_params)
end_time = time.time()
for output in outputs:

    prompt = output.prompt

    generated_text = output.outputs[0].text

    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}, Tps: {round(len(output.outputs[0].token_ids)/(end_time - start_time), 2)}")
from modeling.exl2model import Exl2Engine, Exl2Loop
from modeling.tokenize import Tokenizer
from exllamav2 import ExLlamaV2Cache_Q4
from exllamav2.generator import ExLlamaV2Sampler
import time
import threading
import random

loop = Exl2Loop()
tokenizer = Tokenizer("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
# "dr1-1.5b-exl2-4.0bpw-hb8"
# engine = Exl2Engine("dr1-32b-abliterated-exl2-4.0bpw-hb8", 1024 * 40, ExLlamaV2Cache_Q4, loop, None, 512, None)
engine = Exl2Engine(
    "dr1-1.5b-exl2-4.0bpw-hb8", 1024 * 400, ExLlamaV2Cache_Q4, loop, None, 512, None
)

start_time = time.perf_counter()
count = 0

threads = []

prompt = "Please find the hottest man"
system = "You are a helpful assistant"

encoded = tokenizer.prompt_to_tokens(prompt, system)


def run_thread(prompt):
    temp = random.randint(600, 800) / 1000
    sampler = ExLlamaV2Sampler.Settings(
        top_p=0.8, top_k=50, token_repetition_penalty=1.07, min_p=0.01, temperature=temp
    )
    for i in engine.generate(
        prompt,
        add_bos=True,
        stop_token="<｜end▁of▁sentence｜>",
        max_tokens=128,
        sampler=sampler,
    ):
        global count
        count += 1
        print(i, end="", flush=True)
        # print(round(count/(time.perf_counter() - start_time)), end="\r", flush=True)


for i in range(60):
    thread = threading.Thread(target=run_thread, args=[encoded])
    thread.start()
    threads.append(thread)
for i in threads:
    i.join()
print()
print()
print("tps:", count / (time.perf_counter() - start_time))
print()
print()

time.sleep(2)
print("exiting")
engine.end()
print("exited")

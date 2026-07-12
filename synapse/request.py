import random
import threading
import sys
import traceback

from .neuron import Synapse

sys.path.append("..")
from character.request import RequestContext, TokenCount, SillySampler, stop_token
from character.history import Message
from modeling.tokenize import Tokenizer
from modeling.exl3model import Exl3Engine

def run_handler(engine: Exl3Engine, history, token_count, tokenizer, syn: Synapse, image_embeds=None):
    try:
        temp = random.randint(600, 2000) / 1000
        sampler = SillySampler(temperature=temp)
        answer = ""
        for i in engine.generate(history, add_bos=False, stop_token=stop_token, max_tokens=1024*32, sampler=sampler, image_embeds=image_embeds):
            if isinstance(i, bool):
                pass
            else:
                answer += i
                token_count.inc()
    except Exception as e:
        print(repr(e))
        print(traceback.format_exc())

class SynapseRequest:
    def __init__(self, context, cutoff, req_count):
        self.context: RequestContext = context
        self.cutoff = cutoff
        self.req_count = req_count
    def handle(self, engine, tokenizer: Tokenizer, discord_loop, channel_queue):
        token_count = TokenCount(self.req_count)
        try:
            self.context.history.add_message(Message(self.context.message.id, self.context.prompt, "user", images=self.context.images))
            vision = hasattr(engine, "vision")
            history = (self.context.history.to_tokenizer(limit=self.context.message.id, vision=engine.vision, exl_tokenizer=engine.tokenizer) if vision else self.context.history.to_tokenizer(limit=self.context.message.id))
            if isinstance(history, tuple):
                history, image_embeds = history
                history = tokenizer.history_to_tokens(history, cutoff=self.cutoff, embeds=image_embeds)
            else:
                history = tokenizer.history_to_tokens(history, cutoff=self.cutoff)
            run_thread = threading.Thread(
                target=run_handler,
                args=[
                    
                ],
                kwargs={'image_embeds': image_embeds}
            )
            run_thread.start()
            run_thread.join()
            return token_count.get()
        except Exception as e:
            print(repr(e))
            print(traceback.format_exc())
            return token_count.get()

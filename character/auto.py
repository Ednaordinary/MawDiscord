import re
import random
import asyncio
from queue import Queue

from exllamav3.generator.sampler.custom import *

from .request import Request
from .defaults import MawPrompts
from .history import UnwatchedHistory, Message

class AutoResponder:
    """
    Base implementation defining how the auto responder works
    """
    def __init__(self):
        self.messages = {}
    def log_message(self, message):
        if message.channel.id in self.messages:
            self.messages[message.channel.id].append(message)
        else:
            self.messages[message.channel.id] = [message]
    def should_respond(self):
        # Using previous messages, decide what to respond to based on criteria
        return [] # Messages to respond to

class SelfCriteriaRequest(Request):
    # __init__: prompt, queue, channel
    def handle(self, engine, tokenizer, discord_loop, channel_queue):
        try:
            regex = re.compile(r'|[\S\s]+?|')
            temp = random.randint(400, 700) / 1000
            sampler = ComboSampler(temperature=temp, min_p=0.01, top_k=40, top_p=0.95, rep_p=1.01)
            answer = ""
            for i in engine.generate(self.prompt, add_bos=False, stop_token=stop_token, max_tokens=1024 * 4, sampler=sampler):
                if isinstance(i, bool):
                    pass
                else:
                    answer += i
                    for msg_id in re.findall(regex, answer):
                        msg_id = msg_id[1:-1]
                        self.queue.put((msg_id, self.channel))
            self.queue.put((None, self.channel))
            return None
        except Exception as e:
            print(repr(e))
            print(traceback.format_exc())
            self.queue.put((None, self.channel))
            return None
    def update_progress(self, content, discord_loop):
        pass

class SelfResponder(AutoResponder):
    """
    Choose whether to respond based on what maw itself thinks (sillies)
    """
    def __init__(self, queue, client, cutoff):
        super().__init__()
        self.queue = queue
        self.client = client
        self.cutoff = cutoff
    def should_respond(self):
        responsives = []
        response_queue = Queue()
        for channel, messages in self.messages.items():
            if messages == []:
                del self.message[channel]
            else:
                sys = MawPrompts.default + "\n\n" + MawPrompts.default_personality + "\n\n" + MawPrompts.auto_response_criteria_sys
                history = UnwatchedHistory("", sys)
                for message in messages:
                    prefix = str(message.id) + " " + str(message.author.nick or message.author.global_name or message.author.name or "User").strip() + " said: "
                    role = "assistant" if message.author.id == client.user.id else "user"
                    history.append_message(Message(message.message_id, prefix + message.clean_content, role))
                history.sort_messages()
                history = history.to_tokenizer(includes="</think>")
                history = tokenizer.history_to_tokens(history, cutoff=self.cutoff)
                request = SelfCriteriaRequest(prompt=history, queue=response_queue, channel)
                responsives.append(channel)
                self.queue.put(request)
        while responsives != []:
            resp = response_queue.get()
            if resp[0] == None:
                try:
                    responsives.remove(resp[1])
                except:
                    pass
            else:
                try:
                    message = next(x for x in self.messages[channel] if x.id == int(resp[0]))
                except:
                    pass
                else:
                    yield (message, channel)

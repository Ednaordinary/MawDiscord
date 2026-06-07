from transformers import AutoTokenizer, AutoProcessor
from exllamav3 import Tokenizer as ExlTokenizer, Config


class Tokenizer:
    def __init__(self, model_id, config):
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_id)
        self.exl_tokenizer: ExlTokenizer = ExlTokenizer.from_config(
            Config.from_directory(config)
        )

    def prompt_to_tokens(self, prompt, system=None):
        conversation = []
        if system != None:
            conversation.append({"role": "system", "content": system})
        conversation.append({"role": "user", "content": prompt})
        tokenized = self.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        return tokenized

    def history_to_tokens(self, history, cutoff=None, keep_sys=True, embeds=None):
        # Please note keep_sys will flatten system messages to the top regardless of their relative locations,
        # as well as merge them into a single message
        if isinstance(history, tuple):
            history, embeds = history
            embeds = embeds if embeds is not [] else None
        if cutoff == None:
            formed = self.tokenizer.apply_chat_template(
                conversation=history,
                tokenize=False,
                add_generation_prompt=True,
            )
            return self.exl_tokenizer.encode(
                formed, encode_special_tokens=True, embeddings=embeds
            )
        else:
            if keep_sys:
                print(history)
                print(type(history))
                # sys = [{"content": [{"type": "text", "text": "\n".join([(x["content"] if isinstance(x["content"], str) else x["content"][-1]["text"]) for x in history if x["role"] == "system"])}], "role": "system"}]
                sys = [
                    {
                        "role": "system",
                        "content": "\n".join(
                            [x["content"] for x in history if x["role"] == "system"]
                        ),
                    }
                ]
                history = [x for x in history if x["role"] != "system"]
            else:
                sys = []
            fake_history = []
            last_tokenize = []
            for i in history[::-1]:
                fake_history.append(i)
                print(sys + fake_history[::-1])
                formed = self.tokenizer.apply_chat_template(
                    conversation=sys + fake_history[::-1],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                print(formed)
                tokenized = self.exl_tokenizer.encode(
                    formed, encode_special_tokens=True, embeddings=embeds
                )
                if len(tokenized[0]) > cutoff:
                    return last_tokenize
                else:
                    last_tokenize = tokenized
            return last_tokenize

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=False)


class Processor(Tokenizer):
    def __init__(self, model_id):
        self.tokenizer = AutoProcessor.from_pretrained(model_id)

from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    def prompt_to_tokens(self, prompt, system=None):
        conversation = []
        if system != None:
            conversation.append({"role": "system", "content": system})
        conversation.append({"role": "user", "content": prompt})
        tokenized = self.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=True, return_tensors = "pt", add_generation_prompt=True,
        )
        return tokenized
    def history_to_tokens(self, history, cutoff=None, keep_sys=True):
        # Please note keep_sys will flatten system messages to the top regardless of their relative locations,
        # as well as merge them into a single message
        if cutoff == None:
            return self.tokenizer.apply_chat_template(
                    conversation=history,
                    tokenize=True, return_tensors = "pt", add_generation_prompt=True,
                )
        else:
            if keep_sys:
                if history != []:
                    fake_history = [{"content": "\n".join([x["content"] for x in history if x["role"] == "system"]), "role": "system"}]
                else:
                    fake_history = []
                history = [x for x in history if x["role"] != "system"]
            else:
                fake_history = []
            last_tokenize = None
            print(history)
            for i in history[::-1]:
                print(i)
                fake_history.append(i)
                tokenized = self.tokenizer.apply_chat_template(
                    conversation=fake_history[::-1], tokenize=True,
                    return_tensors = "pt", add_generation_prompt=True,
                )
                if len(tokenized[0]) > cutoff:
                    return last_tokenize
                else: last_tokenize = tokenized
            print(self.decode(last_tokenize[0]))
            return last_tokenize
    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

import os

role_trans = {"user": "u", "system": "s", "assistant": "c"}
role_trans_rev = {"u": "user", "s": "system", "c": "assistant"}

class Message:
    def __init__(self, message_id, content, role):
        self.message_id = message_id
        self.content = content
        self.role = role

class History:
    def __init__(self, path, sys=None):
        self.history = []
        self.path = path
        self.sys = sys
    def touch_history(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            f.write("")
        if self.sys != None:
            self.append_message(Message(0, self.sys, "system"))
            self.write_history()
    def write_history(self):
        with open(self.path, "w") as file:
            for message in self.history:
                content = (role_trans[message.role] if message.role in role_trans.keys() else "c") + "|"
                content = content + str(message.message_id) + "|"
                content = content + message.content.replace("\n", r"\\n").replace("\r", r"\\n") + "\n" 
                file.write(content)
    def read_history(self, limit=None):
        self.history = []
        if os.path.isfile(self.path):
            with open(self.path, "r") as file:
                for line in file:
                    line = line.replace("\n", "")
                    line = line.split("|")
                    role, message_id, content = role_trans_rev[line[0].strip()], int(line[1]), "".join(line[2:]).replace(r"\\n", "\n")
                    self.history.append(Message(message_id, content, role))
        else:
            self.touch_history()
        if limit != None:
            self.history.sort(key=lambda x: x.message_id)
            return [x for x in self.history if x.message_id <= limit]
        else:
            return self.history
    def append_message(self, message):
        self.read_history()
        self.history.append(message)
        self.write_history()
    def add_message(self, message):
        self.append_message(message)
        self.sort_messages()
    def sort_messages(self):
        # Message IDs are in order, which we can use to our advantage
        self.read_history()
        self.history.sort(key=lambda x: x.message_id)
        self.write_history()
    def remove_message(self, message_id):
        self.read_history()
        for idx, i in enumerate(self.history):
            if i.message_id == message_id:
                self.history.pop(idx)
        self.write_history()
        self.sort_messages()
    def edit_message(self, message):
        self.read_history()
        if message.message_id not in [x.message_id for x in self.history]:
            self.add_message(message)
        else:
            for idx, i in enumerate(self.history):
                if i.message_id == message.message_id:
                    self.history[idx].content = message.content
        self.write_history()
    def to_tokenizer(self, group=True, grouper="\n", limit=None):
        if group:
            history = []
            group = ""
            last_role = None
            for i in self.read_history(limit=limit):
                if i.role == last_role:
                    group = group + grouper + i.content
                else:
                    if group != "":
                        history.append({"role": last_role, "content": group})
                    group = i.content
                    last_role = i.role
            if group != "":
                history.append({"role": last_role, "content": group})
            return history
        else:
            history = []
            for i in self.read_history():
                history.append({"role": i.role, "content": i.content})
            return history

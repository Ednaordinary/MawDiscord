import time
import os

role_trans = {"user": "u", "system": "s", "assistant": "c"}
role_trans_rev = {"u": "user", "s": "system", "c": "assistant"}

verbose = False

class Message:
    def __init__(self, message_id, content, role):
        self.message_id = int(message_id)
        self.content = str(content)
        self.role = role
    def __repr__(self):
        return "Message: (" + str(self.message_id) + " " + str(self.role) + ")"
    def __str__(self):
        return "Message: (" + str(self.message_id) + " " + str(self.role) + ")"

class History:
    def __init__(self, path, sys=None):
        self.history = []
        self.path = path
        self.sys = sys
        self.wait = False
        self.usable = True
    def touch_history(self):
        if verbose: print("touch_history called")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.wait = True
        with open(self.path, "w") as f:
            f.write("")
        self.wait = False
        if self.sys != None:
            if verbose: print("adding system message")
            self.add_message(Message(0, self.sys, "system"))
            if verbose: print("(sys):", self.history)
    def renew_sys(self):
        self.edit_message(Message(0, self.sys, "system"))
    def write_history(self):
        if verbose: print("write_history called")
        if verbose: print("(write):", self.history)
        content = ""
        for message in self.history:
            content += (role_trans[message.role] if message.role in role_trans.keys() else "c") + "|"
            content += str(message.message_id) + "|"
            content += message.content.replace("\n", r"\\n").replace("\r", r"\\n") + "\n"
        while self.wait:
            time.sleep(0.01)
        self.wait = True
        with open(self.path, "w") as file:
                file.write(content)
        self.wait = False
        self.read_history()
    def read_history(self, limit=None, usable=False):
        if verbose: print("read_history called")
        history = []
        if usable:
            while not self.usable and self.wait:
                time.sleep(0.01)
        else:
            while self.wait:
                time.sleep(0.01)
        if not os.path.isfile(self.path):
            self.touch_history()
        self.wait = True
        with open(self.path, "r") as file:
            for line in file:
                line = line.replace("\n", "")
                line = line.split("|")
                role, message_id, content = role_trans_rev[line[0].strip()], int(line[1]), "".join(line[2:]).replace(r"\\n", "\n")
                history.append(Message(message_id, content, role))
        self.wait = False
        self.history = history
        if verbose: print("(read):", self.history)
        if limit != None:
            #self.history.sort(key=lambda x: x.message_id)
            return [x for x in self.history if x.message_id <= limit]
        else:
            return self.history
    def append_message(self, message):
        if verbose: print("append_message called")
        self.read_history()
        if message.message_id not in [x.message_id for x in self.history]:
            self.history.append(message)
            self.write_history()
    def add_message(self, message):
        if verbose: print("add_message called")
        self.append_message(message)
        self.sort_messages()
    def sort_messages(self):
        # Message IDs are in order, which we can use to our advantage
        if verbose: print("sort_messages called")
        self.read_history()
        self.history.sort(key=lambda x: x.message_id)
        if verbose: print("(sort):", self.history)
        self.write_history()
    def remove_message(self, message_id):
        if verbose: print("remove_message called")
        self.read_history()
        self.history = [x for x in self.history if x.message_id != message_id]
        self.write_history()
        self.sort_messages()
    def edit_message(self, message):
        if verbose: print("edit_message called")
        self.usable = False
        self.read_history()
        if message.message_id not in [x.message_id for x in self.history]:
            self.add_message(message)
        else:
            for idx, i in enumerate(self.history):
                if i.message_id == message.message_id:
                    self.remove_message(message.message_id)
                    self.add_message(Message(message.message_id, message.content, i.role))
                    break
            self.write_history()
        self.usable = True
    def to_tokenizer(self, group=True, grouper="\n", limit=None):
        if verbose: print("to_tokenizer called")
        if group:
            history = []
            group = ""
            last_role = None
            for i in self.read_history(limit=limit, usable=True):
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
            for i in self.read_history(limit=limit, usable=True):
                history.append({"role": i.role, "content": i.content})
            return history

import traceback
import logging
import json
import time
import os

from json.decoder import JSONDecodeError

logger = logging.getLogger(__name__)

role_trans = {"user": "u", "system": "s", "assistant": "c"}
role_trans_rev = {"u": "user", "s": "system", "c": "assistant"}

class MessageJson(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, **kwargs)

class Message:
    def __init__(self, message_id, content, role):
        self.message_id = int(message_id)
        self.content = str(content)
        self.role = role
    def __str__(self):
        return "Message: (" + str(self.message_id) + " " + str(self.role) + ")"
    def __repr__(self):
        return self.__str__()
    def to_json(self):
        return MessageJson(message_id=self.message_id, content=self.content, role=self.role)

class History:
    """
    Message history autologged to a custom file format. You should use JsonHistory in most cases.
    """
    def __init__(self, path, sys=None):
        self.history = []
        self.path = path
        self.wait = False
        self.usable = True
        self.workers = []
        self.sys = sys if sys != None else self.get_sys()
    def touch_history(self):
        logger.debug("touch_history called")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.wait = True
        with open(self.path, "w") as f:
            f.write("")
        self.wait = False
        if self.sys != None:
            logger.debug("adding system message")
            self.add_message(Message(0, self.sys, "system"))
            logger.debug("(sys): " + str(self.history))
    def get_sys(self):
        self.read_history()
        for x in self.history:
            if x.message_id == 0:
                return x.content
        return None
    def renew_sys(self):
        self.edit_message(Message(0, self.sys, "system"))
    def write_history(self):
        logger.debug("write_history called")
        logger.debug("(write): " + str(self.history))
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
    def read_history(self, limit=None, usable=False, includes=None):
        logger.debug("read_history called")
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
        logger.debug("(read): " + str(self.history))
        if limit != None:
            #self.history.sort(key=lambda x: x.message_id)
            returned_history = [x for x in self.history if x.message_id <= limit]
        else:
            returned_history = self.history
        if includes != None:
            returned_history = [x for x in returned_history if (includes in x.content) or (x.role != "assistant")]
        return returned_history
    def append_message(self, message):
        logger.debug("append_message called")
        self.read_history()
        if message.message_id not in [x.message_id for x in self.history]:
            self.history.append(message)
            self.write_history()
    def add_message(self, message):
        logger.debug("add_message called")
        self.append_message(message)
        self.sort_messages()
    def sort_messages(self):
        # Message IDs are in order, which we can use to our advantage
        logger.debug("sort_messages called")
        self.read_history()
        self.history.sort(key=lambda x: x.message_id)
        logger.debug("(sort): " + str(self.history))
        self.write_history()
    def remove_message(self, message_id):
        logger.debug("remove_message called")
        self.read_history()
        self.history = [x for x in self.history if x.message_id != message_id]
        self.write_history()
        self.sort_messages()
    def edit_message(self, message):
        logger.debug("edit_message called")
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
    def to_tokenizer(self, group=True, grouper="\n", limit=None, includes=None):
        logger.debug("to_tokenizer called")
        if group:
            history = []
            group = ""
            last_role = None
            for i in self.read_history(limit=limit, usable=True, includes=includes):
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
    def to_unwatched(self):
        unwatched = UnwatchedHistory(self.path, self.sys)
        unwatched.copy_history(self.history)
        return unwatched

class JsonHistory(History):
    """
    Message history autologged to a JSON file
    """
    def __init__(self, path, sys=None):
        self.history = []
        self.path = path
        self.wait = False
        self.usable = True
        self.workers = []
        self.sys = sys if sys != None else self.get_sys()
    def touch_history(self):
        logger.debug("touch_history called")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.wait = True
        with open(self.path, "w") as f:
            json.dump([], f, indent=4)
        self.wait = False
        if self.sys != None:
            logger.debug("adding system message")
            self.add_message(Message(0, self.sys, "system"))
            logger.debug("(sys): " + str(self.history))
    def get_sys(self):
        self.read_history()
        for x in self.history:
            if x.message_id == 0:
                return x.content
        return None
    def renew_sys(self):
        self.edit_message(Message(0, self.sys, "system"))
    def write_history(self):
        logger.debug("write_history called")
        logger.debug("(write): " + str(self.history))
        while self.wait:
            time.sleep(0.01)
        self.wait = True
        with open(self.path, "w") as file:
            json_history = [x.to_json() for x in self.history]
            json.dump(json_history, file, indent=4)
        self.wait = False
        self.read_history()
    def read_history(self, limit=None, usable=False, includes=None):
        logger.debug("read_history called")
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
            try:
                self.history = json.load(file)
                self.history = [(Message(**i) if not isinstance(i, Message) else Message) for i in self.history]
            except Exception as e:
                print("file:", file.read())
                print(traceback.format_exc())
        self.wait = False
        logger.debug("(read): " + str(self.history))
        if limit != None:
            #self.history.sort(key=lambda x: x.message_id)
            returned_history = [x for x in self.history if x.message_id <= limit]
        else:
            returned_history = self.history
        if includes != None:
            returned_history = [x for x in returned_history if (includes in x.content) or (x.role != "assistant")]
        return returned_history

class UnwatchedHistory(History):
    """
    Works similarly to normal history, except content will stay in memory and not be updated to file.
    """
    def copy_history(self, history):
        self.history = history
    def touch_history(self):
        pass
    def write_history(self):
        pass
    def read_history(self, limit=None, usable=False, includes=None):
        if limit != None:
            #self.history.sort(key=lambda x: x.message_id)
            returned_history = [x for x in self.history if x.message_id <= limit]
        else:
            returned_history = self.history
        if includes != None:
            returned_history = [x for x in returned_history if (includes in x.content) or (x.role != "assistant")]
        return returned_history

import json
import os

class Config:
    def __init__(self, path):
        self.path = path
    def touch(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            f.write("")
    def write(self, config):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(config, f, indent=4)
    def get(self):
        try:
            with open(self.path, "r") as f:
                config = json.load(f)
        except:
            self.touch()
            config = {}
        return config

import os

class Config:
    def __init__(self, path):
        self.path = path
    def write(self, name):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            f.write(name.replace("\n", ""))
    def get(self):
        config = {}
        with open(self.path, "r") as f:
            lines = [(x[:-1] if x[-1] == "\n" else x) for x in f.readlines()]
            config["name"] = lines[0]
        return config

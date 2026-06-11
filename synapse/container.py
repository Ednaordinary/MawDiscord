import podman
import ujson
from pathlib import Path
import traceback
import ftfy
import cleantext

client = podman.PodmanClient()
with open(str(Path(__file__).parent) + "/config.json", "r") as config_file:
    client_config = ujson.decode(config_file.read())

class Container:
    def __init__(self):
        self.container = client.containers.create(**client_config)
        self.users = 0
        
    def run(self, cmd):
        self.users += 1
        if self.users == 1:
            self.container.start()
        output = ""
        try:
            _, output_bytes = self.container.exec_run(cmd, demux=True)
            stdout_bytes, stderr_bytes = output_bytes
            stdout_bytes, stderr_bytes = stdout_bytes or b"", stderr_bytes or b""
            stdout = ftfy.fix_text(stdout_bytes.decode("utf-8", errors='replace'))
            stdout = cleantext.clean(stdout, to_ascii=False, lower=False).strip()
            stderr = ftfy.fix_text(stderr_bytes.decode("utf-8", errors='replace'))
            stderr = cleantext.clean(stderr, to_ascii=False, lower=False).strip()
            match stdout, stderr:
                case "", "":
                    output = "(No output)"
                case str(), "":
                    output = stdout
                case "", str():
                    output = "err:\n" + stderr
                case str(), str():
                    output = stdout + "\n\nerr:\n" + stderr
        except:
            print(traceback.format_exc())
        self.users -= 1
        if self.users == 0:
            self.container.stop(timeout=0)
        return output

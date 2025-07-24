import time

class RequestsCounter:
    def __init__(self, path):
        self.path = path
        self.block = False
    def inc(self):
        while self.block == True:
            time.sleep(0.01)
        self.block = True
        try:
            with open(self.path, 'rb') as file:
                counter = int.from_bytes(file.read(), byteorder='big')
        except:
            counter = 0
        counter += 1
        with open(self.path, 'wb') as file:
            file.write((counter).to_bytes((max(counter.bit_length() + 7, 1) // 8), byteorder='big', signed=False))
        self.block = False
    def get(self):
        try:
            with open(self.path, 'rb') as file:
                counter = int.from_bytes(file.read(), byteorder='big')
        except:
            counter = 0
        return counter

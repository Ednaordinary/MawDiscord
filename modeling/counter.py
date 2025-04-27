import time

class RequestsCounter:
    def __init__(self, path):
        self.path = path
        self.block = False
    def inc(self):
        while self.block == True:
            time.sleep(0.01)
        self.block = True
        with open(self.path, 'rb') as file:
            counter = int.from_bytes(file.read(), byteorder='big')
        counter += 1
        with open(self.path, 'wb') as file:
            file.write((i).to_bytes((max(i.bit_length() + 7, 1) // 8), byteorder='big', signed=False))
    def get(self):
        with open(self.path, 'rb') as file:
            counter = int.from_bytes(file.read(), byteorder='big')
        return counter

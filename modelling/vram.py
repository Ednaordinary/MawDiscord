# THIS FILE IS MODIFIED FROM https://github.com/Ednaordinary/ModelManager

import time

model_manager_path = "../ModelManager/allocation.txt" # This will be different depending on the path to the model manager

class Vram:
    def __init__(self):
        pass
    def allocate(self, name):
        allocate = True
        with open(model_manager_path, "r") as allocation_file:
            lines = allocation_file.readlines()
            if name.strip() in [x[:-1] for x in lines]:
                allocate = False
        if allocate:
            with open(model_manager_path, "a") as allocation_file:
                allocation_file.write(name+"\n")

    def deallocate(self, name):
        with open(model_manager_path, "r") as allocation_file:
            lines = allocation_file.readlines() 
        with open(model_manager_path, "w") as allocation_file:
            for i in lines:
                if i[:-1] != name:
                    allocation_file.write(i)

    def wait_for_allocation(self, name):
        last_allocation = None
        while True:
            with open(model_manager_path, "r") as allocation_file:
                lines = allocation_file.readlines()
                if len(lines) != 0:
                    if lines[0][:-1] == name:
                        break
                    else:
                        if lines[0][:-1] != last_allocation:
                            yield lines[0][:-1]
                            last_allocation = lines[0][:-1]
            time.sleep(0.02)
        return

    def get_allocations(self):
        with open(model_manager_path, "r") as allocation_file:
            lines = allocation_file.readlines()
            return [x.replace("\n", "").strip() for x in lines]

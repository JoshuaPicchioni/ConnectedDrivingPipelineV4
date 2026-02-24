from csv import writer
import os
import os.path as path
import string
import fcntl
from typing import List

from EasyMLLib.helper import Helper

OUTPUT_FOLDER_PATH = path.join("Outputs", "Output")

class CSVWriter:
    def __init__(self, name: string, columns: List[str], outputpath=OUTPUT_FOLDER_PATH):
        self.name = name
        self.columns = columns
        self.outputpath = outputpath
        Helper().createPath(outputpath)
        
        filepath = path.join(outputpath, self.name)
        
        # Only write header if file does not exist or is empty
        # Use file locking to prevent race conditions in parallel execution
        write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
        
        if write_header:
            with open(filepath, "w", newline='') as file:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)
                try:
                    # Double-check after acquiring lock
                    if file.tell() == 0:
                        writerObj = writer(file)
                        writerObj.writerow(columns)
                finally:
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)


    def addRow(self, row: list):
        filepath = path.join(self.outputpath, self.name)
        with open(filepath, "a", newline='') as file:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            try:
                writerObj = writer(file)
                writerObj.writerow(row)
            finally:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)

from typing import Optional
from collections import defaultdict
import os
import json
import logging
import sys

import numpy as np

class MetricLogger(object):

    def __init__(self, 
                fname: Optional[str] = "test",
                delimiter: str = "\t"
                ) -> None:
        self.metrics = {}
        self.delimiter = delimiter

        logging.basicConfig(
        level=logging.INFO,
        format = '%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('stroke-processing.log'),
            logging.StreamHandler(sys.stdout)
        ]
        )
        self.logger = logging.getLogger(__name__)
    
    def add_vals(self, 
                    keys: list,
                    metrics: list
                ) -> None:
        self.metrics.clear()

        for k, v in zip(keys, metrics):
            self.metrics[k] = v
    
    def __str__(self) -> str:
        res_str = []
        for k, v in self.metrics.items():
            res_str.append(
                f"{k}:{v}"
            )
        return self.delimiter.join(res_str)
    
    def log_msg(self,
                msg: str
                ) -> None:
        self.logger.info(msg)
    

class SaveVals(object):

    def __init__(self,
                fname: str
                ) -> None:
        
        self.fname = fname
        self.vals = {}
    
    def add_vals(self,
                keys: list,
                vals: list
                ) -> None:
        self.vals.clear()
        for k, v in zip(keys, vals):
            self.vals[k] = v

    def save(self):
        os.makedirs(os.path.dirname(self.fname), exist_ok=True)
        with open(self.fname, 'a+', encoding='utf-8') as f:
            json_line = json.dumps(self.vals)
            f.write(json_line + "\n")

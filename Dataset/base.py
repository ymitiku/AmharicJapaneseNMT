from Utils import load_config
import sentencepiece as spm
import os
import glob
from Exceptions import InvalidStateError
from tensorflow.keras.preprocessing import sequence
import numpy as np


class DataGenerator(object):
    def __init__(self, config):
        self.config = config
        self._load_dataset()

    def generate(self, case = None):
        return self._generate(case=case)

    def _get_data_files(self):
        encoded_data_path = self.config["output"]["spm"]["encoded_dir"]
        files = glob.glob(encoded_data_path + "/*." + self.language)
        if len(files) == 0:
            raise InvalidStateError(
                "No encoded files for %s language. Please first encode dataset before using Datagenerator" % self.language)
        for f in files:
            if f.count("train") > -1:
                train_file = f
            if f.count("dev") > -1:
                dev_file = f
            if f.count("test") > -1:
                test_file = f
        return train_file, dev_file, test_file

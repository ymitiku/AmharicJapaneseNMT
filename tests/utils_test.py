import unittest
from Utils import load_config

class UtilsTest(unittest.TestCase):
    def test_load_config(self):
        config = load_config("config/config.yml")
        self.assertNotEqual(config, None)
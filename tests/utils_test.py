import unittest
from Utils import load_config

class UtilsTest(unittest.TestCase):
    def test_load_config_reading(self):
        config = load_config("config/config.yml")
        self.assertNotEqual(config, None)
    def test_load_config_exception(self):
        self.assertRaises(FileNotFoundError, load_config, "invalid-filename")
    def 
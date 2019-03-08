

class Task(object):
    def __init__(self, config, language):
        self.config = config
        self.language = language

    def execute(self):
        return self._execute()

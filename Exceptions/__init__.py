class InvalidStateError(ValueError):
    def __init__(self, message):
        super(InvalidStateError, self).__init__(message)
    
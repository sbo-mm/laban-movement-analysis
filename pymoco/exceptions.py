class Error(Exception):
    """Base class for other exceptions"""
    pass

class ResourceNotLoadedError(Error):
    pass

class ResourceExhaustedError(Error):
    pass

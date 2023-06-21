"""Exceptions used in WSInfer."""


class WsinferException(Exception):
    """Base class for wsinfer exceptions."""


class UnknownArchitectureError(WsinferException):
    """Architecture is unknown and cannot be found."""


class WholeSlideImageDirectoryNotFound(FileNotFoundError):
    ...


class WholeSlideImagesNotFound(FileNotFoundError):
    ...


class ResultsDirectoryNotFound(FileNotFoundError):
    ...


class PatchDirectoryNotFound(FileNotFoundError):
    ...

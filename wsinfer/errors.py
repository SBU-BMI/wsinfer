"""Exceptions used in WSInfer."""

from __future__ import annotations


class WsinferException(Exception):
    """Base class for wsinfer exceptions."""


class UnknownArchitectureError(WsinferException):
    """Architecture is unknown and cannot be found."""


class WholeSlideImageDirectoryNotFound(WsinferException, FileNotFoundError):
    ...


class WholeSlideImagesNotFound(WsinferException, FileNotFoundError):
    ...


class ResultsDirectoryNotFound(WsinferException, FileNotFoundError):
    ...


class PatchDirectoryNotFound(WsinferException, FileNotFoundError):
    ...


class CannotReadSpacing(WsinferException):
    ...


class NoBackendException(WsinferException):
    ...

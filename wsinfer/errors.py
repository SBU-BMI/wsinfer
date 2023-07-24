"""Exceptions used in WSInfer."""

from __future__ import annotations


class WsinferException(Exception):
    """Base class for wsinfer exceptions."""


class UnknownArchitectureError(WsinferException):
    """Architecture is unknown and cannot be found."""


class WholeSlideImageDirectoryNotFound(WsinferException, FileNotFoundError):
    ...


class DuplicateFilePrefixesFound(WsinferException):
    """A duplicate file prefix has been found.

    An example of duplicate file prefixes is files a.svs and a.tif. WSInfer relies on
    the stems as a unique ID, so we cannot allow duplicate stems.
    """


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

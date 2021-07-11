# Copyright 2021 the authors.
# This file is part of Hy, which is free software licensed under the Expat
# license. See the LICENSE.

import re

import hy.errors
from hy.models import Expression, Symbol

from .exceptions import LexException, PrematureEndOfInput  # NOQA
from .mangle import isidentifier, mangle, unmangle
from .reader import HyReader

__all__ = [
    "mangle",
    "unmangle",
    "isidentifier",
    "read",
    "read_str",
    "read_file_contents",
]

class Module:
    def __init__(self, base, source, filename):
        self._base = base
        self.source = source
        self.filename = filename
    def __getattr__(self, attr):
        return getattr(self._base, attr)
    def __iter__(self):
        yield from self._base

def read_many(source, filename=None, reader=None):
    """Parse Hy source as a sequence of forms.

    Args:
      source (string): Source code to parse.
      filename (string, optional): File name corresponding to source.  Defaults to None.

    Returns:
      out : Sequence[hy.models.Expression]
    """
    if reader is None:
        reader = HyReader(source, filename)
    return reader.parse(source, filename)


def read(source):
    filename = "<string>"
    parser = HyReader(source, filename)
    return parser.parse_one_node()


def read_module(source, filename='<string>', reader=None):
    """Parse a Hy source file's contents. Treats the input as a complete module.
    Also removes any shebang line at the beginning of the source.

    Args:
      source (string): Source code to parse.
      filename (string, optional): File name corresponding to source.  Defaults to "<string>".

    Returns:
      out : hy.models.Expression
    """
    _source = re.sub(r'\A#!.*', '', source)
    res = read_many(_source, filename=filename, reader=reader)
    res = Module(res, source, filename)
    return res

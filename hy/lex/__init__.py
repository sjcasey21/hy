# Copyright 2021 the authors.
# This file is part of Hy, which is free software licensed under the Expat
# license. See the LICENSE.

import re

import hy.errors
from hy.models import Expression, Symbol

from .exceptions import LexException, PrematureEndOfInput  # NOQA
from .mangle import isidentifier, mangle, unmangle
from .reader import HyReader, UnexpectedEOF

__all__ = [
    "mangle",
    "unmangle",
    "isidentifier",
    "read",
    "read_str",
    "read_file_contents",
]

class Module:
    def __init__(self, base, source, filename, reader):
        self._base = base
        self.source = source
        self.filename = filename
        self._reader = reader
    def __getattr__(self, attr):
        return getattr(self._base, attr)
    def __iter__(self):
        try:
            yield from self._base
        except UnexpectedEOF as e:
            raise PrematureEndOfInput(e.msg, None, self.filename, self.source, *e.pos) from e
        except hy.errors.HyError:
            raise
        except Exception as e:
            raise LexException(str(e), None, self.filename, self.source, *self._reader.pos) from e

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
    try:
        return reader.parse(source)
    except UnexpectedEOF as e:
        raise PrematureEndOfInput(e.msg, None, filename, source, *e.pos) from e
    except hy.errors.HyError:
        raise
    except Exception as e:
        raise LexException(str(e), None, filename, source, *reader.pos) from e


def read(source):
    filename = "<string>"
    parser = HyReader(source, filename)
    try:
        return parser.parse_one_node()
    except UnexpectedEOF as e:
        raise PrematureEndOfInput(e.msg, None, filename, source, *e.pos) from e
    except hy.errors.HyError:
        raise
    except Exception as e:
        raise LexException(str(e), None, filename, source, *parser.pos) from e


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
    res = Module(res, source, filename, reader)
    return res

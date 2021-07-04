# Copyright 2021 the authors.
# This file is part of Hy, which is free software licensed under the Expat
# license. See the LICENSE.

import re

from hy.models import Expression, Symbol

from .exceptions import LexException, PrematureEndOfInput  # NOQA
from .mangle import isidentifier, mangle, unmangle
from .reader import HyParser, UnexpectedEOF

__all__ = [
    "mangle",
    "unmangle",
    "isidentifier",
    "read",
    "read_str",
    "read_file_contents",
]


def read_many(source, filename=None):
    """Parse Hy source as a sequence of forms.

    Args:
      source (string): Source code to parse.
      filename (string, optional): File name corresponding to source.  Defaults to None.

    Returns:
      out : Sequence[hy.models.Expression]
    """
    parser = HyParser(source, filename)
    try:
        return parser.parse()
    except UnexpectedEOF as e:
        raise PrematureEndOfInput(e.msg, None, filename, source, *e.pos) from e
    except Exception as e:
        raise LexException(str(e), None, filename, source, *parser.pos) from e


def read(source):
    filename = "<string>"
    parser = HyParser(source, filename)
    try:
        return parser.parse_one_node()
    except UnexpectedEOF as e:
        raise PrematureEndOfInput(e.msg, None, filename, source, *e.pos) from e
    except Exception as e:
        raise LexException(str(e), None, filename, source, *parser.pos) from e


def read_module(source, filename='<string>'):
    """Parse a Hy source file's contents. Treats the input as a complete module.
    Also removes any shebang line at the beginning of the source.

    Args:
      source (string): Source code to parse.
      filename (string, optional): File name corresponding to source.  Defaults to "<string>".

    Returns:
      out : hy.models.Expression
    """
    _source = re.sub(r'\A#!.*', '', source)
    res = Expression([Symbol("do")] + read_many(_source + "\n", filename=filename))
    res.source = source
    res.filename = filename
    return res

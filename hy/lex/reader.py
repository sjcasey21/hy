# Copyright 2021 the authors.
# This file is part of Hy, which is free software licensed under the Expat
# license. See the LICENSE.

import itertools
from collections import deque
from contextlib import contextmanager
from io import StringIO
from types import ModuleType

import hy
from hy.models import (
    Bytes,
    Complex,
    Dict,
    Expression,
    FComponent,
    Float,
    FString,
    Integer,
    Keyword,
    List,
    Set,
    String,
    Symbol,
)

from .mangle import mangle

NON_IDENT = '()[]{};"\''


class UnexpectedEOF(Exception):
    def __init__(self, pos, msg):
        super().__init__(self, msg)
        self.pos = pos
        self.msg = msg


DEFAULT_TABLE = {}


def sym(name):
    return Symbol(name, from_parser=True)


def mkexpr(root, *args):
    if isinstance(root, str):
        root = sym(root)
    return Expression((root, *args))


def symbol_like(ident, from_parser=False):
    try:
        return Integer(ident)
    except ValueError:
        pass
    try:
        num, denom = ident.split("/")
        return mkexpr("hy._Fraction", Integer(num), Integer(denom))
    except ValueError:
        pass
    try:
        return Float(ident)
    except ValueError:
        pass
    if ident not in ('j', 'J'):
        try:
            return Complex(ident)
        except ValueError:
            pass

    if "." in ident:
        for chunk in ident.split("."):
            if chunk and not isinstance(
                symbol_like(chunk, from_parser=from_parser), Symbol
            ):
                raise ValueError(
                    f'Cannot access attribute on anything other'
                    ' than a name (in order to get attributes of expressions,'
                    ' use `(. <expression> <attr>)` or `(.<attr> <expression>)`)'
                )

    if not from_parser:
        if (
            not ident
            or ident[:1] == ":"
            or any(c.isspace() for c in ident)
            or set(NON_IDENT).intersection(ident)
        ):
            raise ValueError(f'Syntactically illegal symbol: {ident!r}')

    return sym(ident)


def reader_for(char, args=None):
    def wrapper(f):
        if args is not None:
            DEFAULT_TABLE[char] = f(*args)
        else:
            DEFAULT_TABLE[char] = f
        return f

    return wrapper


class HyParser:
    def __init__(self, source, filename):
        self._set_source(source, filename)
        self._module = ModuleType('<reader>')
        self._module.__dict__.update({
            "hy": hy,
            mangle("&reader"): self,
        })

        self.ends_ident = set(NON_IDENT)
        self.parse_default = HyParser.ident_or_prefixed_string
        self.reader_table = DEFAULT_TABLE.copy()

    def _set_source(self, source=None, filename=None):
        if filename is not None:
            self._filename = filename
        if source is not None:
            self._source = source
            self._stream = StringIO(self._source)
            self._peek_chars = deque()
            self._saved_chars = []
            self._pos = (1, 0)
            self._eof_tracker = self._pos

    @property
    def pos(self):
        return self._pos

    ###
    # Utility
    ###

    def fill_node(self, node, start):
        if node is not None:
            node.start_line, node.start_column = start
            node.end_line, node.end_column = self._pos
            return node

    ###
    # Character streaming
    ###

    @contextmanager
    def saving_chars(self):
        self._saved_chars.append([])
        yield self._saved_chars[-1]
        saved = self._saved_chars.pop()
        if self._saved_chars:
            self._saved_chars[-1].extend(saved)

    def peekc(self):
        if self._peek_chars:
            return self._peek_chars[-1]
        nc = self._stream.read(1)
        self._peek_chars.append(nc)
        return nc

    def peeking(self, eof_ok=False):
        for nc in reversed(self._peek_chars):
            yield nc
        while True:
            c = self._stream.read(1)
            if not c:
                break
            self._peek_chars.appendleft(c)
            yield c
        if not c and not eof_ok:
            raise UnexpectedEOF(
                self._eof_tracker, "Premature end of input while peeking"
            )

    def getc(self):
        """This function does bookkeeping, so it's important
        that any character consumption go through this function"""
        c = self.peekc()
        self._peek_chars.pop()

        if c:
            line, col = self._pos
            col += 1
            if c == '\n':
                line += 1
                col = 0
            self._pos = (line, col)

            if not c.isspace():
                self._eof_tracker = self._pos

        if self._saved_chars:
            self._saved_chars[-1].append(c)

        return c

    def peek_and_getc(self, target):
        """Peek one character. If it is equal to `target`,
        then consume it and return True. Otherwise return False."""
        nc = self.peekc()
        if nc == target:
            self.getc()
            return True
        return False

    def peekahead(self, eof_ok=False):
        while True:
            nc = self.peekc()
            if not nc:
                break
            yield nc
            self.getc()
        if not nc and not eof_ok:
            raise UnexpectedEOF(
                self._eof_tracker, "Premature end of input while peeking ahead"
            )

    def chars(self, eof_ok=False):
        while True:
            c = self.getc()
            if not c:
                break
            yield c
        if not c and not eof_ok:
            raise UnexpectedEOF(
                self._eof_tracker, "Premature end of input while streaming chars"
            )

    ###
    # Reading multiple characters
    ###

    def getn(self, n):
        return ''.join(itertools.islice(self.chars(), n))

    def slurp_space(self):
        s = []
        for c in self.peekahead(eof_ok=True):
            if not c.isspace():
                break
            s.append(c)
        return ''.join(s)

    def read_ident(self, just_peeking=False):
        ident = []
        for nc in self.peeking(eof_ok=True):
            if not nc:
                # EOF, but that's ok
                break
            if nc in self.ends_ident:
                break
            if nc.isspace():
                break
            ident.append(nc)
        if not just_peeking:
            self.getn(len(ident))
        return ''.join(ident)

    ###
    # Reading AST nodes
    ###

    def _try_parse_one_node(self):
        self.slurp_space()
        c = self.getc()
        start = self._pos
        if not c:
            raise UnexpectedEOF(
                self._eof_tracker,
                "Premature end of input while attempting to parse one node",
            )
        handler = self.reader_table.get(c, self.parse_default)
        node = handler(self, c)
        return self.fill_node(node, start)

    def parse_one_node(self):
        node = None
        while node is None:
            node = self._try_parse_one_node()
        return node

    def parse_nodes_until(self, closer):
        while True:
            self.slurp_space()
            if self.peek_and_getc(closer):
                break
            node = self._try_parse_one_node()
            if node is not None:
                yield node

    def parse(self, source=None):
        rname = mangle("&reader")
        old_reader = getattr(hy, rname, None)
        setattr(hy, rname, self)

        self._set_source(source)
        yield from self.parse_nodes_until('')

        if old_reader is None:
            delattr(hy, rname)
        else:
            setattr(hy, rname, old_reader)

    ###
    # Reader dispatch logic
    ###

    def do_dispatch(self, tag):
        return self.reader_table[tag](self, tag)

    def ident_or_prefixed_string(self, key):
        ident = key + self.read_ident()
        if self.peek_and_getc('"'):
            return self.prefixed_string('"', ident)
        return symbol_like(ident, from_parser=True)

    ###
    # Basic atoms
    ###

    @reader_for(";")
    def line_comment(self, _):
        any(c == "\n" for c in self.chars(eof_ok=True))
        return None

    @reader_for(":")
    def keyword(self, _):
        ident = self.read_ident()
        if "." in ident:
            raise ValueError(
                f'Cannot access attribute on anything other'
                ' than a name (in order to get attributes of expressions,'
                ' use `(. <expression> <attr>)` or `(.<attr> <expression>)`)'
            )
        return Keyword(ident, from_parser=True)

    @reader_for('"')
    def prefixed_string(self, _, prefix=""):
        escaping = False

        def quote_closing(c):
            nonlocal escaping
            if c == '\\':
                escaping = not escaping
                return 0
            if c == '"' and not escaping:
                return 1
            escaping = False
            return 0

        return self.read_string_until(quote_closing, prefix, 'f' in prefix.lower())

    ###
    # Special annotations
    ###

    @reader_for("'", ("quote",))
    @reader_for("`", ("quasiquote",))
    def tag_as(root):
        def _tag_as(self, _):
            nc = self.peekc()
            if nc.isspace() or self.reader_table.get(nc) == self.INVALID:
                raise ValueError("Could not identify the next token.")
            node = self.parse_one_node()
            return mkexpr(root, node)

        return _tag_as

    @reader_for("~")
    def unquote(self, key):
        nc = self.peekc()
        if nc.isspace() or self.reader_table.get(nc) == self.INVALID:
            return sym(key)
        if self.peek_and_getc("@"):
            root = "unquote-splice"
        else:
            root = "unquote"
        node = self.parse_one_node()
        return mkexpr(root, node)

    @reader_for("^")
    def annotate_or_xor(self, _):
        suffix = self.read_ident(just_peeking=True)
        if suffix == "":
            return sym("^")
        if suffix == "=":
            self.getc()
            return sym("^=")
        node = self.parse_one_node()
        return mkexpr("annotate", node)

    ###
    # Sequences
    ###

    @reader_for(")")
    @reader_for("]")
    @reader_for("}")
    def INVALID(self, key):
        raise ValueError(f"Ran into a '{key}' where it wasn't expected.")

    @reader_for("[", (List, "]"))
    @reader_for("{", (Dict, "}"))
    @reader_for("#{", (Set, "}"))
    def sequence(seq_type, closer):
        def _sequence(self, _):
            return seq_type(self.parse_nodes_until(closer))

        return _sequence

    @reader_for("(")
    def expression(self, _):
        """Allow for special roots `eval-and-read` and `eval-when-read`
        that are similar to `eval-and-compile` and `eval-when-compile`,
        but operate at read time and have access to this reader instance
        through the special variable `&reader`."""
        nodes = list(self.parse_nodes_until(")"))
        if nodes and isinstance(nodes[0], Symbol):
            root = str(nodes[0])
            if root in ("eval-and-read", "eval-when-read"):
                # need to import here to prevent circular partial import
                from hy.compiler import hy_eval

                fnode = mkexpr("do", *nodes[1:])
                hy_eval(
                    fnode,
                    self._module.__dict__,
                    self._module,
                    filename=self._filename,
                    source=self._source,
                    import_stdlib=False,
                )
                return (
                    mkexpr("eval-and-compile", *nodes[1:])
                    if root == "eval-and-read"
                    else None
                )
        return Expression(nodes)

    ###
    # Reader tag-macros
    ###

    @reader_for("#")
    def dispatch(self, key):
        if not self.peekc():
            raise UnexpectedEOF(
                self._eof_tracker, "Premature end of input while attempting dispatch"
            )

        # try dispatching tagged ident
        ident = self.read_ident(just_peeking=True)
        if ident and key + ident in self.reader_table:
            self.getn(len(ident))
            return self.do_dispatch(key + ident)

        # failing that, dispatch tag + single character
        if key + self.peekc() in self.reader_table:
            tag = key + self.getc()
            return self.do_dispatch(tag)

        # fall back to old tag macro behavior
        tag = key + self.read_ident()
        if tag == key:
            raise ValueError("empty tag name")
        return mkexpr(tag, self.parse_one_node())

    @reader_for("#_")
    def discard(self, _):
        # discard the next node
        self.parse_one_node()
        return None

    @reader_for("#*")
    def hash_star(self, _):
        num_stars = 1
        while self.peek_and_getc("*"):
            num_stars += 1
        if num_stars > 2:
            raise ValueError("too many stars")
        if num_stars == 1:
            root = "unpack-iterable"
        else:
            root = "unpack-mapping"
        node = self.parse_one_node()
        return mkexpr(root, node)

    @reader_for("#@")
    def decorate(self, _):
        node = self.parse_one_node()
        if not isinstance(node, Expression):
            raise ValueError("can only decorate function or class definitions")
        if not node:
            import hy.errors
            raise hy.errors.HyMacroExpansionError("empty decoration")
        decorators, defn = node[:-1], node[-1]
        return mkexpr("with-decorator", *decorators, defn)

    @reader_for("#(")
    def tuple_list(self, _):
        return mkexpr(",", *self.parse_nodes_until(")"))

    ###
    # Strings
    # these are more complicated because f-strings
    # are effectively their own sublanguage
    ###

    @reader_for("#[")
    def bracketed_string(self, _):
        delim = []
        for c in self.chars():
            if c == "[":
                break
            elif c == "]":
                raise ValueError("Ran into a ']' where it wasn't expected.")
            delim.append(c)
        delim = ''.join(delim)
        is_fstring = delim == "f" or delim.startswith("f-")

        # discard single initial newline, if any
        self.peek_and_getc('\n')

        index = -1

        def delim_closing(c):
            nonlocal index
            if c == "]":
                if index == len(delim):
                    # this is the second bracket at the end of the delim
                    return len(delim) + 2
                else:
                    # reset state, this may be the first bracket of closing delim
                    index = 0
            elif 0 <= index <= len(delim):
                # we're inside a possible closing delim
                if index < len(delim) and c == delim[index]:
                    index += 1
                else:
                    # failed delim, reset state
                    index = -1
            return 0

        return self.read_string_until(delim_closing, None, is_fstring, brackets=delim)

    def read_string_until(self, closing, prefix, is_fstring, **kwargs):
        if is_fstring:
            components = self.read_fcomponents_until(closing, prefix)
            return FString(components, **kwargs)
        s = self.read_chars_until(closing, prefix, is_fstring=False)
        if isinstance(s, bytes):
            return Bytes(s, **kwargs)
        return String(''.join(s), **kwargs)

    def read_chars_until(self, closing, prefix, is_fstring):
        s = []
        for c in self.chars():
            s.append(c)
            # check if c is closing
            n_closing_chars = closing(c)
            if n_closing_chars:
                # string has ended
                s = s[:-n_closing_chars]
                break
            # check if c is start of component
            if is_fstring and c == "{":
                # check and handle "{{"
                if self.peek_and_getc("{"):
                    s.append("{")
                else:
                    # remove "{" from end of string component
                    s.pop()
                    break
        res = ''.join(s)
        if prefix is not None:
            res = eval(f'{prefix}"""{res}"""')
        if is_fstring:
            return res, n_closing_chars
        return res

    def read_fcomponents_until(self, closing, prefix):
        components = []
        start = self.pos
        while True:
            s, closed = self.read_chars_until(closing, prefix, is_fstring=True)
            if s:
                components.append(self.fill_node(String(s), start))
            if closed:
                break
            components.extend(self.read_fcomponent(prefix))
        return components

    def read_fcomponent(self, prefix):
        """May return one or two components, since the `=` debugging syntax
        will create a String component."""
        start = self.pos
        values = []
        conversion = None
        has_debug = False

        # read the expression, saving the text verbatim
        # in case we encounter debug `=`
        space_before = self.slurp_space()
        with self.saving_chars() as node_text:
            node = self.parse_one_node()
        space_between = self.slurp_space()

        # check for and handle debug syntax:
        # we emt the verbatim text before we emit the value
        if self.peek_and_getc("="):
            has_debug = True
            space_after = self.slurp_space()
            dbg_prefix = (
                space_before + ''.join(node_text) + space_between + "=" + space_after
            )
            values.append(self.fill_node(String(dbg_prefix), start))

        # handle conversion code
        if self.peek_and_getc("!"):
            conversion = self.getc()
        self.slurp_space()

        def component_closing(c):
            if c == "}":
                return 1
            return 0

        # handle formatting options
        format_components = []
        if self.peek_and_getc(":"):
            format_components = self.read_fcomponents_until(component_closing, prefix)
        else:
            if has_debug and conversion is None:
                conversion = 'r'
            if not self.getc() == "}":
                raise ValueError("f-string: trailing junk in field")
        node = FComponent((node, *format_components), conversion)
        values.append(self.fill_node(node, start))
        return values

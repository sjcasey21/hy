# -*- encoding: utf-8 -*-
# Copyright 2021 the authors.
# This file is part of Hy, which is free software licensed under the Expat
# license. See the LICENSE.

import __future__

import ast
import copy
import importlib
import inspect
import sys
import textwrap
import traceback
import types
from itertools import dropwhile
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Final,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from funcparserlib.parser import NoParseError, many, maybe, oneplus, some, Parser

from hy._compat import PY3_8, reraise
from hy.errors import (
    HyCompileError,
    HyEvalError,
    HyInternalError,
    HyLanguageError,
    HySyntaxError,
    HyTypeError,
)
from hy.lex import mangle, unmangle
from hy.macros import macroexpand, require
from hy.model_patterns import (
    FORM,
    KEYWORD,
    STR,
    SYM,
    Tag,
    brackets,
    dolike,
    notpexpr,
    pexpr,
    sym,
    tag,
    times,
    unpack,
    whole,
)
from hy.models import (
    HyBytes,
    HyComplex,
    HyDict,
    HyExpression,
    HyFComponent,
    HyFloat,
    HyFString,
    HyInteger,
    HyKeyword,
    HyList,
    HyObject,
    HySequence,
    HySet,
    HyString,
    HySymbol,
    wrap_value,
)

# Constants ###################################################################
T = TypeVar("T")
Symbollike = TypeVar("Symbollike", str, HySymbol)
Argument = Tuple[Optional[HyObject], Union[HySymbol, HyList]]
Exprs = List[HyObject]
Varargs = Tuple[Optional[HyObject], HySymbol]

INF: Final = float("inf")
HY_AST_COMPILE_FLAGS: Final = (
    __future__.CO_FUTURE_DIVISION | __future__.CO_FUTURE_PRINT_FUNCTION  # type: ignore
)


# Helper Functions ############################################################
def ast_compile(a: ast.AST, filename: str, mode: str) -> types.CodeType:
    """Compile AST.

    Args:
      a: instance of `ast.AST`
      filename: Filename used for run-time error messages
      mode: `compile` mode parameter

    Returns:
      out: instance of `types.CodeType`
    """
    return compile(a, filename, mode, HY_AST_COMPILE_FLAGS)


def calling_module(n: int = 1) -> Optional[types.ModuleType]:
    """Get the module calling, if available.

    As a fallback, this will import a module using the calling frame's
    globals value of `__name__`.

    Args:
      n (optional): The number of levels up the stack from this function call.
          The default is one level up.

    Returns:
        out: The module at stack level `n + 1` or `None`.
    """
    frame_up = inspect.stack(0)[n + 1][0]
    module = inspect.getmodule(frame_up)
    if module is None:
        # This works for modules like `__main__`
        module_name = frame_up.f_globals.get("__name__", None)
        if module_name:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                pass
    return module


def is_unpack(kind: str, x: object) -> bool:
    return (
        isinstance(x, HyExpression)
        and len(x) > 0
        and isinstance(x[0], HySymbol)
        and x[0] == "unpack-" + kind
    )


def make_hy_model(outer: Callable[[Any], T], x: Iterable, rest: Optional[list]) -> T:
    def convert_to_hy_model(x):
        if isinstance(x, str):
            return HySymbol(x)
        elif isinstance(x, list):
            return x[0]
        else:
            return x

    return outer([convert_to_hy_model(a) for a in x] + (rest or []))


def mkexpr(*items, **kwargs) -> HyExpression:
    return make_hy_model(HyExpression, items, kwargs.get("rest"))


def mklist(*items, **kwargs) -> HyList:
    return make_hy_model(HyList, items, kwargs.get("rest"))


def pvalue(root, wanted) -> Parser:
    return pexpr(sym(root) + wanted) >> (lambda x: x[0])


def is_annotate_expression(model) -> bool:
    return bool(
        isinstance(model, HyExpression)
        and model
        and isinstance(model[0], HySymbol)
        and model[0] == HySymbol("annotate*")
    )


# Compiler Form Decorators ####################################################
_special_form_compilers: Dict[str, Tuple[Callable[..., "Result"], Parser]] = {}
_model_compilers: Dict[
    Type[HyObject], Callable[["HyASTCompiler", HyObject], "Result"]
] = {}
_decoratables = (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
_bad_roots = tuple(
    mangle(x) for x in ("unquote", "unquote-splice", "unpack-mapping", "except")
)
"""`_bad_roots` are fake special operators, which are used internally
by other special forms (e.g., `except` in `try`) but can't be
used to construct special forms themselves.
"""

SpecialFormName = Union[str, Tuple[bool, str]]


def special(
    names: Union[SpecialFormName, List[SpecialFormName]], patterns: List[Parser]
) -> Callable:
    """Declare special operators. The decorated method and the given pattern
    is assigned to _special_form_compilers for each of the listed names."""
    pattern = whole(patterns)

    def dec(fn: Callable):
        for name in names if isinstance(names, list) else [names]:
            if isinstance(name, tuple):
                condition, name = name
                if not condition:
                    continue
            _special_form_compilers[mangle(name)] = (fn, pattern)
        return fn
    return dec


def builds_model(*model_types: Type[HyObject]) -> Callable:
    "Assign the decorated method to _model_compilers for the given types."
    def _dec(fn):
        for t in model_types:
            _model_compilers[t] = fn
        return fn
    return _dec


AST_TYPE = TypeVar("AST_TYPE", bound=ast.AST)


def with_lineno(ast_type: Type[AST_TYPE]) -> Callable[..., AST_TYPE]:
    """Factory method for wrapping an ``ast.AST`` class with a ``HyObject``'s line numbers

    Equivalent to ``ast.Foo(..., lineno=x.start_line, col_offset=x.start_column) with
    ``ast.Foo(..., lineno=x.lineno, col_offset=x.col_offset)`` as a fallback if
    `start_line` or `start_column` is not defined for the ``HyObject`` `x`.

    Examples:
        ::

           => obj = HyString("Hello World!")
           => obj.start_line = 5
           => print(obj.start_line, obj.start_column)
           5 1
           => with_lineno(ast.Str)(obj, s=str(obj))
           ast.Str(value='Hello World!', lineno=5, col_offset=1)

    Args:
        ast_type: a class from python's ``ast`` module

    Returns:
        A function that returns an instance of the same type as ``ast_type``.
        Signature for the wrapper is ``(x: HyObject, **kwargs) -> AST_TYPE`` where
        ``kwargs`` is passed through the ``ast_type`` constructor.
    """
    def wrapped(x: HyObject, **kwargs):
        return ast_type(
            lineno=getattr(x, "start_line", getattr(x, "lineno", None)),
            col_offset=getattr(x, "start_column", getattr(x, "col_offset", None)),
            **kwargs
        )

    return wrapped


# Internal Compiler Classes ###################################################
class Result(object):
    """
    Smart representation of the result of a hy->AST compilation

    This object tries to reconcile the hy world, where everything can be used
    as an expression, with the Python world, where statements and expressions
    need to coexist.

    To do so, we represent a compiler result as a list of statements `stmts`,
    terminated by an expression context `expr`. The expression context is used
    when the compiler needs to use the result as an expression.

    Results are chained by addition: adding two results together returns a
    Result representing the succession of the two Results' statements, with
    the second Result's expression context.

    We make sure that a non-empty expression context does not get clobbered by
    adding more results, by checking accesses to the expression context. We
    assume that the context has been used, or deliberately ignored, if it has
    been accessed.

    The Result object is interoperable with python AST objects: when an AST
    object gets added to a Result object, it gets converted on-the-fly.
    """
    __slots__ = ("stmts", "temp_variables", "_expr", "__used_expr")

    def __init__(
        self,
        *,
        stmts: Optional[Sequence[Union[ast.stmt, ast.excepthandler]]] = None,
        expr: Optional[ast.expr] = None,
        temp_variables: Optional[
            Sequence[Union[ast.Name, ast.FunctionDef, ast.AsyncFunctionDef]]
        ] = None
    ):

        self.stmts: Sequence[Union[ast.stmt, ast.excepthandler]] = stmts or []
        self.temp_variables: Sequence[
            Union[ast.Name, ast.FunctionDef, ast.AsyncFunctionDef]
        ] = (temp_variables or [])
        self._expr: Optional[ast.expr] = expr

        self.__used_expr: bool = False

    @property
    def expr(self) -> Optional[ast.expr]:
        self.__used_expr = True
        return self._expr

    @expr.setter
    def expr(self, value: Optional[ast.expr]):
        self.__used_expr = False
        self._expr = value

    @property
    def lineno(self) -> Optional[int]:
        if self._expr is not None:
            return self._expr.lineno
        elif self.stmts:
            return self.stmts[-1].lineno
        else:
            return None

    @property
    def col_offset(self) -> Optional[int]:
        if self._expr is not None:
            return self._expr.col_offset
        elif self.stmts:
            return self.stmts[-1].col_offset
        else:
            return None

    def is_expr(self) -> bool:
        """Check whether I am a pure expression"""
        return bool(self._expr and not self.stmts)

    @property
    def force_expr(self) -> Union[ast.Constant, ast.expr]:
        """Force the expression context of the Result.

        If there is no expression context, we return a "None" expression.
        """
        if self.expr:
            return self.expr
        return ast.Constant(
            value=None,
            lineno=self.stmts[-1].lineno if self.stmts else 0,
            col_offset=self.stmts[-1].col_offset if self.stmts else 0)

    def expr_as_stmt(self) -> "Result":
        """Convert the Result's expression context to a statement

        This is useful when we want to use the stored expression in a
        statement context (for instance in a code branch).

        We drop ast.Names if they are appended to statements, as they
        can't have any side effect. "Bare" names still get converted to
        statements.

        If there is no expression context, return an empty result.
        """
        if self.expr and not (isinstance(self.expr, ast.Name) and self.stmts):
            return Result() + with_lineno(ast.Expr)(self.expr, value=self.expr)
        return Result()

    def rename(self, new_name: str):
        """Rename the Result's temporary variables to a `new_name`.

        We know how to handle ast.Names and ast.FunctionDefs.
        """
        new_name = mangle(new_name)
        for var in self.temp_variables:
            if isinstance(var, ast.Name):
                var.id = new_name
                # var.arg = new_name  # TODO: Check if can be removed
            elif isinstance(var, (ast.FunctionDef, ast.AsyncFunctionDef)):
                var.name = new_name
            else:
                raise TypeError("Don't know how to rename a %s!" % (
                    var.__class__.__name__))
        self.temp_variables = []

    def __add__(
        self, other: Union["Result", ast.stmt, ast.expr, ast.excepthandler, ast.AST]
    ) -> "Result":
        # If we add an ast statement, convert it first
        if isinstance(other, ast.stmt):
            return self + Result(stmts=[other])

        # If we add an ast expression, clobber the expression context
        if isinstance(other, ast.expr):
            return self + Result(expr=other)

        if isinstance(other, ast.excepthandler):
            return self + Result(stmts=[other])

        if not isinstance(other, Result):
            raise TypeError("Can't add %r with non-compiler result %r" % (
                self, other))

        # Check for expression context clobbering
        if self.expr and other.expr and not self.__used_expr:
            traceback.print_stack()
            print("Bad boy clobbered expr %s with %s" % (
                ast.dump(self.expr),
                ast.dump(other.expr)))

        # Fairly obvious addition
        result = Result()
        result.stmts = [*self.stmts, *other.stmts]
        result.expr = other.expr
        result.temp_variables = other.temp_variables

        return result

    def __repr__(self):
        return "Result(stmts=[%s], expr=%s)" % (
            ", ".join(ast.dump(x) for x in self.stmts),
            ast.dump(self.expr) if self.expr else None,
        )

    def __str__(self):
        return "Result(stmts=[%s], expr=%s)" % (
            ", ".join(ast.dump(x) for x in self.stmts),
            ast.dump(self.expr) if self.expr else None,
        )


# Compiler Helper Variables ###################################################
OPTIONAL_ANNOTATION = maybe(pvalue("annotate*", FORM))
"""Parse an annotation setting."""


# Compiler ####################################################################
class HyASTCompiler(object):
    """A Hy-to-Python AST compiler"""

    def __init__(self, module: Union[str, types.ModuleType], filename: str = "<string>",
                 source: str = None):
        """
        Args:
            module: Module name or object in which the Hy tree is evaluated.
            filename (optional): The name of the file for the source to be compiled.
                This is optional information for informative error messages and
                debugging. Will default to ``"<string>"`` if unspecified.
            source (optional): The source for the file, if any, being compiled.
                This is optional information for informative error messages and
                debugging.
        """
        self.anon_var_count = 0
        self.temp_if = None

        if isinstance(module, types.ModuleType):
            self.module = module
        else:
            self.module = importlib.import_module(module)

        self.module_name = self.module.__name__

        self.filename = filename
        self.source = source

        # Hy expects this to be present, so we prep the module for Hy
        # compilation.
        self.module.__dict__.setdefault("__macros__", {})

    def get_anon_var(self) -> str:
        self.anon_var_count += 1
        return "_hy_anon_var_%s" % self.anon_var_count

    def compile_atom(self, atom: HyObject) -> Result:
        # Compilation methods may mutate the atom, so copy it first.
        atom = copy.copy(atom)
        return Result() + _model_compilers[type(atom)](self, atom)

    def compile(self, tree: Optional[HyObject]) -> Result:
        if tree is None:
            return Result()
        try:
            ret = self.compile_atom(tree)
            return ret
        except HyCompileError:
            # compile calls compile, so we're going to have multiple raise
            # nested; so let's re-raise this exception, let's not wrap it in
            # another HyCompileError!
            raise
        except HyLanguageError as e:
            # These are expected errors that should be passed to the user.
            reraise(type(e), e, sys.exc_info()[2])
        except Exception:
            # These are unexpected errors that will--hopefully--never be seen
            # by the user.
            f_exc = traceback.format_exc()
            exc_msg = "Internal Compiler Bug 😱\n⤷ {}".format(f_exc)
            reraise(HyCompileError, HyCompileError(exc_msg), sys.exc_info()[2])

    def _syntax_error(self, expr: Any, message: str) -> HySyntaxError:
        return HySyntaxError(message, expr, self.filename, self.source)

    def _compile_collect(
        self,
        exprs: Sequence[HyObject],
        with_kwargs: bool = False,
        dict_display: bool = False,
    ) -> Tuple[List[ast.expr], Result, List[ast.keyword]]:
        """Collect the expression contexts from a list of compiled expression.

        Args:
            exprs: list of ``HyObject``'s to compile
            with_kwargs: whether or not to compile found ``HyKeyword``'s or
                ``unpack-mapping``'s into ``ast.keyword`` nodes. Defaults to ``False``
            dict_display: whether or not we are compiling ``unpack-mapping``'s for use
                in ``ast.Dict`` nodes which expect a ``None`` to be in the associated
                keys position.
        Returns:
            Tuple of a list of compiled ast nodes for each expression in ``exprs``,
                ``Result`` instance representing sum of compiling each expression in
                ``exprs``, and a list of ``ast.keywords`` for arguments passed by
                keywords (is the empty list if ``with_kwargs`` is False)

        .. note::

           For more information about what ``with_kwargs`` and ``dict_display`` are
           used to achieve, see:

           - https://docs.python.org/3/library/ast.html#ast.Call
           - https://docs.python.org/3/library/ast.html#ast.Dict
        """
        compiled_exprs = []
        ret = Result()
        keywords = []

        exprs_iter = iter(exprs)
        for expr in exprs_iter:

            if is_unpack("mapping", expr):
                ret += self.compile(cast(HyExpression, expr)[1])
                if dict_display:
                    compiled_exprs.append(None)
                    compiled_exprs.append(ret.force_expr)
                elif with_kwargs:
                    keywords.append(with_lineno(ast.keyword)(
                        expr, arg=None, value=ret.force_expr))

            elif with_kwargs and isinstance(expr, HyKeyword):
                try:
                    value = next(exprs_iter)
                except StopIteration:
                    raise self._syntax_error(
                        expr, "Keyword argument {kw} needs a value.".format(kw=expr)
                    )

                if not expr:
                    raise self._syntax_error(
                        expr, "Can't call a function with the empty keyword"
                    )

                compiled_value = self.compile(value)
                ret += compiled_value

                arg = str(expr)[1:]
                keywords.append(
                    with_lineno(ast.keyword)(
                        expr, arg=mangle(arg), value=compiled_value.force_expr
                    )
                )

            else:
                ret += self.compile(expr)
                compiled_exprs.append(ret.force_expr)

        return compiled_exprs, ret, keywords

    def _compile_branch(self, exprs: List[HyObject]) -> Result:
        """Compile a sequence of exprs into a ``Result`` with last expr as the expr context

        Used to compile ``do`` statements, both explicit and implicit (i.e. the body
        of a function call, class defs, etc). Each expression in ``exprs`` is compiled
        in order into a ``Result`` where the last expression in the sequence is
        used as the return value (or expression context) of ``Result``.

        .. note::

           See the api documentation for ``do`` for an example of how an explicit
           do works in practice.
        """
        ret = Result()
        for x in map(self.compile, exprs[:-1]):
            ret += x
            ret += x.expr_as_stmt()
        if exprs:
            ret += self.compile(exprs[-1])
        return ret

    def _storeize(
        self,
        expr: HyObject,
        name: Union[Result, ast.expr],
        func: Optional[Union[Type[ast.Load], Type[ast.Store], Type[ast.Del]]] = None,
    ) -> Union[
        ast.Tuple, ast.List, ast.Name, ast.Subscript, ast.Attribute, ast.Starred
    ]:
        """Return a new `name` object with an ast.Store() context

        Args:
            expr: the expression tree ``name`` is drawn from. Used for syntax errors
            name: identifier to compile into an ``ast`` var reference. Tuples and lists
                represent iterable unpacking syntax with multiple ``ast`` var
                references's returned
            func: class constructor for ``ast.Name``'s ``ctx`` attribute.
                Either ``ast.Load``, ``ast.Del``, or ``ast.Store``. Will default
                to ``ast.Store`` if none is given.

        Returns:
            an ``ast`` variable reference (or sequence of them for unpacking
            assignments) with the appropriate ``ctx`` set

        Raises:
            HySyntaxError: if attempting to assign/delete an invalid ``name``

        .. note::

           See https://docs.python.org/3/library/ast.html#variables for more info
        """
        if not func:
            func = ast.Store

        if isinstance(name, Result):
            if not name.is_expr():
                raise self._syntax_error(
                    expr, "Can't assign or delete a non-expression"
                )
            # name.expr can't be none after checking is_expr
            name = cast(ast.expr, name.expr)

        if isinstance(name, (ast.Tuple, ast.List)):
            typ = type(name)
            new_elts = []
            for x in name.elts:
                new_elts.append(self._storeize(expr, x, func))
            new_name = typ(elts=new_elts)
        elif isinstance(name, ast.Name):
            new_name = ast.Name(id=name.id)
        elif isinstance(name, ast.Subscript):
            new_name = ast.Subscript(value=name.value, slice=name.slice)
        elif isinstance(name, ast.Attribute):
            new_name = ast.Attribute(value=name.value, attr=name.attr)
        elif isinstance(name, ast.Starred):
            new_name = ast.Starred(
                value=self._storeize(expr, name.value, func))
        else:
            raise self._syntax_error(expr, "Can't assign or delete a " + (
                "constant"
                if isinstance(name, ast.Constant)
                else type(expr).__name__))

        new_name.ctx = func()
        ast.copy_location(new_name, name)
        return new_name

    def _nonconst(self, name):
        """Ensure a ``name`` is not an already defined constant

        Raises:
            HySyntaxError: if ``name`` is a non literal constant

        Returns:
            the given ``name`` unchanged
        """
        if str(name) in ("None", "True", "False"):
            raise self._syntax_error(name, "Can't assign to constant")
        return name

    def _render_quoted_form(
        self, form: HyObject, level: int
    ) -> Tuple[HyObject, bool]:
        """
        Render a quoted form as a new HyExpression.

        `level` is the level of quasiquoting of the current form. We can
        unquote if level is 0.

        Returns a two-tuple (`expression`, `splice`).

        The `splice` return value is used to mark `unquote-splice`d forms.
        We need to distinguish them as want to concatenate them instead of
        just nesting them.
        """

        op = None
        if isinstance(form, HyExpression) and form and (
                isinstance(form[0], HySymbol)):
            op = unmangle(mangle(form[0]))
            if level == 0 and op in ("unquote", "unquote-splice"):
                if len(form) != 2:
                    raise HyTypeError(
                        "`%s' needs 1 argument, got %s" % op,
                        len(form) - 1,
                        self.filename,
                        form,
                        self.source,
                    )
                return form[1], op == "unquote-splice"
            elif op == "quasiquote":
                level += 1
            elif op in ("unquote", "unquote-splice"):
                level -= 1

        hytype = form.__class__
        name = ".".join((hytype.__module__, hytype.__name__))
        body: Sequence[Any] = [form]

        if isinstance(form, HySequence):
            contents = []
            for x in form:
                f_contents, splice = self._render_quoted_form(x, level)
                if splice:
                    contents.append(HyExpression([
                        HySymbol("list"),
                        HyExpression([HySymbol("or"), f_contents, HyList()])]))
                else:
                    contents.append(HyList([f_contents]))
            if form:
                # If there are arguments, they can be spliced
                # so we build a sum...
                body = [HyExpression([HySymbol("+"), HyList()] + contents)]
            else:
                body = [HyList()]

            if isinstance(form, HyFString) and form.brackets is not None:
                body.extend([HyKeyword("brackets"), form.brackets])
            elif isinstance(form, HyFComponent) and form.conversion is not None:
                body.extend([HyKeyword("conversion"), HyString(form.conversion)])

        elif isinstance(form, HySymbol):
            body = [HyString(form)]

        elif isinstance(form, HyKeyword):
            body = [HyString(form.name)]

        elif isinstance(form, HyString):
            if form.brackets is not None:
                body.extend([HyKeyword("brackets"), form.brackets])

        ret = HyExpression([HySymbol(name)] + body).replace(form)
        return ret, False

    @special(["quote", "quasiquote"], [FORM])
    def compile_quote(self, expr: HyExpression, root: str, arg: HyObject) -> Result:
        level = INF if root == "quote" else 0   # Only quasiquotes can unquote
        stmts, _ = self._render_quoted_form(arg, level)
        ret = self.compile(stmts)
        return ret

    @special("unpack-iterable", [FORM])
    def compile_unpack_iterable(
        self, expr: HyExpression, root: str, arg: HyObject
    ) -> Result:
        ret = self.compile(arg)
        ret += with_lineno(ast.Starred)(expr, value=ret.force_expr, ctx=ast.Load())
        return ret

    @special("do", [many(FORM)])
    def compile_do(self, expr: HyExpression, root: str, body: List[HyObject]) -> Result:
        return self._compile_branch(body)

    @special("raise", [maybe(FORM), maybe(sym(":from") + FORM)])
    def compile_raise_expression(
        self,
        expr: HyExpression,
        root: str,
        exc: Optional[HyObject],
        cause: Optional[HyObject],
    ) -> Result:
        ret = Result()
        exc_ast = None
        cause_ast = None

        if exc is not None:
            exc_ret = self.compile(exc)
            ret += exc_ret
            exc_ast = exc_ret.force_expr

        if cause is not None:
            cause_ret = self.compile(cause)
            ret += cause_ret
            cause_ast = cause_ret.force_expr

        return ret + with_lineno(ast.Raise)(
            expr, type=ret.expr, exc=exc_ast,
            inst=None, tback=None, cause=cause_ast)

    @special(
        "try",
        [
            many(notpexpr("except", "else", "finally")),
            many(
                pexpr(
                    sym("except"),
                    brackets() | brackets(FORM) | brackets(SYM, FORM),
                    many(FORM),
                )
            ),
            maybe(dolike("else")),
            maybe(dolike("finally")),
        ],
    )
    def compile_try_expression(
        self,
        expr: HyExpression,
        root: str,
        body: List[HyObject],
        catchers: List[HyExpression],
        orelse: List[HyObject],
        finalbody: List[HyObject],
    ) -> Result:
        body_ret: Result = self._compile_branch(body)

        return_var = with_lineno(ast.Name)(
            expr, id=mangle(self.get_anon_var()), ctx=ast.Store())

        handler_results = Result()
        handlers_ast = []
        for catcher in catchers:
            handler_results += self._compile_catch_expression(
                catcher, return_var, *catcher)
            *handler_results.stmts, last_catch_stmt = handler_results.stmts
            handlers_ast.append(last_catch_stmt)

        if orelse is None:
            orelse_ast = []
        else:
            orelse_ret = self._compile_branch(orelse)
            orelse_ret += with_lineno(ast.Assign)(expr, targets=[return_var],
                                                  value=orelse_ret.force_expr)
            orelse_ret += orelse_ret.expr_as_stmt()
            orelse_ast = orelse_ret.stmts

        if finalbody is None:
            finalbody_ast = []
        else:
            finalbody_ret = self._compile_branch(finalbody)
            finalbody_ret += finalbody_ret.expr_as_stmt()
            finalbody_ast = finalbody_ret.stmts

        # Using (else) without (except) is verboten!
        if orelse_ast and not handlers_ast:
            raise self._syntax_error(expr, "`try' cannot have `else' without `except'")
        # Likewise a bare (try) or (try BODY).
        if not (handlers_ast or finalbody_ast):
            raise self._syntax_error(
                expr, "`try' must have an `except' or `finally' clause"
            )

        returnable = Result(
            expr=with_lineno(ast.Name)(expr, id=return_var.id, ctx=ast.Load()),
            temp_variables=[return_var])
        body_ret += body_ret.expr_as_stmt() if orelse_ast else with_lineno(ast.Assign)(
            expr, targets=[return_var], value=body_ret.force_expr)
        body_ast = body_ret.stmts or [with_lineno(ast.Pass)(expr)]

        x = with_lineno(ast.Try)(
            expr,
            body=body_ast,
            handlers=handlers_ast,
            orelse=orelse_ast,
            finalbody=finalbody_ast)
        return handler_results + x + returnable

    def _compile_catch_expression(self, expr, var, exceptions, body):
        # exceptions catch should be either:
        # [[list of exceptions]]
        # or
        # [variable [list of exceptions]]
        # or
        # [variable exception]
        # or
        # [exception]
        # or
        # []

        name = None
        if len(exceptions) == 2:
            name = mangle(self._nonconst(exceptions[0]))

        exceptions_list = exceptions[-1] if exceptions else HyList()
        if isinstance(exceptions_list, HyList):
            if len(exceptions_list):
                # [FooBar BarFoo] → catch Foobar and BarFoo exceptions
                elements, types, _ = self._compile_collect(exceptions_list)
                types += with_lineno(ast.Tuple)(
                    exceptions_list, elts=elements, ctx=ast.Load()
                )
            else:
                # [] → all exceptions caught
                types = Result()
        else:
            types = self.compile(exceptions_list)

        body = self._compile_branch(body)
        body += with_lineno(ast.Assign)(expr, targets=[var], value=body.force_expr)
        body += body.expr_as_stmt()

        return types + with_lineno(ast.ExceptHandler)(
            expr, type=types.expr, name=name,
            body=body.stmts or [with_lineno(ast.Pass)(expr)])

    @special("if*", [FORM, FORM, maybe(FORM)])
    def compile_if(
        self,
        expr: HyExpression,
        _: str,
        cond_expr: HyObject,
        body_expr: HyObject,
        orel_expr: Optional[HyObject],
    ) -> Result:
        cond = self.compile(cond_expr)
        body = self.compile(body_expr)

        nested = root = False
        orel = Result()
        if orel_expr is not None:
            if isinstance(orel_expr, HyExpression) and isinstance(orel_expr[0],
               HySymbol) and orel_expr[0] == "if*":
                # Nested ifs: don't waste temporaries
                root = self.temp_if is None
                nested = True
                self.temp_if = self.temp_if or self.get_anon_var()
            orel = self.compile(orel_expr)

        if not cond.stmts and isinstance(cond.force_expr, ast.Name):
            name = cond.force_expr.id
            branch = None
            if name == "True":
                branch = body
            elif name in ("False", "None"):
                branch = orel
            if branch is not None:
                if self.temp_if and branch.stmts:
                    name = with_lineno(ast.Name)(expr,
                                                 id=mangle(self.temp_if),
                                                 ctx=ast.Store())

                    branch += with_lineno(ast.Assign)(expr,
                                                      targets=[name],
                                                      value=body.force_expr)

                return branch

        # We want to hoist the statements from the condition
        ret = cond

        if body.stmts or orel.stmts:
            # We have statements in our bodies
            # Get a temporary variable for the result storage
            var = self.temp_if or self.get_anon_var()
            name = with_lineno(ast.Name)(expr,
                                         id=mangle(var),
                                         ctx=ast.Store())

            # Store the result of the body
            body += with_lineno(ast.Assign)(expr,
                                            targets=[name],
                                            value=body.force_expr)

            # and of the else clause
            if not nested or not orel.stmts or (not root and
               var != self.temp_if):
                orel += with_lineno(ast.Assign)(
                    expr, targets=[name], value=orel.force_expr
                )

            # Then build the if
            ret += with_lineno(ast.If)(expr,
                                       test=ret.force_expr,
                                       body=body.stmts,
                                       orelse=orel.stmts)

            # And make our expression context our temp variable
            expr_name = with_lineno(ast.Name)(expr, id=mangle(var), ctx=ast.Load())

            ret += Result(expr=expr_name, temp_variables=[expr_name, name])
        else:
            # Just make that an if expression
            ret += with_lineno(ast.IfExp)(
                expr, test=ret.force_expr, body=body.force_expr, orelse=orel.force_expr
            )

        if root:
            self.temp_if = None

        return ret

    @special(["break", "continue"], [])
    def compile_break_or_continue_expression(
        self, expr: HyExpression, root: str
    ) -> Result:
        node = with_lineno(ast.Break) if root == "break" else with_lineno(ast.Continue)
        return Result(stmts=[node(expr)])

    @special("assert", [FORM, maybe(FORM)])
    def compile_assert_expression(
        self, expr: HyExpression, root: str, test: HyObject, msg: Optional[HyObject]
    ) -> Result:
        if msg is None or type(msg) is HySymbol:
            ret = self.compile(test)
            return ret + with_lineno(ast.Assert)(
                expr,
                test=ret.force_expr,
                msg=(None if msg is None else self.compile(msg).force_expr))

        # The `msg` part may involve statements, which we only
        # want to be executed if the assertion fails. Rewrite the
        # form to set `msg` to a variable.
        msg_var = self.get_anon_var()
        return self.compile(
            mkexpr(
                "if*",
                mkexpr("and", "__debug__", mkexpr("not", [test])),
                mkexpr(
                    "do",
                    mkexpr("setv", msg_var, [msg]),
                    mkexpr("assert", "False", msg_var),
                ),
            ).replace(expr)
        )

    @special(["global", "nonlocal"], [oneplus(SYM)])
    def compile_global_or_nonlocal(
        self, expr: HyExpression, root: str, syms: List[HyObject]
    ) -> Result:
        node = with_lineno(ast.Global) if root == "global" else with_lineno(ast.Nonlocal)
        return Result(stmts=[node(expr, names=list(map(mangle, syms)))])

    @special("yield", [maybe(FORM)])
    def compile_yield_expression(
        self, expr: HyExpression, root: str, arg: Optional[HyObject]
    ) -> Result:
        ret = Result()
        if arg is not None:
            ret += self.compile(arg)
        return ret + with_lineno(ast.Yield)(expr, value=ret.force_expr)

    @special(["yield-from", "await"], [FORM])
    def compile_yield_from_or_await_expression(
        self, expr: HyExpression, root: str, arg: HyObject
    ) -> Result:
        ret = Result() + self.compile(arg)
        node = (
            with_lineno(ast.YieldFrom)
            if root == "yield-from"
            else with_lineno(ast.Await)
        )
        return ret + node(expr, value=ret.force_expr)

    @special("get", [FORM, oneplus(FORM)])
    def compile_index_expression(
        self,
        expr: HyExpression,
        name: str,
        obj: HyObject,
        indices_exprs: List[HyObject],
    ) -> Result:
        indices, ret, _ = self._compile_collect(indices_exprs)
        ret += self.compile(obj)

        for ix in indices:
            ret += with_lineno(ast.Subscript)(
                expr,
                value=ret.force_expr,
                slice=ast.Index(value=ix),
                ctx=ast.Load())

        return ret

    @special(".", [FORM, many(SYM | brackets(FORM))])
    def compile_attribute_access(
        self,
        expr: HyExpression,
        name: str,
        invocant: HyObject,
        keys: List[Union[HySymbol, HyList]],
    ) -> Result:
        ret = self.compile(invocant)

        for attr in keys:
            if isinstance(attr, HySymbol):
                ret += with_lineno(ast.Attribute)(attr,
                                                  value=ret.force_expr,
                                                  attr=mangle(attr),
                                                  ctx=ast.Load())
            else:  # attr is a HyList
                compiled_attr = self.compile(attr[0])
                ret = compiled_attr + ret + with_lineno(ast.Subscript)(
                    attr,
                    value=ret.force_expr,
                    slice=ast.Index(value=compiled_attr.force_expr),
                    ctx=ast.Load())

        return ret

    @special("del", [many(FORM)])
    def compile_del_expression(
        self, expr: HyExpression, name: str, args: List[HyObject]
    ) -> Result:
        if not args:
            return Result(stmts=[with_lineno(ast.Pass)(expr)])

        del_targets = []
        ret = Result()
        for target in args:
            compiled_target = self.compile(target)
            ret += compiled_target
            del_targets.append(self._storeize(target, compiled_target,
                                              ast.Del))

        return ret + with_lineno(ast.Delete)(expr, targets=del_targets)

    @special("cut", [FORM, maybe(FORM), maybe(FORM), maybe(FORM)])
    def compile_cut_expression(
        self,
        expr: HyExpression,
        name: str,
        obj: HyObject,
        lower: Optional[HyObject],
        upper: Optional[HyObject],
        step: Optional[HyObject],
    ) -> Result:
        ret = [Result()]

        def c(e):
            ret[0] += self.compile(e)
            return ret[0].force_expr

        s = with_lineno(ast.Subscript)(
            expr,
            value=c(obj),
            slice=with_lineno(ast.Slice)(
                expr, lower=c(lower), upper=c(upper), step=c(step)
            ),
            ctx=ast.Load(),
        )
        return ret[0] + s

    @special("with-decorator", [oneplus(FORM)])
    def compile_decorate_expression(
        self, expr: HyExpression, name: str, args: List[HyObject]
    ) -> Result:
        decs, fn = args[:-1], self.compile(args[-1])
        decorated_fn = fn.stmts[-1] if fn.stmts else None
        if decorated_fn is None or not isinstance(decorated_fn, _decoratables):
            raise self._syntax_error(args[-1], "Decorated a non-function")
        decs, ret, _ = self._compile_collect(decs)
        decorated_fn.decorator_list = decs + decorated_fn.decorator_list
        return ret + fn

    @special(["with*", "with/a*"],
             [brackets(FORM, maybe(FORM)), many(FORM)])
    def compile_with_expression(
        self, expr: HyExpression, root: str, args, body
    ) -> Result:
        thing, ctx = (None, args[0]) if args[1] is None else args
        if thing is not None:
            thing = self._storeize(thing, self.compile(thing))
        ctx = self.compile(ctx)

        body = self._compile_branch(body)

        # Store the result of the body in a tempvar
        var = self.get_anon_var()
        name = with_lineno(ast.Name)(expr, id=mangle(var), ctx=ast.Store())
        body += with_lineno(ast.Assign)(expr, targets=[name], value=body.force_expr)
        # Initialize the tempvar to None in case the `with` exits
        # early with an exception.
        initial_assign = with_lineno(ast.Assign)(
            expr, targets=[name], value=with_lineno(ast.Constant)(expr, value=None))

        node = with_lineno(ast.With) if root == "with*" else with_lineno(ast.AsyncWith)
        the_with = node(expr,
                        context_expr=ctx.force_expr,
                        optional_vars=thing,
                        body=body.stmts,
                        items=[ast.withitem(context_expr=ctx.force_expr,
                                            optional_vars=thing)])

        ret = Result(stmts=[initial_assign]) + ctx + the_with
        # And make our expression context our temp variable
        expr_name = with_lineno(ast.Name)(expr, id=mangle(var), ctx=ast.Load())

        ret += Result(expr=expr_name)
        # We don't give the Result any temp_vars because we don't want
        # Result.rename to touch `name`. Otherwise, initial_assign will
        # clobber any preexisting value of the renamed-to variable.

        return ret

    @special(",", [many(FORM)])
    def compile_tuple(
        self, expr: HyExpression, root: str, args: List[HyObject]
    ) -> Result:
        elements, ret, _ = self._compile_collect(args)
        return ret + with_lineno(ast.Tuple)(expr, elts=elements, ctx=ast.Load())

    _loopers = many(
        tag("setv", sym(":setv") + FORM + FORM) |
        tag("if", sym(":if") + FORM) |
        tag("do", sym(":do") + FORM) |
        tag("afor", sym(":async") + FORM + FORM) |
        tag("for", FORM + FORM))

    @special("for", [brackets(_loopers), many(notpexpr("else")), maybe(dolike("else"))])
    def compile_for(
        self,
        expr: HyExpression,
        root: str,
        bracketed_parts: HyList,
        body_exprs: List[HyObject],
        else_expr: Optional[List[HyObject]],
    ) -> Result:
        orel = []
        if else_expr is not None:
            orel.append(self._compile_branch(else_expr))
            orel[0] += orel[0].expr_as_stmt()

        parts: List[Tag] = bracketed_parts[0]

        if not parts:
            return Result(expr=with_lineno(ast.Constant)(expr, value=None))

        parts = [
            Tag(p.tag, self.compile(p.value) if p.tag in ["if", "do"] else [
                self._storeize(p.value[0], self.compile(p.value[0])),
                self.compile(p.value[1])])
            for p in parts]

        def f(parts: List[Tag]) -> Result:
            # This function is called recursively to construct
            # the nested loop.
            if not parts:
                if body_exprs:
                    body = self._compile_branch(body_exprs)
                    return body + body.expr_as_stmt()
                return Result(stmts=[with_lineno(ast.Pass)(expr)])
            else:
                (tagname, v), parts = parts[0], parts[1:]
                if tagname in ("for", "afor"):
                    orelse = orel and orel.pop().stmts
                    node = (
                        with_lineno(ast.AsyncFor)
                        if tagname == "afor"
                        else with_lineno(ast.For)
                    )
                    return v[1] + node(
                        v[1], target=v[0], iter=v[1].force_expr, body=f(parts).stmts,
                        orelse=orelse)
                elif tagname == "setv":
                    return v[1] + with_lineno(ast.Assign)(
                        v[1], targets=[v[0]], value=v[1].force_expr) + f(parts)
                elif tagname == "if":
                    return v + with_lineno(ast.If)(
                        v, test=v.force_expr, body=f(parts).stmts, orelse=[])
                elif tagname == "do":
                    return v + v.expr_as_stmt() + f(parts)
                else:
                    raise ValueError("can't happen")
        return f(parts)

    @special("dfor", [_loopers, brackets(FORM, FORM)])
    def compile_dict_comprehension(
        self,
        expr: HyExpression,
        root: str,
        parts: List[Tag],
        final: Tuple[HyObject, HyObject],
    ) -> Result:
        key, element = map(self.compile, final)

        # Compile the parts.
        if not parts:
            return Result(expr=with_lineno(ast.Dict)(expr, keys=[], values=[]))

        orel = []
        parts = [
            Tag(p.tag, self.compile(p.value) if p.tag in ["if", "do"] else [
                self._storeize(p.value[0], self.compile(p.value[0])),
                self.compile(p.value[1])])
            for p in parts]

        # Produce a result.
        should_lift_into_function = (
            element.stmts
            or (key is not None and key.stmts)
            or any(
                p.tag == "do"
                or (
                    p.value[1].stmts
                    if p.tag in ("for", "afor", "setv")
                    else p.value.stmts
                )
                for p in parts
            )
        )
        if should_lift_into_function:
            # The desired comprehension can't be expressed as a
            # real Python comprehension. We'll write it as a nested
            # loop in a function instead.
            def f(parts):
                # This function is called recursively to construct
                # the nested loop.
                if not parts:
                    ret = key + element
                    val = with_lineno(ast.Tuple)(
                        key, ctx=ast.Load(), elts=[key.force_expr, element.force_expr]
                    )
                    return ret + with_lineno(ast.Expr)(
                        element, value=with_lineno(ast.Yield)(element, value=val))

                (tagname, v), parts = parts[0], parts[1:]
                if tagname in ("for", "afor"):
                    orelse = orel and orel.pop().stmts
                    node = (
                        with_lineno(ast.AsyncFor)
                        if tagname == "afor"
                        else with_lineno(ast.For)
                    )
                    return v[1] + node(
                        v[1], target=v[0], iter=v[1].force_expr, body=f(parts).stmts,
                        orelse=orelse)
                elif tagname == "setv":
                    return v[1] + with_lineno(ast.Assign)(
                        v[1], targets=[v[0]], value=v[1].force_expr) + f(parts)
                elif tagname == "if":
                    return v + with_lineno(ast.If)(
                        v, test=v.force_expr, body=f(parts).stmts, orelse=[])
                elif tagname == "do":
                    return v + v.expr_as_stmt() + f(parts)
                else:
                    raise ValueError("can't happen")
            fname = self.get_anon_var()
            # Define the generator function.
            ret = Result() + with_lineno(ast.FunctionDef)(
                expr,
                name=fname,
                args=ast.arguments(
                    args=[], vararg=None, kwarg=None, posonlyargs=[],
                    kwonlyargs=[], kw_defaults=[], defaults=[]),
                body=f(parts).stmts,
                decorator_list=[])
            # Immediately call the new function. Unless the user asked
            # for a generator, wrap the call in `[].__class__(...)` or
            # `{}.__class__(...)` or `{1}.__class__(...)` to get the
            # right type. We don't want to just use e.g. `list(...)`
            # because the name `list` might be rebound.
            return ret + Result(
                expr=cast(
                    ast.Expr,
                    ast.parse("{}({}())".format("{}.__class__", fname)).body[0],
                ).value
            )

        # We can produce a real comprehension.
        generators = []
        for tagname, v in parts:
            if tagname in ("for", "afor"):
                generators.append(ast.comprehension(
                    target=v[0], iter=v[1].expr, ifs=[],
                    is_async=int(tagname == "afor")))
            elif tagname == "setv":
                generators.append(ast.comprehension(
                    target=v[0],
                    iter=with_lineno(ast.Tuple)(v[1], elts=[v[1].expr], ctx=ast.Load()),
                    ifs=[], is_async=0))
            elif tagname == "if":
                generators[-1].ifs.append(v.expr)
            else:
                raise ValueError("can't happen")
        return Result(
            expr=with_lineno(ast.DictComp)(
                expr, key=key.expr, value=element.expr, generators=generators
            )
        )

    @special(["lfor", "sfor", "gfor"], [_loopers, FORM])
    def compile_comprehension(
        self, expr: HyExpression, root: str, parts: List[Tag], final: HyObject
    ) -> Result:
        node_class = {
            "lfor": ast.ListComp,
            "sfor": ast.SetComp,
            "gfor": ast.GeneratorExp}[root]

        # Get the final value
        element = self.compile(final)

        orel = []
        # Compile the parts.
        if not parts:
            return Result(expr=cast(ast.Expr, ast.parse({
                ast.ListComp: "[]",
                ast.SetComp: "{1}.__class__()",
                ast.GeneratorExp: "(_ for _ in [])"}[node_class]).body[0]).value)
        parts = [
            Tag(p.tag, self.compile(p.value) if p.tag in ["if", "do"] else [
                self._storeize(p.value[0], self.compile(p.value[0])),
                self.compile(p.value[1])])
            for p in parts]

        # Produce a result.
        should_lift_into_function = (
            element.stmts
            or any(
                p.tag == "do"
                or (
                    p.value[1].stmts
                    if p.tag in ("for", "afor", "setv")
                    else p.value.stmts
                )
                for p in parts
            )
        )
        if should_lift_into_function:
            # The desired comprehension can't be expressed as a
            # real Python comprehension. We'll write it as a nested
            # loop in a function instead.
            def f(parts):
                # This function is called recursively to construct
                # the nested loop.
                if not parts:
                    ret = element
                    val = element.force_expr
                    return ret + with_lineno(ast.Expr)(
                        element, value=with_lineno(ast.Yield)(element, value=val))
                (tagname, v), parts = parts[0], parts[1:]
                if tagname in ("for", "afor"):
                    orelse = orel and orel.pop().stmts
                    node = (
                        with_lineno(ast.AsyncFor)
                        if tagname == "afor"
                        else with_lineno(ast.For)
                    )
                    return v[1] + node(
                        v[1], target=v[0], iter=v[1].force_expr, body=f(parts).stmts,
                        orelse=orelse)
                elif tagname == "setv":
                    return v[1] + with_lineno(ast.Assign)(
                        v[1], targets=[v[0]], value=v[1].force_expr) + f(parts)
                elif tagname == "if":
                    return v + with_lineno(ast.If)(
                        v, test=v.force_expr, body=f(parts).stmts, orelse=[])
                elif tagname == "do":
                    return v + v.expr_as_stmt() + f(parts)
                else:
                    raise ValueError("can't happen")
            fname = self.get_anon_var()
            # Define the generator function.
            ret = Result() + with_lineno(ast.FunctionDef)(
                expr,
                name=fname,
                args=ast.arguments(
                    args=[], vararg=None, kwarg=None, posonlyargs=[],
                    kwonlyargs=[], kw_defaults=[], defaults=[]),
                body=f(parts).stmts,
                decorator_list=[])
            # Immediately call the new function. Unless the user asked
            # for a generator, wrap the call in `[].__class__(...)` or
            # `{}.__class__(...)` or `{1}.__class__(...)` to get the
            # right type. We don't want to just use e.g. `list(...)`
            # because the name `list` might be rebound.
            return ret + Result(expr=cast(ast.Expr, ast.parse(
                "{}({}())".format(
                    {ast.ListComp: "[].__class__",
                     ast.DictComp: "{}.__class__",
                     ast.SetComp: "{1}.__class__",
                     ast.GeneratorExp: ""}[node_class],
                    fname)).body[0]).value)

        # We can produce a real comprehension.
        generators = []
        for tagname, v in parts:
            if tagname in ("for", "afor"):
                generators.append(ast.comprehension(
                    target=v[0], iter=v[1].expr, ifs=[],
                    is_async=int(tagname == "afor")))
            elif tagname == "setv":
                generators.append(ast.comprehension(
                    target=v[0],
                    iter=with_lineno(ast.Tuple)(v[1], elts=[v[1].expr], ctx=ast.Load()),
                    ifs=[], is_async=0))
            elif tagname == "if":
                generators[-1].ifs.append(v.expr)
            else:
                raise ValueError("can't happen")
        return with_lineno(node_class)(expr, elt=element.expr, generators=generators)

    @special(["not", "~"], [FORM])
    def compile_unary_operator(
        self, expr: HyExpression, root: str, arg: HyObject
    ) -> Result:
        ops = {"not": ast.Not,
               "~": ast.Invert}
        operand = self.compile(arg)
        return operand + with_lineno(ast.UnaryOp)(
            expr, op=ops[root](), operand=operand.force_expr)

    def _importlike(*name_types: Type[HyObject]) -> List[Parser]:
        name = some(lambda x: isinstance(x, name_types) and "." not in x)
        return [many(
            SYM |
            brackets(SYM, sym(":as"), name) |
            brackets(SYM, brackets(many(
                name + maybe(sym(":as") + name)))))]

    @special("import", _importlike(HySymbol))
    @special("require", _importlike(HySymbol, HyString))
    def compile_import_or_require(
        self, expr: HyExpression, root: str, entries: List[Union[HySymbol, HyList]]
    ) -> Result:
        ret = Result()

        for entry in entries:
            assignments = "ALL"
            prefix = ""

            if isinstance(entry, HySymbol):
                # e.g., (import foo)
                module, prefix = entry, entry
            elif isinstance(entry, HyList) and isinstance(entry[1], HySymbol):
                # e.g., (import [foo :as bar])
                module, prefix = entry
            else:
                # e.g., (import [foo [bar baz :as MyBaz bing]])
                # or (import [foo [*]])
                module, kids = entry
                kids = kids[0]
                if (HySymbol("*"), None) in kids:
                    if len(kids) != 1:
                        star = kids[kids.index((HySymbol("*"), None))][0]
                        raise self._syntax_error(
                            star, "* in an import name list must be on its own"
                        )
                else:
                    assignments = [(k, v or k) for k, v in kids]

            ast_module = mangle(module)

            if root == "import":
                module = ast_module.lstrip(".")
                level = len(ast_module) - len(module)
                if assignments == "ALL" and prefix == "":
                    node = with_lineno(ast.ImportFrom)
                    names = [ast.alias(name="*", asname=None)]
                elif assignments == "ALL":
                    node = with_lineno(ast.Import)
                    prefix = mangle(prefix)
                    names = [ast.alias(
                        name=ast_module,
                        asname=prefix if prefix != module else None)]
                else:
                    node = with_lineno(ast.ImportFrom)
                    names = [
                        ast.alias(
                            name=mangle(k),
                            asname=None if v == k else mangle(v))
                        for k, v in assignments]
                ret += node(
                    expr, module=module or None, names=names, level=level)

            elif require(ast_module, self.module, assignments=assignments,
                         prefix=prefix):
                # Actually calling `require` is necessary for macro expansions
                # occurring during compilation.
                # The `require` we're creating in AST is the same as above, but used at
                # run-time (e.g. when modules are loaded via bytecode).
                ret += self.compile(HyExpression([
                    HySymbol("hy.macros.require"),
                    HyString(ast_module),
                    HySymbol("None"),
                    HyKeyword("assignments"),
                    (HyString("ALL") if assignments == "ALL" else
                        [[HyString(k), HyString(v)] for k, v in assignments]),
                    HyKeyword("prefix"),
                    HyString(prefix)]).replace(expr))
                ret += ret.expr_as_stmt()

        return ret

    @special(["and", "or"], [many(FORM)])
    def compile_logical_or_and_and_operator(
        self, expr: HyExpression, operator: str, args: List[HyObject]
    ) -> Result:
        ops = {"and": (ast.And, True),
               "or": (ast.Or, None)}
        opnode, default = ops[operator]
        osym = expr[0]
        if len(args) == 0:
            return Result(expr=with_lineno(ast.Constant)(osym, value=default))
        elif len(args) == 1:
            return self.compile(args[0])
        ret = Result()
        values = list(map(self.compile, args))
        if any(value.stmts for value in values):
            # Compile it to an if...else sequence
            var = self.get_anon_var()
            name = with_lineno(ast.Name)(osym, id=var, ctx=ast.Store())
            expr_name = with_lineno(ast.Name)(osym, id=var, ctx=ast.Load())
            temp_variables = [name, expr_name]

            def make_assign(value, node=None):
                positioned_name = with_lineno(ast.Name)(
                    node or osym, id=var, ctx=ast.Store())
                temp_variables.append(positioned_name)
                return with_lineno(ast.Assign)(
                    node or osym, targets=[positioned_name], value=value)

            current = root = []
            for i, value in enumerate(values):
                if value.stmts:
                    node = value.stmts[0]
                    current.extend(value.stmts)
                else:
                    node = value.expr
                current.append(make_assign(value.force_expr, value.force_expr))
                if i == len(values)-1:
                    # Skip a redundant 'if'.
                    break
                if operator == "and":
                    cond = expr_name
                else:  # operator must be "or"
                    cond = with_lineno(ast.UnaryOp)(node, op=ast.Not(), operand=expr_name)
                current.append(with_lineno(ast.If)(node, test=cond, body=[], orelse=[]))
                current = current[-1].body
            ret = sum(root, ret)
            ret += Result(expr=expr_name, temp_variables=temp_variables)
        else:
            ret += with_lineno(ast.BoolOp)(osym,
                                           op=opnode(),
                                           values=[value.force_expr for value in values])
        return ret

    _c_ops = {
        "=": ast.Eq,
        "!=": ast.NotEq,
        "<": ast.Lt,
        "<=": ast.LtE,
        ">": ast.Gt,
        ">=": ast.GtE,
        "is": ast.Is,
        "is-not": ast.IsNot,
        "in": ast.In,
        "not-in": ast.NotIn,
    }
    _c_ops = {mangle(k): v for k, v in _c_ops.items()}

    def _get_c_op(self, sym) -> ast.AST:
        k = mangle(sym)
        if k not in self._c_ops:
            raise self._syntax_error(sym, "Illegal comparison operator: " + str(sym))
        return self._c_ops[k]()

    @special(["=", "is", "<", "<=", ">", ">="], [oneplus(FORM)])
    @special(["!=", "is-not", "in", "not-in"], [times(2, INF, FORM)])
    def compile_compare_op_expression(
        self, expr: HyExpression, root: str, args: List[HyObject]
    ) -> Result:
        if len(args) == 1:
            return (self.compile(args[0]) +
                    with_lineno(ast.Constant)(expr, value=True))

        ops = [self._get_c_op(root) for _ in args[1:]]
        exprs, ret, _ = self._compile_collect(args)
        return ret + with_lineno(ast.Compare)(
            expr, left=exprs[0], ops=ops, comparators=exprs[1:])

    @special("cmp", [FORM, many(SYM + FORM)])
    def compile_chained_comparison(
        self,
        expr: HyExpression,
        root: str,
        arg1_expr: HyObject,
        arg_exprs: List[Tuple[HySymbol, HyObject]],
    ) -> Result:
        ret = self.compile(arg1_expr)
        arg1 = ret.force_expr

        ops = [self._get_c_op(op) for op, _ in arg_exprs]
        args, ret2, _ = self._compile_collect(
            [x for _, x in arg_exprs])

        return (
            ret
            + ret2
            + with_lineno(ast.Compare)(expr, left=arg1, ops=ops, comparators=args)
        )

    # The second element of each tuple below is an aggregation operator
    # that's used for augmented assignment with three or more arguments.
    m_ops = {
        "+": (ast.Add, "+"),
        "/": (ast.Div, "*"),
        "//": (ast.FloorDiv, "*"),
        "*": (ast.Mult, "*"),
        "-": (ast.Sub, "+"),
        "%": (ast.Mod, None),
        "**": (ast.Pow, "**"),
        "<<": (ast.LShift, "+"),
        ">>": (ast.RShift, "+"),
        "|": (ast.BitOr, "|"),
        "^": (ast.BitXor, None),
        "&": (ast.BitAnd, "&"),
        "@": (ast.MatMult, "@"),
    }

    @special(["+", "*", "|"], [many(FORM)])
    @special(["-", "/", "&", "@"], [oneplus(FORM)])
    @special(["**", "//", "<<", ">>"], [times(2, INF, FORM)])
    @special(["%", "^"], [times(2, 2, FORM)])
    def compile_maths_expression(
        self, expr: HyExpression, root: str, args: List[HyObject]
    ) -> Result:
        if len(args) == 0:
            # Return the identity element for this operator.
            return Result(expr=with_lineno(ast.Num)(expr, n=(
                {"+": 0, "|": 0, "*": 1}[root])))

        if len(args) == 1:
            if root == "/":
                # Compute the reciprocal of the argument.
                args = [HyInteger(1).replace(expr), args[0]]
            elif root in ("+", "-"):
                # Apply unary plus or unary minus to the argument.
                op = {"+": ast.UAdd, "-": ast.USub}[root]()
                ret = self.compile(args[0])
                return ret + with_lineno(ast.UnaryOp)(expr, op=op, operand=ret.force_expr)
            else:
                # Return the argument unchanged.
                return self.compile(args[0])

        op = self.m_ops[root][0]
        right_associative = root == "**"
        ret = self.compile(args[-1 if right_associative else 0])
        for child in args[-2 if right_associative else 1 ::
                          -1 if right_associative else 1]:
            left_expr = ret.force_expr
            ret += self.compile(child)
            right_expr = ret.force_expr
            if right_associative:
                left_expr, right_expr = right_expr, left_expr
            ret += with_lineno(ast.BinOp)(expr, left=left_expr, op=op(), right=right_expr)

        return ret

    a_ops = {x + "=": v for x, v in m_ops.items()}

    @special([x for x, (_, v) in a_ops.items() if v is not None], [FORM, oneplus(FORM)])
    @special([x for x, (_, v) in a_ops.items() if v is None], [FORM, times(1, 1, FORM)])
    def compile_augassign_expression(
        self,
        expr: HyExpression,
        root: str,
        target_expr: HyObject,
        values: List[HyObject],
    ) -> Result:
        if len(values) > 1:
            return self.compile(
                mkexpr(
                    root, [target_expr], mkexpr(self.a_ops[root][1], rest=values)
                ).replace(expr)
            )

        op = self.a_ops[root][0]
        target = self._storeize(target_expr, self.compile(target_expr))
        ret = self.compile(values[0])
        return ret + with_lineno(ast.AugAssign)(
            expr, target=target, value=ret.force_expr, op=op())

    @special("setv", [many(OPTIONAL_ANNOTATION + FORM + FORM)])
    @special((PY3_8, "setx"), [times(1, 1, SYM + FORM)])
    def compile_def_expression(
        self,
        expr: HyExpression,
        root: str,
        decls: List[tuple],
    ) -> Result:
        if not decls:
            return Result(expr=with_lineno(ast.Constant)(expr, value=None))

        result = Result()
        is_assignment_expr = root == HySymbol("setx")
        ann: Optional[HyObject]
        name: HyObject
        value: HyObject
        for decl in decls:
            if is_assignment_expr:
                ann = None
                name, value = decl
            else:
                ann, name, value = decl

            result += self._compile_assign(ann, name, value,
                                           is_assignment_expr=is_assignment_expr)
        return result

    @special(["annotate*"], [FORM, FORM])
    def compile_basic_annotation(
        self, expr: HyExpression, root: str, ann: HyObject, target: HyObject
    ) -> Result:
        return self._compile_assign(ann, target, None)

    def _compile_assign(
        self,
        ann: Optional[HyObject],
        name: HyObject,
        value: Optional[HyObject],
        *,
        is_assignment_expr: bool = False
    ) -> Result:
        # Ensure that assignment expressions have a result and no annotation.
        assert not is_assignment_expr or (value is not None and ann is None)

        def annotated_assignment_fn():
            """return an ast.AST like function for an `AnnAssign`"""
            def node(x, **kwargs):
                return with_lineno(ast.AnnAssign)(
                    x,
                    annotation=ann_result.force_expr,
                    simple=int(isinstance(name, HySymbol)),
                    **kwargs
                )
            return node

        ld_name = self.compile(name)

        annotate_only = value is None
        if annotate_only:
            result = Result()
        else:
            result = self.compile(value)

        if (result.temp_variables
                and isinstance(name, HySymbol)
                and "." not in name):
            result.rename(self._nonconst(name))
            if not is_assignment_expr:
                # Throw away .expr to ensure that (setv ...) returns None.
                result.expr = None
        else:
            st_name = self._storeize(name, ld_name)

            if ann is not None:
                ann_result = self.compile(ann)
                result = ann_result + result

            if is_assignment_expr:
                node = with_lineno(ast.NamedExpr)
            elif ann is not None:
                node = annotated_assignment_fn()
            else:
                node = with_lineno(ast.Assign)

            result += node(
                name if hasattr(name, "start_line") else result,
                value=result.force_expr if not annotate_only else None,
                target=st_name, targets=[st_name])

        return result

    @special(["while"], [FORM, many(notpexpr("else")), maybe(dolike("else"))])
    def compile_while_expression(
        self,
        expr: HyExpression,
        root: str,
        cond: HyObject,
        body_exprs: List[HyObject],
        else_expr: Optional[List[HyObject]],
    ) -> Result:
        cond_compiled = self.compile(cond)

        body = self._compile_branch(body_exprs)
        body += body.expr_as_stmt()
        body_stmts = body.stmts or [with_lineno(ast.Pass)(expr)]

        if cond_compiled.stmts:
            # We need to ensure the statements for the condition are
            # executed on every iteration. Rewrite the loop to use a
            # single anonymous variable as the condition, i.e.:
            #  anon_var = True
            #  while anon_var:
            #    condition stmts...
            #    anon_var = condition expr
            #    if anon_var:
            #      while loop body
            cond_var = with_lineno(ast.Name)(cond, id=self.get_anon_var(), ctx=ast.Load())

            def make_not(operand):
                return with_lineno(ast.UnaryOp)(cond, op=ast.Not(), operand=operand)

            body_stmts = [
                *cond_compiled.stmts,
                with_lineno(ast.Assign)(
                    cond,
                    targets=[self._storeize(cond, cond_var)],
                    # Cast the condition to a bool in case it's mutable and
                    # changes its truth value, but use (not (not ...)) instead of
                    # `bool` in case `bool` has been redefined.
                    value=make_not(make_not(cond_compiled.force_expr)),
                ),
                with_lineno(ast.If)(cond, test=cond_var, body=body_stmts, orelse=[]),
            ]

            cond_compiled = (
                Result()
                + with_lineno(ast.Assign)(
                    cond,
                    targets=[self._storeize(cond, cond_var)],
                    value=with_lineno(ast.Constant)(cond, value=True),
                )
                + cond_var
            )

        orel = Result()
        if else_expr is not None:
            orel = self._compile_branch(else_expr)
            orel += orel.expr_as_stmt()

        ret = cond_compiled + with_lineno(ast.While)(
            expr, test=cond_compiled.force_expr,
            body=body_stmts,
            orelse=orel.stmts)

        return ret

    NASYM = some(
        lambda x: isinstance(x, HySymbol) and x not in (HySymbol("/"), HySymbol("*"))
    )
    argument = OPTIONAL_ANNOTATION + (NASYM | brackets(NASYM, FORM))
    varargs = lambda unpack_type, wanted: OPTIONAL_ANNOTATION + pvalue(  # noqa: E731
        unpack_type, wanted
    )
    kwonly_delim = some(lambda x: isinstance(x, HySymbol) and x == HySymbol("*"))

    @special(["fn", "fn*", "fn/a"], [
        # The starred version is for internal use (particularly, in the
        # definition of `defn`). It ensures that a FunctionDef is
        # produced rather than a Lambda.
        OPTIONAL_ANNOTATION,
        brackets(
            maybe(many(argument) + sym("/")),
            many(argument),
            maybe(kwonly_delim | varargs("unpack-iterable", NASYM)),
            many(argument),
            maybe(varargs("unpack-mapping", NASYM))),
        many(FORM)])
    def compile_function_def(
        self,
        expr: HyExpression,
        root: str,
        returns: Optional[HyObject],
        params: Tuple[
            Optional[List[Argument]],
            List[Argument],
            Optional[Varargs],
            List[Argument],
            Optional[Varargs],
        ],
        body_exprs: List[HyObject],
    ) -> Result:
        force_functiondef = root in ("fn*", "fn/a")
        node = (
            with_lineno(ast.AsyncFunctionDef)
            if root == "fn/a"
            else with_lineno(ast.FunctionDef)
        )
        ret = Result()

        # NOTE: Our evaluation order of return type annotations is
        # different from Python: Python evalautes them after the argument
        # annotations / defaults (as that's where they are in the source),
        # but Hy evaluates them *first*, since here they come before the #
        # argument list. Therefore, it would be more confusing for
        # readability to evaluate them after like Python.

        ret = Result()
        if returns is not None:
            returns_result = self.compile(returns)
            ret += returns_result
        else:
            returns_result = None

        posonly, args, rest, kwonly, kwargs = params

        if not (posonly or posonly is None):
            raise self._syntax_error(
                params, "positional only delimiter '/' must have an argument"
            )

        posonly = posonly or []

        def is_positional_arg(elem):
            return isinstance(elem[1], HySymbol)

        invalid_non_default = next(
            (arg
             for arg in dropwhile(is_positional_arg, posonly + args)
             if is_positional_arg(arg)),
            None
        )
        if invalid_non_default:
            raise self._syntax_error(
                invalid_non_default[1], "non-default argument follows default argument"
            )

        posonly_ast, posonly_defaults, ret = self._compile_arguments_set(posonly, ret)
        args_ast, args_defaults, ret = self._compile_arguments_set(args, ret)
        kwonly_ast, kwonly_defaults, ret = self._compile_arguments_set(kwonly, ret, True)
        rest_ast = kwargs_ast = None

        if rest == HySymbol("*"):  # rest is a positional only marker
            if not kwonly:
                raise self._syntax_error(rest, "named arguments must follow bare *")
            rest_ast = None
        elif rest:  # rest is capturing varargs
            [rest_ast], _, ret = self._compile_arguments_set([rest], ret)
        if kwargs:
            [kwargs_ast], _, ret = self._compile_arguments_set([kwargs], ret)

        args = ast.arguments(
            args=args_ast,
            defaults=[*posonly_defaults, *args_defaults],
            vararg=rest_ast,
            posonlyargs=posonly_ast,
            kwonlyargs=kwonly_ast,
            kw_defaults=kwonly_defaults,
            kwarg=kwargs_ast)

        body = self._compile_branch(body_exprs)

        if not force_functiondef and not body.stmts and returns is None:
            return ret + with_lineno(ast.Lambda)(expr, args=args, body=body.force_expr)

        if body.expr:
            body += with_lineno(ast.Return)(body.expr, value=body.expr)

        name = self.get_anon_var()

        def_ast_node = node(
            expr,
            name=name,
            args=args,
            body=body.stmts or [with_lineno(ast.Pass)(expr)],
            decorator_list=[],
            returns=returns_result.force_expr if returns is not None else None,
        )
        ret += def_ast_node

        ast_name = with_lineno(ast.Name)(expr, id=name, ctx=ast.Load())
        ret += Result(expr=ast_name, temp_variables=[ast_name, def_ast_node])
        return ret

    def _compile_arguments_set(
        self,
        decls: List[Argument],
        ret: Result,
        is_kwonly: bool = False,
    ) -> Tuple[List[ast.arg], List[Optional[ast.AST]], Result]:
        args_ast = []
        args_defaults = []

        for ann, decl in decls:
            default = None

            # funcparserlib will check to make sure that the only times we
            # ever have a HyList here are due to a default value.
            if isinstance(decl, HyList):
                sym, default = decl
            else:
                sym = decl

            if ann is not None:
                ret += self.compile(ann)
                ann_ast = ret.force_expr
            else:
                ann_ast = None

            if default is not None:
                ret += self.compile(default)
                args_defaults.append(ret.force_expr)
            # Kwonly args without defaults are considered required
            elif not isinstance(decl, HyList) and is_kwonly:
                args_defaults.append(None)
            elif isinstance(decl, HyList):
                # Note that the only time any None should ever appear here
                # is in kwargs, since the order of those with defaults vs
                # those without isn't significant in the same way as
                # positional args.
                args_defaults.append(None)

            args_ast.append(with_lineno(ast.arg)(
                sym, arg=mangle(self._nonconst(sym)), annotation=ann_ast))

        return args_ast, args_defaults, ret

    @special("return", [maybe(FORM)])
    def compile_return(
        self, expr: HyExpression, root: str, arg: Optional[HyObject]
    ) -> Result:
        ret = Result()
        if arg is None:
            return Result(stmts=[with_lineno(ast.Return)(expr, value=None)])
        ret += self.compile(arg)
        return ret + with_lineno(ast.Return)(expr, value=ret.force_expr)

    @special("defclass", [SYM, maybe(brackets(many(FORM)) + maybe(STR) + many(FORM))])
    def compile_class_expression(
        self,
        expr: HyExpression,
        root: str,
        name: Optional[str],
        rest: Optional[Tuple[List[List[HyObject]], Optional[HyString], List[HyObject]]],
    ) -> Result:
        base_list, docstring, body = rest or ([[]], None, [])

        bases_expr, bases, keywords = (
            self._compile_collect(base_list[0], with_kwargs=True))

        bodyr = Result()

        if docstring is not None:
            bodyr += self.compile(docstring).expr_as_stmt()

        for e in body:
            e = self.compile(self._rewire_init(
                macroexpand(e, self.module, self)))
            bodyr += e + e.expr_as_stmt()

        return bases + with_lineno(ast.ClassDef)(
            expr,
            decorator_list=[],
            name=mangle(self._nonconst(name)),
            keywords=keywords,
            starargs=None,
            kwargs=None,
            bases=bases_expr,
            body=bodyr.stmts or [with_lineno(ast.Pass)(expr)])

    def _rewire_init(self, expr: HyObject) -> HyObject:
        "Given a (setv …) form, append None to definitions of __init__."

        if not (isinstance(expr, HyExpression)
                and len(expr) > 1
                and isinstance(expr[0], HySymbol)
                and expr[0] == HySymbol("setv")):
            return expr

        new_args = []
        decls = list(expr[1:])
        while decls:
            if is_annotate_expression(decls[0]):
                # Handle annotations.
                ann = decls.pop(0)
            else:
                ann = None

            if len(decls) < 2:
                break
            k, v = (decls.pop(0), decls.pop(0))
            if (
                isinstance(k, HySymbol)
                and mangle(k) == "__init__"
                and isinstance(v, HyExpression)
            ):
                v += HyExpression([HySymbol("None")])

            if ann is not None:
                new_args.append(ann)

            new_args.extend((k, v))
        return HyExpression([HySymbol("setv")] + new_args + decls).replace(expr)

    @special(["eval-and-compile", "eval-when-compile"], [many(FORM)])
    def compile_eval_and_compile(
        self, expr: HyExpression, root: str, body: Exprs
    ) -> Result:
        new_expr = HyExpression([HySymbol("do").replace(expr[0])]).replace(expr)

        try:
            hy_eval(new_expr + body,
                    self.module.__dict__,
                    self.module,
                    filename=self.filename,
                    source=self.source,
                    import_stdlib=False)
        except HyInternalError:
            # Unexpected "meta" compilation errors need to be treated
            # like normal (unexpected) compilation errors at this level
            # (or the compilation level preceding this one).
            raise
        except Exception as e:
            # These could be expected Hy language errors (e.g. syntax errors)
            # or regular Python runtime errors that do not signify errors in
            # the compilation *process* (although compilation did technically
            # fail).
            # We wrap these exceptions and pass them through.
            reraise(HyEvalError,
                    HyEvalError(str(e),
                                self.filename,
                                body,
                                self.source),
                    sys.exc_info()[2])

        return (self._compile_branch(body)
                if mangle(root) == "eval_and_compile"
                else Result())

    @special(["py", "pys"], [STR])
    def compile_inline_python(
        self, expr: HyExpression, root: str, code: str
    ) -> Result:
        exec_mode = root == HySymbol("pys")

        try:
            o = ast.parse(
                textwrap.dedent(code) if exec_mode else code,
                self.filename,
                "exec" if exec_mode else "eval")
        except (SyntaxError, ValueError) as e:
            raise self._syntax_error(
                expr,
                "Python parse error in '{}': {}".format(root, e))

        return (
            Result(stmts=cast(ast.Module, o).body)
            if exec_mode
            else Result(expr=cast(ast.Expression, o).body)
        )

    @builds_model(HyExpression)
    def compile_expression(
        self, expr: HyExpression, *, allow_annotation_expression: bool = False
    ) -> Result:
        # Perform macro expansions
        expanded = macroexpand(expr, self.module, self)
        if not isinstance(expanded, HyExpression):
            # Go through compile again if the type changed.
            return self.compile(expanded)

        expr = expanded

        if not expr:
            raise self._syntax_error(
                expr, "empty expressions are not allowed at top level"
            )

        args = list(expr)
        root = args.pop(0)
        func: Optional[Result] = None

        if isinstance(root, HySymbol):
            # First check if `root` is a special operator, unless it has an
            # `unpack-iterable` in it, since Python's operators (`+`,
            # etc.) can't unpack. An exception to this exception is that
            # tuple literals (`,`) can unpack. Finally, we allow unpacking in
            # `.` forms here so the user gets a better error message.
            sroot = mangle(root)

            bad_root = sroot in _bad_roots or (sroot == mangle("annotate*")
                                               and not allow_annotation_expression)

            if (sroot in _special_form_compilers or bad_root) and (
                    sroot in (mangle(","), mangle(".")) or
                    not any(is_unpack("iterable", x) for x in args)):
                if bad_root:
                    raise self._syntax_error(
                        expr, "The special form '{}' is not allowed here".format(root)
                    )
                # `sroot` is a special operator. Get the build method and
                # pattern-match the arguments.
                build_method, pattern = _special_form_compilers[sroot]
                try:
                    parse_tree: list = pattern.parse(args)
                except NoParseError as e:
                    raise self._syntax_error(
                        expr[min(e.state.pos + 1, len(expr) - 1)],
                        "parse error for special form '{}': {}".format(
                            root, e.msg.replace("<EOF>", "end of form")))
                return Result() + build_method(
                    self, expr, unmangle(sroot), *parse_tree)

            if root.startswith("."):
                # (.split "test test") -> "test test".split()
                # (.a.b.c x v1 v2) -> (.c (. x a b) v1 v2) ->  x.a.b.c(v1, v2)

                # Get the method name (the last named attribute
                # in the chain of attributes)
                attrs = [HySymbol(a).replace(root) for a in root.split(".")[1:]]
                if not all(attrs):
                    raise self._syntax_error(expr, "cannot access empty attribute")
                root = attrs.pop()

                # Get the object we're calling the method on
                # (extracted with the attribute access DSL)
                # Skip past keywords and their arguments.
                try:
                    kws, obj, rest = (
                        many(KEYWORD + FORM | unpack("mapping")) +
                        FORM +
                        many(FORM)).parse(args)
                except NoParseError:
                    raise self._syntax_error(expr, "attribute access requires object")
                # Reconstruct `args` to exclude `obj`.
                args = [x for p in kws for x in p] + list(rest)
                if is_unpack("iterable", obj):
                    raise self._syntax_error(
                        obj, "can't call a method on an unpacking form"
                    )
                func = self.compile(
                    HyExpression([HySymbol(".").replace(root), obj] + attrs)
                )

                # And get the method
                func += with_lineno(ast.Attribute)(root,
                                                   value=func.force_expr,
                                                   attr=mangle(root),
                                                   ctx=ast.Load())

        elif is_annotate_expression(root):
            # Flatten and compile the annotation expression.
            ann_expr = HyExpression(root + args).replace(root)
            return self.compile_expression(ann_expr, allow_annotation_expression=True)

        if not func:
            func = self.compile(root)

        args, ret, keywords = self._compile_collect(args, with_kwargs=True)

        return func + ret + with_lineno(ast.Call)(
            expr, func=func.expr, args=args, keywords=keywords)

    @builds_model(HyInteger, HyFloat, HyComplex)
    def compile_numeric_literal(
        self, x: Union[HyInteger, HyFloat, HyComplex]
    ) -> Result:
        f = {HyInteger: int,
             HyFloat: float,
             HyComplex: complex}[type(x)]
        return Result() + with_lineno(ast.Num)(x, n=f(x))

    @builds_model(HySymbol)
    def compile_symbol(self, symbol: HySymbol) -> Result:
        if "." in symbol:
            glob, local = symbol.rsplit(".", 1)

            if not glob:
                error_msg = (
                    "cannot access attribute on anything other than a name "
                    "(in order to get attributes of expressions, use "
                    "`(. <expression> {attr})` or `(.{attr} <expression>)`)"
                )
                raise self._syntax_error(symbol, error_msg.format(attr=local))

            if not local:
                raise self._syntax_error(symbol, "cannot access empty attribute")

            glob = HySymbol(glob).replace(symbol)
            ret = self.compile_symbol(glob)

            return Result(expr=with_lineno(ast.Attribute)(
                symbol,
                value=ret.force_expr,
                attr=mangle(local),
                ctx=ast.Load()))

        if mangle(symbol) in ("None", "False", "True"):
            return Result(
                expr=with_lineno(ast.Constant)(
                    symbol, value=ast.literal_eval(mangle(symbol))
                )
            )

        return Result(
            expr=with_lineno(ast.Name)(symbol, id=mangle(symbol), ctx=ast.Load())
        )

    @builds_model(HyKeyword)
    def compile_keyword(self, obj: HyKeyword) -> Result:
        ret = Result()
        ret += with_lineno(ast.Call)(
            obj,
            func=with_lineno(ast.Name)(obj, id="HyKeyword", ctx=ast.Load()),
            args=[with_lineno(ast.Str)(obj, s=obj.name)],
            keywords=[])
        return ret

    @builds_model(HyString, HyBytes)
    def compile_string(self, string: Union[HyString, HyBytes]) -> Result:
        if isinstance(string, HyString):
            return Result(expr=with_lineno(ast.Str)(string, s=str(string)))
        else:
            return Result(expr=with_lineno(ast.Bytes)(string, s=bytes(string)))

    @builds_model(HyFComponent)
    def compile_fcomponent(self, fcomponent: HyFComponent) -> Result:
        conversion = ord(fcomponent.conversion) if fcomponent.conversion else -1
        root, *rest = fcomponent
        value = self.compile(root)
        elements, ret, _ = self._compile_collect(rest)
        if elements:
            spec = with_lineno(ast.JoinedStr)(fcomponent, values=elements)
        else:
            spec = None
        return value + ret + with_lineno(ast.FormattedValue)(
            fcomponent, value=value.expr, conversion=conversion, format_spec=spec)

    @builds_model(HyFString)
    def compile_fstring(self, fstring: HyFString) -> Result:
        elements, ret, _ = self._compile_collect(fstring)
        return ret + with_lineno(ast.JoinedStr)(fstring, values=elements)

    @builds_model(HyList, HySet)
    def compile_list(self, expression: Union[HyList, HySet]) -> Result:
        elements, ret, _ = self._compile_collect(expression)
        node = {HyList: with_lineno(ast.List), HySet: with_lineno(ast.Set)}[
            type(expression)
        ]
        return ret + node(expression, elts=elements, ctx=ast.Load())

    @builds_model(HyDict)
    def compile_dict(self, m: HyDict) -> Result:
        keyvalues, ret, _ = self._compile_collect(m, dict_display=True)
        return ret + with_lineno(ast.Dict)(m, keys=keyvalues[::2], values=keyvalues[1::2])


# Public Methods ##############################################################
def _get_compiler_module(
    module: Optional[Union[str, types.ModuleType]] = None,
    compiler: HyASTCompiler = None,
    calling_frame: bool = False,
):
    """Get a module object from a compiler, given module object,
    string name of a module, and (optionally) the calling frame; otherwise,
    raise an error."""

    module = getattr(compiler, "module", None) or module

    if isinstance(module, str):
        if module.startswith("<") and module.endswith(">"):
            module = types.ModuleType(module)
        else:
            module = importlib.import_module(mangle(module))

    if calling_frame and not module:
        module = calling_module(n=2)

    if not isinstance(module, types.ModuleType):
        raise TypeError("Invalid module type: {}".format(type(module)))

    return module


def hy_eval(
    hytree: HyObject,
    locals: Optional[dict] = None,
    module: Optional[Union[str, types.ModuleType]] = None,
    ast_callback: Optional[Callable] = None,
    compiler: Optional[HyASTCompiler] = None,
    filename: Optional[str] = None,
    source: Optional[str] = None,
    import_stdlib: bool = True,
):
    """Evaluates a quoted expression and returns the value.

    If you're evaluating hand-crafted AST trees, make sure the line numbers
    are set properly.  Try `fix_missing_locations` and related functions in the
    Python `ast` library.

    Examples:
      ::

         => (hy.eval '(print "Hello World"))
         "Hello World"

      If you want to evaluate a string, use ``read-str`` to convert it to a
      form first::

         => (hy.eval (read-str "(+ 1 1)"))
         2

    Args:
      hytree (HyObject):
          The Hy AST object to evaluate.

      locals (dict, optional):
          Local environment in which to evaluate the Hy tree.  Defaults to the
          calling frame.

      module (str or types.ModuleType, optional):
          Module, or name of the module, to which the Hy tree is assigned and
          the global values are taken.
          The module associated with `compiler` takes priority over this value.
          When neither `module` nor `compiler` is specified, the calling frame's
          module is used.

      ast_callback (callable, optional):
          A callback that is passed the Hy compiled tree and resulting
          expression object, in that order, after compilation but before
          evaluation.

      compiler (HyASTCompiler, optional):
          An existing Hy compiler to use for compilation.  Also serves as
          the `module` value when given.

      filename (str, optional):
          The filename corresponding to the source for `tree`.  This will be
          overridden by the `filename` field of `tree`, if any; otherwise, it
          defaults to "<string>".  When `compiler` is given, its `filename` field
          value is always used.

      source (str, optional):
          A string containing the source code for `tree`.  This will be
          overridden by the `source` field of `tree`, if any; otherwise,
          if `None`, an attempt will be made to obtain it from the module given by
          `module`.  When `compiler` is given, its `source` field value is always
          used.

    Returns:
      Result of evaluating the Hy compiled tree.
    """

    module = _get_compiler_module(module, compiler, True)

    if locals is None:
        frame = inspect.stack()[1][0]
        locals = inspect.getargvalues(frame).locals

    if not isinstance(locals, dict):
        raise TypeError("Locals must be a dictionary")

    # Does the Hy AST object come with its own information?
    filename = cast(str, getattr(hytree, "filename", filename) or "<string>")
    source = getattr(hytree, "source", source)

    _ast, expr = cast(tuple, hy_compile(hytree, module, get_expr=True,
                                        compiler=compiler, filename=filename,
                                        source=source, import_stdlib=import_stdlib))

    if ast_callback:
        ast_callback(_ast, expr)

    # Two-step eval: eval() the body of the exec call
    eval(ast_compile(_ast, filename, "exec"),
         module.__dict__, locals)

    # Then eval the expression context and return that
    return eval(ast_compile(expr, filename, "eval"),
                module.__dict__, locals)


def hy_compile(
    tree: HyObject,
    module: Union[str, types.ModuleType],
    root: Type[ast.AST] = ast.Module,
    get_expr: bool = False,
    compiler: Optional[HyASTCompiler] = None,
    filename: Optional[str] = None,
    source: Optional[str] = None,
    import_stdlib: bool = True,
) -> Union[ast.AST, Tuple[ast.AST, Optional[ast.Expression]]]:
    """Compile a HyObject tree into a Python AST Module.

    Args:
        tree: The Hy AST object to compile.
        module (optional): Module, or name of the module, in which the Hy tree
            is evaluated. The module associated with `compiler` takes priority over
            this value.
        root (optional): Root object for the Python AST tree.
        get_expr (optional): If true, return a tuple with `(root_obj, last_expression)`.
        compiler (optional): An existing Hy compiler to use for compilation.  Also
            serves as the `module` value when given.
        filename (optional): The filename corresponding to the source for `tree`.
            This will be overridden by the `filename` field of `tree`, if any;
            otherwise, it defaults to "<string>".  When `compiler` is given, its
            `filename` field value is always used.
        source (optional): A string containing the source code for `tree`. This will
            be overridden by the `source` field of `tree`, if any; otherwise,
            if `None`, an attempt will be made to obtain it from the module given by
            `module`.  When `compiler` is given, its `source` field value is always
            used.

    Returns:
        out: A Python AST tree
    """
    module = _get_compiler_module(module, compiler, False)

    if isinstance(module, str):
        if module.startswith("<") and module.endswith(">"):
            module = types.ModuleType(module)
        else:
            module = importlib.import_module(mangle(module))

    if not inspect.ismodule(module):
        raise TypeError("Invalid module type: {}".format(type(module)))

    filename = getattr(tree, "filename", filename)
    source = getattr(tree, "source", source)

    tree = wrap_value(tree)
    if not isinstance(tree, HyObject):
        raise TypeError("`tree` must be a HyObject or capable of "
                        "being promoted to one")

    compiler = compiler or HyASTCompiler(
        module, filename=filename or "<string>", source=source
    )

    if import_stdlib:
        # Import hy for compile time, but save the compiled AST.
        stdlib_ast = compiler.compile(
            mkexpr("eval-and-compile", mkexpr("import", "hy"))
        )
    else:
        stdlib_ast = Result()

    result = compiler.compile(tree)
    expr = result.force_expr

    if not get_expr:
        result += result.expr_as_stmt()

    body = []

    if issubclass(root, ast.Module):
        # Pull out a single docstring and prepend to the resulting body.
        if (
            result.stmts
            and isinstance(result.stmts[0], ast.Expr)
            and isinstance(result.stmts[0].value, ast.Str)
        ):
            docstring, *result.stmts = result.stmts
            body += [docstring]

        # Pull out any __future__ imports, since they are required to be at the
        # beginning.
        while (
            result.stmts
            and isinstance(result.stmts[0], ast.ImportFrom)
            and result.stmts[0].module == "__future__"
        ):
            future_import, *result.stmts = result.stmts
            body += [future_import]

        # Import hy for runtime.
        if import_stdlib:
            body += stdlib_ast.stmts

    body += result.stmts
    ret = root(body=body, type_ignores=[])

    if get_expr:
        expr = ast.Expression(body=expr)
        return ret, expr

    return ret

import ast

import hy
from funcparserlib.parser import many, oneplus
from hy.compiler import Result, asty, maybe, mkexpr
from hy.lex import mangle, unmangle
from hy.model_patterns import FORM, SYM, brackets, sym
from hy.models import Expression, Symbol


@hy.macros.macro("do")
def do(compiler, *body):
    return compiler._compile_branch(body)


@hy.macros.macro("unpack-iterable")
def unpack_iterable(compiler, arg):
    ret = compiler.compile(arg)
    ret += asty.Starred(compiler.this, value=ret.force_expr, ctx=ast.Load())
    return ret


@hy.macros.pattern_macro("raise", [maybe(FORM), maybe(sym(":from") + FORM)])
def compile_raise_expression(compiler, exc, cause):
    ret = Result()

    if exc is not None:
        exc = compiler.compile(exc)
        ret += exc
        exc = exc.force_expr

    if cause is not None:
        cause = compiler.compile(cause)
        ret += cause
        cause = cause.force_expr

    return ret + asty.Raise(
        compiler.this,
        type=ret.expr,
        exc=exc,
        inst=None,
        tback=None,
        cause=cause,
    )


@hy.macros.pattern_macro("if", [FORM, FORM, maybe(FORM)])
def compile_if(compiler, cond, body, orel_expr):
    cond = compiler.compile(cond)
    body = compiler.compile(body)

    nested = root = False
    orel = Result()
    if orel_expr is not None:
        if (
            isinstance(orel_expr, Expression)
            and isinstance(orel_expr[0], Symbol)
            and orel_expr[0] == Symbol("if*")
        ):
            # Nested ifs: don't waste temporaries
            root = compiler.temp_if is None
            nested = True
            compiler.temp_if = compiler.temp_if or compiler.get_anon_var()
        orel = compiler.compile(orel_expr)

    if not cond.stmts and isinstance(cond.force_expr, ast.Name):
        name = cond.force_expr.id
        branch = None
        if name == "True":
            branch = body
        elif name in ("False", "None"):
            branch = orel
        if branch is not None:
            if compiler.temp_if and branch.stmts:
                name = asty.Name(
                    compiler.this, id=mangle(compiler.temp_if), ctx=ast.Store()
                )

                branch += asty.Assign(
                    compiler.this, targets=[name], value=body.force_expr
                )

            return branch

    # We want to hoist the statements from the condition
    ret = cond

    if body.stmts or orel.stmts:
        # We have statements in our bodies
        # Get a temporary variable for the result storage
        var = compiler.temp_if or compiler.get_anon_var()
        name = asty.Name(compiler.this, id=mangle(var), ctx=ast.Store())

        # Store the result of the body
        body += asty.Assign(compiler.this, targets=[name], value=body.force_expr)

        # and of the else clause
        if not nested or not orel.stmts or (not root and var != compiler.temp_if):
            orel += asty.Assign(compiler.this, targets=[name], value=orel.force_expr)

        # Then build the if
        ret += asty.If(
            compiler.this, test=ret.force_expr, body=body.stmts, orelse=orel.stmts
        )

        # And make our expression context our temp variable
        expr_name = asty.Name(compiler.this, id=mangle(var), ctx=ast.Load())

        ret += Result(expr=expr_name, temp_variables=[expr_name, name])
    else:
        # Just make that an if expression
        ret += asty.IfExp(
            compiler.this,
            test=ret.force_expr,
            body=body.force_expr,
            orelse=orel.force_expr,
        )

    if root:
        compiler.temp_if = None

    return ret


@hy.macros.macro("break")
@hy.macros.macro("continue")
def compile_break_or_continue_expression(compiler):
    node = asty.Break if compiler.this[0] == Symbol("break") else asty.Continue
    return Result() + node(compiler.this)


@hy.macros.macro("assert")
def compile_assert_expression(compiler, test, msg=None):
    if msg is None or type(msg) is Symbol:
        ret = compiler.compile(test)
        return ret + asty.Assert(
            compiler.this,
            test=ret.force_expr,
            msg=(None if msg is None else compiler.compile(msg).force_expr),
        )

    # The `msg` part may involve statements, which we only
    # want to be executed if the assertion fails. Rewrite the
    # form to set `msg` to a variable.
    msg_var = compiler.get_anon_var()
    return compiler.compile(
        mkexpr(
            "if",
            mkexpr("and", "__debug__", mkexpr("not", [test])),
            mkexpr(
                "do", mkexpr("setv", msg_var, [msg]), mkexpr("assert", "False", msg_var)
            ),
        ).replace(compiler.this)
    )


@hy.macros.pattern_macro("global", [oneplus(SYM)])
@hy.macros.pattern_macro("nonlocal", [oneplus(SYM)])
def compile_global_or_nonlocal(compiler, syms):
    node = asty.Global if compiler.this[0] == Symbol("global") else asty.Nonlocal
    return Result() + node(compiler.this, names=list(map(mangle, syms)))


@hy.macros.macro("yield")
def compile_yield_expression(compiler, arg=None):
    ret = Result()
    if arg is not None:
        ret += compiler.compile(arg)
    return ret + asty.Yield(compiler.this, value=ret.force_expr)


@hy.macros.macro("yield-from")
@hy.macros.macro("await")
def compile_yield_from_or_await_expression(compiler, arg):
    ret = Result() + compiler.compile(arg)
    node = asty.YieldFrom if compiler.this[0] == Symbol("yield-from") else asty.Await
    return ret + node(compiler.this, value=ret.force_expr)


@hy.macros.pattern_macro("get", [FORM, oneplus(FORM)])
def compile_index_expression(compiler, obj, indices):
    indices, ret, _ = compiler._compile_collect(indices)
    ret += compiler.compile(obj)

    for ix in indices:
        ret += asty.Subscript(
            compiler.this,
            value=ret.force_expr,
            slice=ast.Index(value=ix),
            ctx=ast.Load())

    return ret

@hy.macros.pattern_macro(".", [FORM, many(SYM | brackets(FORM))])
def compile_attribute_access(compiler, invocant, keys):
    ret = compiler.compile(invocant)

    for attr in keys:
        if isinstance(attr, Symbol):
            ret += asty.Attribute(attr,
                                    value=ret.force_expr,
                                    attr=mangle(attr),
                                    ctx=ast.Load())
        else: # attr is a hy List
            compiled_attr = compiler.compile(attr[0])
            ret = compiled_attr + ret + asty.Subscript(
                attr,
                value=ret.force_expr,
                slice=ast.Index(value=compiled_attr.force_expr),
                ctx=ast.Load())

    return ret

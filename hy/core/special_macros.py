import ast

import hy
from hy.compiler import Result, asty, maybe
from hy.model_patterns import FORM, sym
from hy.models import Expression, Symbol
from hy.lex import mangle, unmangle


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

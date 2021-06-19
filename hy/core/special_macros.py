import ast

import hy
from funcparserlib.parser import many, oneplus
from hy.compiler import Result, asty, is_unpack, maybe, mkexpr
from hy.lex import mangle, unmangle
from hy.model_patterns import FORM, SYM, brackets, dolike, notpexpr, pexpr, sym, times
from hy.models import Expression, Integer, List, Symbol

Inf = float("inf")


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
            ctx=ast.Load(),
        )

    return ret


@hy.macros.pattern_macro(".", [FORM, many(SYM | brackets(FORM))])
def compile_attribute_access(compiler, invocant, keys):
    ret = compiler.compile(invocant)

    for attr in keys:
        if isinstance(attr, Symbol):
            ret += asty.Attribute(
                attr, value=ret.force_expr, attr=mangle(attr), ctx=ast.Load()
            )
        else:  # attr is a hy List
            compiled_attr = compiler.compile(attr[0])
            ret = (
                compiled_attr
                + ret
                + asty.Subscript(
                    attr,
                    value=ret.force_expr,
                    slice=ast.Index(value=compiled_attr.force_expr),
                    ctx=ast.Load(),
                )
            )

    return ret


@hy.macros.macro("del")
def compile_del_expression(compiler, *args):
    if not args:
        return Result() + asty.Pass(compiler.this)

    del_targets = []
    ret = Result()
    for target in args:
        compiled_target = compiler.compile(target)
        ret += compiled_target
        del_targets.append(compiler._storeize(target, compiled_target, ast.Del))

    return ret + asty.Delete(compiler.this, targets=del_targets)


@hy.macros.macro("cut")
def compile_cut_expression(compiler, obj, lower=None, upper=None, step=None):
    ret = [Result()]

    def c(e):
        ret[0] += compiler.compile(e)
        return ret[0].force_expr

    if upper is None:
        # cut with single index is an upper bound,
        # this is consistent with slice and islice
        upper = lower
        lower = Symbol("None")

    s = asty.Subscript(
        compiler.this,
        value=c(obj),
        slice=asty.Slice(compiler.this, lower=c(lower), upper=c(upper), step=c(step)),
        ctx=ast.Load(),
    )
    return ret[0] + s


_decoratables = (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)


@hy.macros.macro("with-decorator")
def compile_decorate_expression(compiler, arg, *args):
    args = [arg, *args]
    decs, fn = args[:-1], compiler.compile(args[-1])
    if not fn.stmts or not isinstance(fn.stmts[-1], _decoratables):
        raise compiler._syntax_error(args[-1], "Decorated a non-function")
    decs, ret, _ = compiler._compile_collect(decs)
    fn.stmts[-1].decorator_list = decs + fn.stmts[-1].decorator_list
    return ret + fn


@hy.macros.macro(",")
def compile_tuple(compiler, *args):
    elts, ret, _ = compiler._compile_collect(args)
    return ret + asty.Tuple(compiler.this, elts=elts, ctx=ast.Load())


@hy.macros.macro("not")
@hy.macros.macro("~")
def compile_unary_operator(compiler, arg):
    root = str(compiler.this[0])
    ops = {"not": ast.Not, "~": ast.Invert}
    operand = compiler.compile(arg)
    return operand + asty.UnaryOp(
        compiler.this, op=ops[root](), operand=operand.force_expr
    )


def fallback_on_call(compiler, root, args):
    func = compiler.compile(root)

    args, ret, keywords = compiler._compile_collect(args, with_kwargs=True)

    return (
        func
        + ret
        + asty.Call(compiler.this, func=func.expr, args=args, keywords=keywords)
    )


@hy.macros.macro("and")
@hy.macros.macro("or")
def compile_logical_or_and_and_operator(compiler, *args):
    operator = str(compiler.this[0])
    # and/or can't unpack so we have to fallback on shadowed operators
    if any(is_unpack("iterable", arg) for arg in args):
        return fallback_on_call(compiler, compiler.this[0], args)

    ops = {"and": (ast.And, True), "or": (ast.Or, None)}
    opnode, default = ops[operator]
    osym = compiler.this[0]
    if len(args) == 0:
        return Result() + asty.Constant(osym, value=default)
    elif len(args) == 1:
        return compiler.compile(args[0])
    ret = Result()
    values = list(map(compiler.compile, args))
    if any(value.stmts for value in values):
        # Compile it to an if...else sequence
        var = compiler.get_anon_var()
        name = asty.Name(osym, id=var, ctx=ast.Store())
        expr_name = asty.Name(osym, id=var, ctx=ast.Load())
        temp_variables = [name, expr_name]

        def make_assign(value, node=None):
            positioned_name = asty.Name(node or osym, id=var, ctx=ast.Store())
            temp_variables.append(positioned_name)
            return asty.Assign(node or osym, targets=[positioned_name], value=value)

        current = root = []
        for i, value in enumerate(values):
            if value.stmts:
                node = value.stmts[0]
                current.extend(value.stmts)
            else:
                node = value.expr
            current.append(make_assign(value.force_expr, value.force_expr))
            if i == len(values) - 1:
                # Skip a redundant 'if'.
                break
            if operator == "and":
                cond = expr_name
            else:
                cond = asty.UnaryOp(node, op=ast.Not(), operand=expr_name)
            current.append(asty.If(node, test=cond, body=[], orelse=[]))
            current = current[-1].body
        ret = sum(root, ret)
        ret += Result(expr=expr_name, temp_variables=temp_variables)
    else:
        ret += asty.BoolOp(
            osym, op=opnode(), values=[value.force_expr for value in values]
        )
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


def _get_c_op(compiler, sym):
    k = mangle(sym)
    if k not in _c_ops:
        raise compiler._syntax_error(sym, "Illegal comparison operator: " + str(sym))
    return _c_ops[k]()


@hy.macros.pattern_macro(["=", "is", "<", "<=", ">", ">="], [oneplus(FORM)])
@hy.macros.pattern_macro(["!=", "is-not", "in", "not-in"], [times(2, Inf, FORM)])
def compile_compare_op_expression(compiler, args):
    if any(is_unpack("iterable", arg) for arg in args):
        return fallback_on_call(compiler, compiler.this[0], args)

    if len(args) == 1:
        return compiler.compile(args[0]) + asty.Constant(compiler.this, value=True)

    ops = [_get_c_op(compiler, compiler.this[0]) for _ in args[1:]]
    exprs, ret, _ = compiler._compile_collect(args)
    return ret + asty.Compare(
        compiler.this, left=exprs[0], ops=ops, comparators=exprs[1:]
    )


# @hy.macros.pattern_macro("chainc", [FORM, many(SYM + FORM)])
# def compile_chained_comparison(compiler, arg1, args):
#     ret = compiler.compile(arg1)
#     arg1 = ret.force_expr

#     ops = [_get_c_op(compiler, op) for op, _ in args]
#     args, ret2, _ = compiler._compile_collect(
#         [x for _, x in args])

#     return ret + ret2 + asty.Compare(compiler.this,
#         left=arg1, ops=ops, comparators=args)

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


@hy.macros.pattern_macro(["+", "*", "|"], [many(FORM)])
@hy.macros.pattern_macro(["-", "/", "&", "@", "**", "//", "<<", ">>"], [oneplus(FORM)])
@hy.macros.pattern_macro(["%", "^"], [times(1, 2, FORM)])
def compile_maths_expression(compiler, args):
    root = str(compiler.this[0])
    if any(is_unpack("iterable", arg) for arg in args) or (
        len(args) == 1 and (root in ["**", "//", "<<", ">>", "%", "^"])
    ):
        return fallback_on_call(compiler, compiler.this[0], args)
    if len(args) == 0:
        # Return the identity element for this operator.
        return Result() + asty.Num(compiler.this, n=({"+": 0, "|": 0, "*": 1}[root]))

    if len(args) == 1:
        if root == "/":
            # Compute the reciprocal of the argument.
            args = [Integer(1).replace(compiler.this), args[0]]
        elif root in ("+", "-"):
            # Apply unary plus or unary minus to the argument.
            op = {"+": ast.UAdd, "-": ast.USub}[root]()
            ret = compiler.compile(args[0])
            return ret + asty.UnaryOp(compiler.this, op=op, operand=ret.force_expr)
        else:
            # Return the argument unchanged.
            return compiler.compile(args[0])

    op = m_ops[root][0]
    right_associative = root == "**"
    ret = compiler.compile(args[-1 if right_associative else 0])
    for child in args[-2 if right_associative else 1 :: -1 if right_associative else 1]:
        left_expr = ret.force_expr
        ret += compiler.compile(child)
        right_expr = ret.force_expr
        if right_associative:
            left_expr, right_expr = right_expr, left_expr
        ret += asty.BinOp(compiler.this, left=left_expr, op=op(), right=right_expr)

    return ret


a_ops = {x + "=": v for x, v in m_ops.items()}


@hy.macros.pattern_macro(
    [x for x, (_, v) in a_ops.items() if v is not None], [FORM, oneplus(FORM)]
)
@hy.macros.pattern_macro(
    [x for x, (_, v) in a_ops.items() if v is None], [FORM, times(1, 1, FORM)]
)
def compile_augassign_expression(compiler, target, values):
    if any(is_unpack("iterable", arg) for arg in values):
        return fallback_on_call(compiler, compiler.this[0], values)
    root = str(compiler.this[0])
    if len(values) > 1:
        return compiler.compile(
            mkexpr(
                root, [target], mkexpr(a_ops[root][1], rest=values)
            ).replace(compiler.this)
        )

    op = a_ops[root][0]
    target = compiler._storeize(target, compiler.compile(target))
    ret = compiler.compile(values[0])
    return ret + asty.AugAssign(
        compiler.this, target=target, value=ret.force_expr, op=op()
    )


@hy.macros.pattern_macro(
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
def compile_try_expression(compiler, body, catchers, orelse, finalbody):
    body = compiler._compile_branch(body)

    return_var = asty.Name(
        compiler.this, id=mangle(compiler.get_anon_var()), ctx=ast.Store()
    )

    handler_results = Result()
    handlers = []
    for catcher in catchers:
        handler_results += _compile_catch_expression(
            compiler, catcher, return_var, *catcher
        )
        handlers.append(handler_results.stmts.pop())

    if orelse is None:
        orelse = []
    else:
        orelse = compiler._compile_branch(orelse)
        orelse += asty.Assign(
            compiler.this, targets=[return_var], value=orelse.force_expr
        )
        orelse += orelse.expr_as_stmt()
        orelse = orelse.stmts

    if finalbody is None:
        finalbody = []
    else:
        finalbody = compiler._compile_branch(finalbody)
        finalbody += finalbody.expr_as_stmt()
        finalbody = finalbody.stmts

    # Using (else) without (except) is verboten!
    if orelse and not handlers:
        raise compiler._syntax_error(
            compiler.this, "`try' cannot have `else' without `except'"
        )
    # Likewise a bare (try) or (try BODY).
    if not (handlers or finalbody):
        raise compiler._syntax_error(
            compiler.this, "`try' must have an `except' or `finally' clause"
        )

    returnable = Result(
        expr=asty.Name(compiler.this, id=return_var.id, ctx=ast.Load()),
        temp_variables=[return_var],
    )
    body += (
        body.expr_as_stmt()
        if orelse
        else asty.Assign(compiler.this, targets=[return_var], value=body.force_expr)
    )
    body = body.stmts or [asty.Pass(compiler.this)]

    x = asty.Try(
        compiler.this, body=body, handlers=handlers, orelse=orelse, finalbody=finalbody
    )
    return handler_results + x + returnable


def _compile_catch_expression(compiler, expr, var, exceptions, body):
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
        name = mangle(compiler._nonconst(exceptions[0]))

    exceptions_list = exceptions[-1] if exceptions else List()
    if isinstance(exceptions_list, List):
        if len(exceptions_list):
            # [FooBar BarFoo] → catch Foobar and BarFoo exceptions
            elts, types, _ = compiler._compile_collect(exceptions_list)
            types += asty.Tuple(exceptions_list, elts=elts, ctx=ast.Load())
        else:
            # [] → all exceptions caught
            types = Result()
    else:
        types = compiler.compile(exceptions_list)

    body = compiler._compile_branch(body)
    body += asty.Assign(expr, targets=[var], value=body.force_expr)
    body += body.expr_as_stmt()

    return types + asty.ExceptHandler(
        expr, type=types.expr, name=name, body=body.stmts or [asty.Pass(expr)]
    )


@hy.macros.pattern_macro(
    ["with", "with/a"],
    [
        (brackets(times(1, Inf, FORM + FORM)))
        | brackets((FORM >> (lambda x: [(Symbol("_"), x)]))),
        many(FORM),
    ],
)
def compile_with_expression(compiler, args, body):
    body = compiler._compile_branch(body)

    # Store the result of the body in a tempvar
    temp_var = compiler.get_anon_var()
    name = asty.Name(compiler.this, id=mangle(temp_var), ctx=ast.Store())
    body += asty.Assign(compiler.this, targets=[name], value=body.force_expr)
    # Initialize the tempvar to None in case the `with` exits
    # early with an exception.
    initial_assign = asty.Assign(
        compiler.this, targets=[name], value=asty.Constant(compiler.this, value=None)
    )

    ret = Result(stmts=[initial_assign])
    items = []
    for variable, ctx in args[0]:
        ctx = compiler.compile(ctx)
        ret += ctx
        variable = (
            None
            if variable == Symbol("_")
            else compiler._storeize(variable, compiler.compile(variable))
        )
        items.append(
            asty.withitem(
                compiler.this, context_expr=ctx.force_expr, optional_vars=variable
            )
        )

    node = asty.With if str(compiler.this[0]) == "with" else asty.AsyncWith
    ret += node(compiler.this, body=body.stmts, items=items)

    # And make our expression context our temp variable
    expr_name = asty.Name(compiler.this, id=mangle(temp_var), ctx=ast.Load())

    ret += Result(expr=expr_name)
    # We don't give the Result any temp_vars because we don't want
    # Result.rename to touch `name`. Otherwise, initial_assign will
    # clobber any preexisting value of the renamed-to variable.

    return ret

import hy
from hy.compiler import Result, asty

@hy.macros.macro('m-ret')
def f(hy_compiler, arg = None):
    if arg is None:
        return asty.Return(hy_compiler.this, value=None)
    ret = Result() + hy_compiler.compile(arg)
    return ret + asty.Return(hy_compiler.this, value=ret.force_expr)

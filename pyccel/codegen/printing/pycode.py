# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=R0201
# pylint: disable=missing-function-docstring

from itertools import chain

from sympy.core import Tuple

from sympy.printing.pycode import PythonCodePrinter as SympyPythonCodePrinter
from sympy.printing.pycode import _known_functions
from sympy.printing.pycode import _known_functions_math
from sympy.printing.pycode import _known_constants_math

from pyccel.ast.core       import CodeBlock
from pyccel.ast.core import Nil
from pyccel.ast.datatypes import NativeSymbol, NativeString, str_dtype
from pyccel.ast.datatypes import NativeInteger, NativeBool, NativeReal
from pyccel.ast.utilities  import build_types_decorator
from pyccel.ast.operators  import PyccelAdd, PyccelMul, PyccelDiv, PyccelMinus
from pyccel.ast.literals  import LiteralInteger, LiteralFloat
from pyccel.ast.literals  import LiteralTrue
from pyccel.ast.builtins  import (PythonEnumerate, PythonInt, PythonLen,
                                  PythonMap, PythonPrint, PythonRange,
                                  PythonZip, PythonFloat, PythonTuple, PythonList)
from pyccel.ast.builtins  import PythonComplex, PythonBool
from pyccel.ast.itertoolsext import Product

from pyccel.errors.errors import Errors
from pyccel.errors.messages import *

errors = Errors()

#==============================================================================
def _construct_header(func_name, args):
    args = build_types_decorator(args, order='F')
    args = ','.join("{}".format(i) for i in args)
    pattern = '#$ header function static {name}({args})'
    return pattern.format(name=func_name, args=args)

#==============================================================================
class PythonCodePrinter(SympyPythonCodePrinter):
    _kf = dict(chain(
        _known_functions.items(),
        [(k, '' + v) for k, v in _known_functions_math.items()]
    ))
    _kc = {k: ''+v for k, v in _known_constants_math.items()}

    def __init__(self, parser=None, settings={}):
        self.assert_contiguous = settings.pop('assert_contiguous', False)
        self.parser = parser
        SympyPythonCodePrinter.__init__(self, settings=settings)

    def _print_Variable(self, expr):
        return self._print(expr.name)

    def _print_VariableAddress(self, expr):
        return self._print(expr.variable)

    def _print_Idx(self, expr):
        return self._print(expr.name)

    def _print_IndexedElement(self, expr):
        indices = expr.indices
        if isinstance(indices, (tuple, list, Tuple)):
            # this a fix since when having a[i,j] the generated code is a[(i,j)]
            if len(indices) == 1 and isinstance(indices[0], (tuple, list, Tuple)):
                indices = indices[0]

            indices = [self._print(i) for i in indices]
            indices = ','.join(i for i in indices)
        else:
            errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                severity='fatal')

        base = self._print(expr.base)
        return '{base}[{indices}]'.format(base=base, indices=indices)

    def _print_FunctionDef(self, expr):
        name = self._print(expr.name)
        body = self._print(expr.body)
        body = self._indent_codestring(body)
        args = ', '.join(self._print(i) for i in expr.arguments)

        imports = '\n'.join(self._print(i) for i in expr.imports)
        imports = self._indent_codestring(imports)

        functions = expr.functions
        if len(functions)>0:
            functions = '\n'.join(self._print(i) for  i in functions)
            functions = self._indent_codestring(functions)
            body = functions + '\n' + body

        code = ('def {name}({args}):\n'
                '\n{imports}\n{body}\n').format(name=name, args=args,imports=imports, body=body)

        decorators = expr.decorators

        if decorators:
            for n,f in decorators.items():
                # TODO - All decorators must be stored in a list
                if not isinstance(f, list):
                    f = [f]
                dec = ''
                for func in f:
                    args = func.args
                    if args:
                        args = ', '.join("{}".format(self._print(i)) for i in args)
                        dec += '@{name}({args})\n'.format(name=n, args=args)

                    else:
                        dec += '@{name}\n'.format(name=n)

                code = '{dec}{code}'.format(dec=dec, code=code)
        headers = expr.headers
        if headers:
            headers = self._print(headers)
            code = '{header}\n{code}'.format(header=header, code=code)

        return code

    def _print_Return(self, expr):

        code = ''
        if expr.stmt:
            code += self._print(expr.stmt)

        if len(expr.expr) == 1:
            return_code = 'return {}'.format(self._print(expr.expr[0]))

        else:
            return_code = 'return {}'.format(self._print(expr.expr))

        return '{0}\n{1}'.format(code, return_code)


    def _print_PythonTuple(self, expr):
        args = ', '.join(self._print(i) for i in expr.args)
        return '('+args+')'

    def _print_Comment(self, expr):
        txt = self._print(expr.text)
        return '# {0} '.format(txt)

    def _print_EmptyNode(self, expr):
        return ''

    def _print_NewLine(self, expr):
        return '\n'

    def _print_DottedName(self, expr):
        return '.'.join(self._print(n) for n in expr.name)

    def _print_FunctionCall(self, expr):
        func = self._print(expr.func)
        args = ','.join(self._print(i) for i in expr.arguments)
        return'{func}({args})'.format(func=func, args=args)

    def _print_Len(self, expr):
        return 'len({})'.format(self._print(expr.arg))

    def _print_Import(self, expr):
        target = ', '.join([self._print(i) for i in expr.target])
        if expr.source is None:
            return 'import {target}'.format(target=target)
        else:
            source = self._print(expr.source)
            return 'from {source} import {target}'.format(source=source, target=target)

    def _print_CodeBlock(self, expr):
        code = '\n'.join(self._print(c) for c in expr.body)
        return code

    def _print_For(self, expr):
#        if not isinstance(expr.iterable, (PythonRange, Product , PythonZip,
#                            PythonEnumerate, PythonMap)):
        if not isinstance(expr.iterable, (PythonRange, Product, PythonMap)):
            # Only iterable currently supported are PythonRange or Product
            errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                severity='fatal')

        iterable = expr.iterable
        if isinstance(iterable, PythonRange):
            iterable = self._print(iterable)

        elif isinstance(iterable, Product):
            iterable = self._print(iterable)

        elif isinstance(iterable, PythonMap):
            iterable = PythonRange(PythonLen(iterable.args[1]))
            iterable = self._print(iterable)

        else:
            raise NotImplementedError()

        target   = expr.target
        if not isinstance(target,(list, tuple, Tuple)):
            target = [target]

        target = ','.join(self._print(i) for i in target)
        body   = self._print(expr.body)
        body   = self._indent_codestring(body)
        code   = ('for {0} in {1}:\n'
                '{2}\n').format(target,iterable,body)

        return code

    def _print_Assign(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return'{0} = {1}'.format(lhs,rhs)

    def _print_AugAssign(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        op  = self._print(expr.op._symbol)
        return'{0} {1}= {2}'.format(lhs,op,rhs)

    def _print_Range(self, expr):
        start = self._print(expr.start)
        stop  = self._print(expr.stop)
        step  = self._print(expr.step)
        return 'range({}, {}, {})'.format(start,stop,step)

    def _print_PythonRange(self, expr):
        start = self._print(expr.start)
        stop  = self._print(expr.stop)
        step  = self._print(expr.step)
        return 'range({}, {}, {})'.format(start,stop,step)

    def _print_Product(self, expr):
        args = ','.join(self._print(i) for i in expr.elements)
        return 'product({})'.format(args)

    def _print_IndexedBase(self, expr):
        return self._print(expr.label)

    def _print_Indexed(self, expr):
        inds = list(expr.indices)
        #indices of indexedElement of len==1 shouldn't be a Tuple
        for i, ind in enumerate(inds):
            if isinstance(ind, Tuple) and len(ind) == 1:
                inds[i] = ind[0]

        inds = [self._print(i) for i in inds]

        return "%s[%s]" % (self._print(expr.base.label), ", ".join(inds))

    def _print_Zeros(self, expr):
        return 'zeros('+ self._print(expr.shape)+')'

    def _print_ZerosLike(self, expr):
        return 'zeros_like('+ self._print(expr.rhs)+')'

    def _print_Max(self, expr):
        args = ', '.join(self._print(e) for e in expr.args)
        return 'max({})'.format(args)

    def _print_Min(self, expr):
        args = ', '.join(self._print(e) for e in expr.args)
        return 'min({})'.format(args)

    def _print_Slice(self, expr):
        return str(expr)

    def _print_Nil(self, expr):
        return 'None'

    def _print_Pass(self, expr):
        return 'pass'

    def _print_PyccelIs(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return'{0} is {1}'.format(lhs,rhs)

    def _print_If(self, expr):
        lines = []
        for i, (c, e) in enumerate(expr.args):
            if i == 0:
                lines.append("if (%s):" % self._print(c))

            elif i == len(expr.args) - 1 and c is True:
                lines.append("else:")

            else:
                lines.append("elif (%s):" % self._print(c))

            if isinstance(e, CodeBlock):
                body = self._indent_codestring(self._print(e))
                lines.append(body)
            else:
                lines.append(self._print(e))
        return "\n".join(lines)

    def _print_LiteralString(self, expr):
        return '"{}"'.format(self._print(expr.arg))

    def _print_LiteralInteger(self, expr):
        return '{}'.format(self._print(expr.python_value))

    def _print_Shape(self, expr):
        arg = self._print(expr.arg)
        if expr.index is None:
            return '{}.shape'.format(arg)

        else:
            index = self._print(expr.index)
            return '{0}.shape[{1}]'.format(arg, index)

    def _print_Print(self, expr):
        args = []
        for f in expr.expr:
            if isinstance(f, str):
                args.append("'{}'".format(f))

            elif isinstance(f, Tuple):
                for i in f:
                    args.append("{}".format(self._print(i)))

            else:
                args.append("{}".format(self._print(f)))

        fs = ', '.join(i for i in args)

        return 'print({0})'.format(fs)

    def _print_Module(self, expr):
        return '\n'.join(self._print(e) for e in expr.body)

    def _print_PyccelPow(self, expr):
        base = self._print(expr.args[0])
        e    = self._print(expr.args[1])
        return '{} ** {}'.format(base, e)

    def _print_PyccelAdd(self, expr):
        return ' + '.join(self._print(a) for a in expr.args)

    def _print_PyccelMinus(self, expr):
        return ' - '.join(self._print(a) for a in expr.args)

    def _print_PyccelMul(self, expr):
        return ' * '.join(self._print(a) for a in expr.args)

    def _print_PyccelDiv(self, expr):
        return ' / '.join(self._print(a) for a in expr.args)

    def _print_PyccelMod(self, expr):
        return '%'.join(self._print(a) for a in expr.args)

    def _print_PyccelFloorDiv(self, expr):
        return '//'.join(self._print(a) for a in expr.args)

    def _print_PyccelAssociativeParenthesis(self, expr):
        return '({})'.format(self._print(expr.args[0]))

    def _print_PyccelUnary(self, expr):
        return '({})'.format(self._print(expr.args[0]))

    def _print_PyccelAnd(self, expr):
        return ' and '.join(self._print(a) for a in expr.args)

    def _print_PyccelOr(self, expr):
        return ' or '.join(self._print(a) for a in expr.args)

    def _print_PyccelEq(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} == {1} '.format(lhs, rhs)

    def _print_PyccelNe(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} != {1} '.format(lhs, rhs)

    def _print_PyccelLt(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} < {1}'.format(lhs, rhs)

    def _print_PyccelLe(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} <= {1}'.format(lhs, rhs)

    def _print_PyccelGt(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} > {1}'.format(lhs, rhs)

    def _print_PyccelGe(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} >= {1}'.format(lhs, rhs)

    def _print_PyccelNot(self, expr):
        a = self._print(expr.args[0])
        return 'not {}'.format(a)

    def _print_PyccelMinus(self, expr):
        args = [self._print(a) for a in expr.args]

        if len(args) == 1:
            return '-{}'.format(args[0])
        return ' - '.join(args)

    def _print_PyccelArraySize(self, expr):
        arg = self._print(expr.arg)
        index = self._print(expr.index)
        return 'shape({0})[{1}]'.format(arg, index)

    def _print_LiteralInteger(self, expr):
        return "{0}".format(str(expr.p))

    def _print_FunctionalFor(self, expr):
        loops = '\n'.join(self._print(i) for i in expr.loops)
        return loops

    def _print_NativeBool(self, expr):
        return 'bool'

    def _print_NativeInteger(self, expr):
        return 'int'

    def _print_NativeReal(self, expr):
        return 'float'

    def _print_NativeComplex(self, expr):
        return 'complex'

    def _print_LiteralTrue(self, expr):
        return 'True'

    def _print_LiteralFalse(self, expr):
        return 'False'

    def _print_FunctionCall(self, expr):
        func = expr.funcdef
        f_name = func.name if not expr.interface else expr.interface.name

        args = [a for a in expr.arguments if not isinstance(a, Nil)]
        args = ['{}'.format(self._print(a)) for a in args]
        args = ', '.join(args)

        return '{0}({1})'.format(f_name, args)

    def _print_PythonLen(self, expr):
        var = expr.arg
        idx = 1 if var.order == 'F' else var.rank

        dtype = var.dtype
        if dtype is NativeString():
            return 'len({})'.format(self._print(var))
        elif var.rank == 1:
            return 'len({})'.format(self._print(var))
        else:
            raise NotImplementedError()
#            return 'size({},{},{})'.format(self._print(var), self._print(idx), prec)


    def _print_Deallocate(self, expr):
        return ''

    def _print_Allocate(self, expr):
        # Transpose indices because of Fortran column-major ordering
        shape = expr.shape

        var_code = self._print(expr.variable)
        size_code = ', '.join(self._print(i) for i in shape)
        shape_code = ', '.join(self._print(i) for i in shape)
        dtype_code = self._print(expr.variable.dtype)
        code = ''

        if expr.status == 'unallocated':
            code += '{0} = zeros(({1}), dtype={2})\n'.format(var_code,
                                                             shape_code,
                                                             dtype_code)

        elif expr.status == 'unknown':
            # TODO improve
            code += '{0} = zeros(({1}), dtype={2})\n'.format(var_code,
                                                             shape_code,
                                                             dtype_code)

        else:
            raise NotImplementedError('only expr.status=unallocated/unknown is treated')

#        elif expr.status == 'unknown':
#            code += 'if (allocated({})) then\n'.format(var_code)
#            code += '  if (any(size({}) /= [{}])) then\n'.format(var_code, size_code)
#            code += '    deallocate({})\n'     .format(var_code)
#            code += '    allocate({0}({1}))\n'.format(var_code, shape_code)
#            code += '  end if\n'
#            code += 'else\n'
#            code += '  allocate({0}({1}))\n'.format(var_code, shape_code)
#            code += 'end if\n'
#
#        elif expr.status == 'allocated':
#            code += 'if (any(size({}) /= [{}])) then\n'.format(var_code, size_code)
#            code += '  deallocate({})\n'     .format(var_code)
#            code += '  allocate({0}({1}))\n'.format(var_code, shape_code)
#            code += 'end if\n'

        return code

#==============================================================================
def pycode(expr, **settings):
    """ Converts an expr to a string of Python code
    Parameters
    ==========
    expr : Expr
        A SymPy expression.
    fully_qualified_modules : bool
        Whether or not to write out full module names of functions
        (``math.sin`` vs. ``sin``). default: ``True``.
    Examples
    ========
    >>> from sympy import tan, Symbol
    >>> from sympy.printing.pycode import pycode
    >>> pycode(tan(Symbol('x')) + 1)
    'math.tan(x) + 1'
    """
    settings.pop('parser', None)
    return PythonCodePrinter(settings).doprint(expr)

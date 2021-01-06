#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
This module contains two classes. Basic that provides a python AST and PyccelAstNode which describes each PyccelAstNode
"""

from sympy.core.basic import Basic as sp_Basic

__all__ = ('Basic', 'PyccelAstNode')

#==============================================================================
class PrecisionNode:
    """
    The goal of this class is reprecenting an object for the precision,
    that object is a property in every class that contains the precision attribute,
    and it has its printer method in pyccel.codegen.printing.fcode.py.
    So we can use it as a simple way to print the pricision instead of
    printing the precision by iso_c_binding dictionary in pyccel.ast.datatypes.
    """
    def __init__(self, dtype, precision):
        if not isinstance(precision, int):
            raise TypeError('PrecisionNode: precision must be an integer object.')
        self._precision = precision
        self._dtype = dtype

    @property
    def precision(self):
        """
        precision of PrecisionNode
        """
        return self._precision

    @property
    def dtype(self):
        """
        dtype of PrecisionNode
        """
        return self._dtype

    @precision.setter
    def set_precision(self, precision):
        self._precision = precision
    
    @dtype.setter
    def set_dtype(self, precision):
        self._dtype = precision

    def __gt__(self, other):
        return self.precision > other.precision
    def __lt__(self, other):
        return self.precision < other.precision
#==============================================================================
class Basic(sp_Basic):
    """Basic class for Pyccel AST."""
    _fst = None

    def set_fst(self, fst):
        """Sets the python.ast fst."""
        self._fst = fst

    @property
    def fst(self):
        return self._fst

class PyccelAstNode:
    stage           = None
    _shape          = None
    _rank           = None
    _dtype          = None
    _precision      = None
    _order          = None
    
    def __init__(self, dtype=None, precision=None, rank=None, shape=None, order=None):
        self._dtype = dtype
        if isinstance(precision, PrecisionNode):
            self._precision = precision
        else:
            self._precision = PrecisionNode(dtype, precision)
        self._rank = rank
        self._shape = shape
        self._order = order

    @property
    def shape(self):
        return self._shape

    @property
    def rank(self):
        return self._rank

    @property
    def dtype(self):
        return self._dtype

    @property
    def precision(self):
        return self._precision

    @property
    def order(self):
        return self._order

    def copy_attributes(self, x):
        self._shape     = x.shape
        self._rank      = x.rank
        self._dtype     = x.dtype
        self._precision = x.precision
        self._order     = x.order


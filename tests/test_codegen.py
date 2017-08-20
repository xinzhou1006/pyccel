# coding: utf-8

"""
"""

from symcc.printers import fcode

from pyccel.syntax import ( \
                           # statements
                           DeclarationStmt, \
                           DelStmt, \
                           PassStmt, \
                           AssignStmt, \
                           IfStmt, ImportFromStmt, ForStmt, FunctionDefStmt)


# ...
def test_Assign():
    from test_parser import test_Assign as test
    ast = test()

    for stmt in ast.statements:
        if isinstance(stmt, DeclarationStmt):
            decs = stmt.expr
            for dec in decs:
                code = fcode(dec)
                print code
        if isinstance(stmt, AssignStmt):
            code = fcode(stmt.expr)
            print code
# ...

# ...
def test_Declare():
    from test_parser import test_Declare as test
    ast = test()

    for stmt in ast.statements:
        if isinstance(stmt, DeclarationStmt):
            decs = stmt.expr
            for dec in decs:
                code = fcode(dec)
                print code
# ...

# ...
def test_Del():
    from test_parser import test_Del as test
    ast = test()
# ...

# ...
def test_Flow():
    from test_parser import test_Flow as test
    ast = test()
# ...

# ...
def test_For():
    from test_parser import test_For as test
    ast = test()

    for stmt in ast.statements:
        if isinstance(stmt, DeclarationStmt):
            decs = stmt.expr
            for dec in decs:
                code = fcode(dec)
                print code
        if isinstance(stmt, ForStmt):
            code    = fcode(stmt.expr)

            prelude = ""
            for s in stmt.statements:
                prelude += fcode(s) + "\n"
            print prelude
            print code
# ...

# ...
def test_FunctionDef():
    from test_parser import test_FunctionDef as test
    ast = test()

    for stmt in ast.statements:
        if isinstance(stmt, FunctionDefStmt):
            code    = fcode(stmt.expr)

            prelude = ""
            for s in stmt.statements:
                prelude += fcode(s) + "\n"
#            print prelude
            print code
# ...

# ...
def test_If():
    from test_parser import test_If as test
    ast = test()

    for stmt in ast.statements:
        if isinstance(stmt, DeclarationStmt):
            decs = stmt.expr
            for dec in decs:
                code = fcode(dec)
                print code
        if isinstance(stmt, IfStmt):
            code = fcode(stmt.expr)
            print code
# ...

# ...
def test_Import():
    from test_parser import test_Import as test
    ast = test()

    for stmt in ast.statements:
        if isinstance(stmt, ImportFromStmt):
            code = fcode(stmt.expr)
            print code
# ...

# ...
def test_Pass():
    from test_parser import test_Pass as test
    ast = test()
# ...


######################################
if __name__ == "__main__":
#    test_Assign()
#    test_Declare()
#    test_Del()
#    test_Flow()
#    test_For()
#    test_FunctionDef()
#    test_If()
    test_Import()
#    test_Pass()

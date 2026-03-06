import ast
import operator
import math

class SafeMathREPL:
    """
    AST-based secure execution environment for financial calculations.
    Blocks all system calls, imports, and non-mathematical operations.
    """
    def __init__(self):
        self.allowed_operators = {
            ast.Add: operator.add, 
            ast.Sub: operator.sub, 
            ast.Mult: operator.mul, 
            ast.Div: operator.truediv, 
            ast.Pow: operator.pow, 
            ast.USub: operator.neg,
            ast.Mod: operator.mod
        }
        self.allowed_functions = {
            "math": math, 
            "min": min, 
            "max": max, 
            "round": round,
            "abs": abs
        }

    def _eval_node(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Invalid constant type: {type(node.value)}")
            
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.allowed_operators[type(node.op)](left, right)
            
        elif isinstance(node, ast.UnaryOp):
            return self.allowed_operators[type(node.op)](self._eval_node(node.operand))
            
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name in self.allowed_functions:
                args = [self._eval_node(arg) for arg in node.args]
                return self.allowed_functions[func_name](*args)
            raise ValueError(f"Function '{func_name}' is not permitted.")
            
        raise ValueError(f"Unsupported syntax tree node: {ast.dump(node)}")

    def execute(self, expression: str) -> str:
        try:
            tree = ast.parse(expression.strip(), mode='eval')
            result = self._eval_node(tree.body)
            return f"Calculation Result: {round(result, 4)}"
        except Exception as e:
            return f"Math Execution Error: {str(e)}"

finance_calculator = SafeMathREPL()
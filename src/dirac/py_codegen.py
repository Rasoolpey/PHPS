"""Translate C++ DAE kernel snippets to Python callables.

Handles the specific patterns emitted by DiracCompiler for component step and
output functions: variable declarations, if/else chains, ternary operators,
and C standard math functions.
"""

from __future__ import annotations

import math
import re
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Math function substitutions (C → Python)
# ---------------------------------------------------------------------------
_MATH_SUBS = [
    (re.compile(r'\bsqrt\b'), 'math.sqrt'),
    (re.compile(r'\bsin\b'),  'math.sin'),
    (re.compile(r'\bcos\b'),  'math.cos'),
    (re.compile(r'\btan\b'),  'math.tan'),
    (re.compile(r'\basin\b'), 'math.asin'),
    (re.compile(r'\bacos\b'), 'math.acos'),
    (re.compile(r'\batan\b'), 'math.atan'),
    (re.compile(r'\batan2\b'), 'math.atan2'),
    (re.compile(r'\bexp\b'),  'math.exp'),
    (re.compile(r'\blog\b'),  'math.log'),
    (re.compile(r'\bfabs\b'), 'abs'),
    (re.compile(r'\bfmin\b'), 'min'),
    (re.compile(r'\bfmax\b'), 'max'),
    (re.compile(r'\bpow\b'),  '**_pow_placeholder'),  # handled separately
]

_DECL_RE = re.compile(
    r'^(?:const\s+)?(?:double|float|int|long)\s+(\w+)\s*(?:\[(\d+)\])?\s*(?:=\s*(.+))?$'
)


def _find_matching_paren(s: str, start: int) -> int:
    """Return index of closing ')' matching the '(' at position *start*."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return len(s) - 1


def _translate_condition(cond: str) -> str:
    cond = re.sub(r'&&', ' and ', cond)
    cond = re.sub(r'\|\|', ' or ', cond)
    # !x but not !=
    cond = re.sub(r'!(?!=)', 'not ', cond)
    return cond.strip()


def _translate_expr(expr: str) -> str:
    """Translate a C++ expression to Python."""
    expr = expr.strip()

    # Ternary: (cond) ? a : b
    # Must be done before other substitutions because ? and : are C++ syntax
    # Strategy: find the ? and : at the top level (not inside parens)
    q_pos = None
    depth = 0
    for i, ch in enumerate(expr):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '?' and depth == 0:
            q_pos = i
            break

    if q_pos is not None:
        # Find matching : after the ?
        depth = 0
        c_pos = None
        for i in range(q_pos + 1, len(expr)):
            ch = expr[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == ':' and depth == 0:
                c_pos = i
                break
        if c_pos is not None:
            cond_part = expr[:q_pos].strip()
            # Remove outer parens from condition if present
            if cond_part.startswith('(') and cond_part.endswith(')'):
                cond_part = cond_part[1:-1]
            yes_part = expr[q_pos + 1:c_pos].strip()
            no_part  = expr[c_pos + 1:].strip()
            cond_py = _translate_condition(cond_part)
            yes_py  = _translate_expr(yes_part)
            no_py   = _translate_expr(no_part)
            return f'({yes_py} if ({cond_py}) else {no_py})'

    # M_PI
    expr = expr.replace('M_PI', '3.141592653589793')

    # Math function substitutions
    for pat, repl in _MATH_SUBS:
        expr = pat.sub(repl, expr)

    # pow(a, b) → a**b  (simple case: no nested calls)
    expr = re.sub(r'\*\*_pow_placeholder\(([^,]+),\s*([^)]+)\)', r'(\1)**(\2)', expr)
    # Fallback: pow(...) not yet handled → replace with **
    # This shouldn't appear but guard anyway

    # C++ casts: (int)expr → int(expr), (double)expr → float(expr)
    expr = re.sub(r'\(int\)\s*(\w+)', r'int(\1)', expr)
    expr = re.sub(r'\(double\)\s*(\w+)', r'float(\1)', expr)
    expr = re.sub(r'\(float\)\s*(\w+)', r'float(\1)', expr)

    # Boolean operators
    expr = re.sub(r'&&', ' and ', expr)
    expr = re.sub(r'\|\|', ' or ', expr)

    return expr


def _translate_statement(stmt: str) -> Optional[str]:
    """Translate a single C++ statement to Python. Returns None to skip."""
    stmt = stmt.strip()
    if not stmt:
        return None

    # Remove trailing semicolon
    if stmt.endswith(';'):
        stmt = stmt[:-1].strip()

    if not stmt:
        return None

    # Variable declaration: [const] double/int/float X [= expr]
    m = _DECL_RE.match(stmt)
    if m:
        var_name = m.group(1)
        arr_size = m.group(2)
        initializer = m.group(3)

        if arr_size is not None:
            # Array declaration: double arr[N] = {0};
            size = int(arr_size)
            return f'{var_name} = [0.0] * {size}'

        if initializer is not None:
            rhs = _translate_expr(initializer.strip())
            return f'{var_name} = {rhs}'
        else:
            # Uninitialized declaration
            return f'{var_name} = 0.0'

    # Regular expression statement (assignment, function call, etc.)
    return _translate_expr(stmt)


def _strip_inline_comment(line: str) -> Tuple[str, str]:
    """Strip C++ inline comment from end of line.

    Returns (code_part, comment_python) where comment_python is '' or '  # ...'.
    Safe because C++ kernels never contain // inside strings.
    """
    if '//' not in line:
        return line, ''
    pos = line.index('//')
    code = line[:pos].strip()
    comment = line[pos + 2:].strip()
    return code, (f'  # {comment}' if comment else '')


def translate_cpp_kernel(cpp_code: str) -> str:
    """Translate a C++ kernel body to equivalent Python code.

    The generated C++ kernels are flat blocks of:
      - variable declarations
      - arithmetic assignments
      - if / else if / else (both one-liner and block form)
      - ternary expressions
      - standard math function calls

    Parameters
    ----------
    cpp_code : str
        Raw C++ kernel text (body only — no function signature).

    Returns
    -------
    str
        Python code string.
    """
    # ------------------------------------------------------------------
    # Pre-process: join multi-line C++ statements.
    # A C++ statement is not complete until it ends with ';', '{', or '}'.
    # Continuation lines (e.g. multi-line dxdt assignments) must be merged
    # before the per-line translator can handle them.
    # ------------------------------------------------------------------
    raw_lines = cpp_code.split('\n')
    lines: List[str] = []
    accum = ''
    for raw in raw_lines:
        stripped = raw.strip()
        # Skip blank lines — flush accumulator first
        if not stripped:
            if accum:
                lines.append(accum)
                accum = ''
            lines.append('')
            continue
        # Pure comment lines stand alone
        if stripped.startswith('//'):
            if accum:
                lines.append(accum)
                accum = ''
            lines.append(raw)
            continue
        # Strip inline comment for analysis but keep the raw text
        code_part, _ = _strip_inline_comment(stripped)
        if not code_part:
            if accum:
                lines.append(accum)
                accum = ''
            lines.append(raw)
            continue

        if accum:
            # We are continuing a previous incomplete statement.
            # Append this line's code to the accumulator.
            accum = accum + ' ' + code_part
        else:
            accum = code_part

        # Check if statement is now complete
        if (accum.rstrip().endswith(';') or accum.rstrip().endswith('{')
                or accum.rstrip() in ('}', '};', '} ;')):
            lines.append(accum)
            accum = ''
    if accum:
        lines.append(accum)

    result: List[str] = []
    indent = 0
    IND = '    '

    def emit(s: str) -> None:
        result.append(IND * indent + s)

    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        i += 1

        if not stripped:
            result.append('')
            continue

        # Pure comment line
        if stripped.startswith('//'):
            emit('# ' + stripped[2:].strip())
            continue

        # Strip trailing inline comment
        code, inline_cmt = _strip_inline_comment(stripped)
        if not code:
            # Was a pure comment
            emit('# ' + stripped[2:].strip())
            continue

        # ---- Closing brace ----
        if code in ('}', '};', '} ;'):
            indent = max(0, indent - 1)
            continue

        # ---- } else if (...) { ----
        m = re.match(r'^}\s*else\s+if\s*\(', code)
        if m:
            # Find matching paren
            paren_start = code.index('(', m.end() - 1)
            paren_end = _find_matching_paren(code, paren_start)
            cond = code[paren_start + 1:paren_end]
            indent = max(0, indent - 1)
            emit(f'elif {_translate_condition(cond)}:{inline_cmt}')
            indent += 1
            continue

        # ---- } else { ----
        if re.match(r'^}\s*else\s*\{?$', code):
            indent = max(0, indent - 1)
            emit(f'else:{inline_cmt}')
            indent += 1
            continue

        # ---- if (...) { or if (...) single_stmt ----
        if re.match(r'^if\s*\(', code):
            paren_start = code.index('(')
            paren_end = _find_matching_paren(code, paren_start)
            cond = code[paren_start + 1:paren_end]
            rest = code[paren_end + 1:].strip()

            if rest in ('{', '') or rest == '{' + '':
                # Block form
                emit(f'if {_translate_condition(cond)}:{inline_cmt}')
                indent += 1
            elif rest == '{' or rest.endswith('{'):
                emit(f'if {_translate_condition(cond)}:{inline_cmt}')
                indent += 1
            else:
                # One-liner: if (cond) stmt
                if rest.endswith('{'):
                    # Edge case: if (cond) { — treat as block
                    emit(f'if {_translate_condition(cond)}:{inline_cmt}')
                    indent += 1
                else:
                    stmt_py = _translate_statement(rest) or ''
                    emit(f'if {_translate_condition(cond)}:{inline_cmt}')
                    indent += 1
                    emit(stmt_py)
                    indent -= 1
            continue

        # ---- else if (...) { ----
        m = re.match(r'^else\s+if\s*\(', code)
        if m:
            paren_start = code.index('(')
            paren_end = _find_matching_paren(code, paren_start)
            cond = code[paren_start + 1:paren_end]
            rest = code[paren_end + 1:].strip()
            emit(f'elif {_translate_condition(cond)}:{inline_cmt}')
            if rest == '{' or not rest:
                indent += 1
            else:
                indent += 1
                stmt_py = _translate_statement(rest) or ''
                emit(stmt_py)
                indent -= 1
            continue

        # ---- else { or else single_stmt ----
        if re.match(r'^else\b', code):
            rest = re.sub(r'^else\s*', '', code).strip()
            emit(f'else:{inline_cmt}')
            if rest in ('{', ''):
                indent += 1
            else:
                indent += 1
                stmt_py = _translate_statement(rest) or ''
                emit(stmt_py)
                indent -= 1
            continue

        # ---- for loop ----
        m = re.match(
            r'^for\s*\(\s*int\s+(\w+)\s*=\s*(\d+)\s*;\s*\1\s*<\s*(\w+)\s*;\s*\+\+\1\s*\)\s*\{?$',
            code
        )
        if m:
            var, start, end = m.group(1), m.group(2), m.group(3)
            emit(f'for {var} in range({start}, {end}):{inline_cmt}')
            indent += 1
            continue

        # ---- Opening brace alone (shouldn't happen in kernels but guard) ----
        if code == '{':
            indent += 1
            continue

        # ---- Regular statement ----
        py_stmt = _translate_statement(code)
        if py_stmt is not None:
            emit(py_stmt + inline_cmt)

    return '\n'.join(result)


# ---------------------------------------------------------------------------
# High-level: build Python callable from C++ component kernel
# ---------------------------------------------------------------------------

def _build_kernel_code(comp, mode: str) -> str:
    """Return C++ kernel with parameter constants prepended.

    mode: 'step' for dynamics, 'out' for outputs.
    """
    param_lines = []
    for p_name, p_val in comp.params.items():
        if p_name not in comp.param_schema:
            continue
        if isinstance(p_val, (int, float)) and not math.isnan(float(p_val)):
            param_lines.append(f'const double {p_name} = {float(p_val)};')
        elif isinstance(p_val, str) and ',' not in p_val:
            # Single numeric expression
            try:
                val = float(eval(p_val, {'M_PI': math.pi}))
                param_lines.append(f'const double {p_name} = {val};')
            except Exception:
                pass

    if mode == 'step':
        kernel = comp.get_cpp_step_code()
    else:
        kernel = comp.get_cpp_compute_outputs_code()

    return '\n'.join(param_lines) + '\n' + kernel


def make_step_func(comp):
    """Return a Python callable ``step(x, dxdt, inputs, outputs, t)``
    that implements the component's dynamics kernel.

    The returned function modifies *dxdt* in-place (numpy array slice).
    """
    n_st = len(comp.state_schema)
    n_in = len(comp.port_schema['in'])
    n_out = len(comp.port_schema['out'])

    cpp_code = _build_kernel_code(comp, 'step')
    py_body = translate_cpp_kernel(cpp_code)

    func_src = (
        'import math as math\n'
        'import numpy as _np\n'
        f'def _step(x, dxdt, inputs, outputs, t):\n'
    )
    # Indent the translated body
    for line in py_body.split('\n'):
        func_src += '    ' + line + '\n'

    ns: dict = {}
    try:
        exec(compile(func_src, f'<step_{comp.name}>', 'exec'), ns)
    except SyntaxError as exc:
        raise RuntimeError(
            f'Python codegen failed for step_{comp.name}:\n'
            f'{func_src}\nError: {exc}'
        )
    return ns['_step']


def make_out_func(comp):
    """Return a Python callable ``out(x, inputs, outputs, t)``
    that implements the component's output kernel.

    The returned function modifies *outputs* in-place.
    """
    try:
        cpp_code = _build_kernel_code(comp, 'out')
    except NotImplementedError:
        # Component has no output function
        def _noop(x, inputs, outputs, t):
            pass
        return _noop

    py_body = translate_cpp_kernel(cpp_code)

    func_src = (
        'import math as math\n'
        'import numpy as _np\n'
        f'def _out(x, inputs, outputs, t):\n'
    )
    for line in py_body.split('\n'):
        func_src += '    ' + line + '\n'

    ns: dict = {}
    try:
        exec(compile(func_src, f'<out_{comp.name}>', 'exec'), ns)
    except SyntaxError as exc:
        raise RuntimeError(
            f'Python codegen failed for out_{comp.name}:\n'
            f'{func_src}\nError: {exc}'
        )
    return ns['_out']

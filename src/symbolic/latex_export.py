"""
LaTeX generation from SymbolicPHS objects.

Exports PHS structure matrices, Hamiltonian, gradient, and dynamics
as publication-ready LaTeX snippets or full standalone .tex documents.
"""

import re
import sympy as sp
from sympy import latex as _sp_latex_raw
from typing import List, Optional, Dict
from src.symbolic.core import SymbolicPHS


# Pattern: a prime/double-prime symbol immediately followed by ^{...}
# e.g.  E'_{q}^{2}  →  {E'_{q}}^{2}
#        x''_{d}^{3} →  {x''_{d}}^{3}
_DOUBLE_SUP_RE = re.compile(
    r"([A-Za-z](?:'{1,2})(?:_\{[^}]*\})?)\^"
)


def sp_latex(expr) -> str:
    """SymPy latex() with double-superscript fix for primed symbols."""
    raw = _sp_latex_raw(expr)
    return _DOUBLE_SUP_RE.sub(r"{\1}^", raw)


def _boxed_eq(inner: str) -> str:
    """Wrap equation content in adjustbox so it shrinks to fit the margins."""
    return (r"\adjustbox{max width=\linewidth}{$\displaystyle "
            + inner + "$}")


def phs_to_latex(sphs: SymbolicPHS, include_dynamics: bool = True) -> str:
    """Generate a LaTeX snippet for a single SymbolicPHS component.

    Parameters
    ----------
    sphs : SymbolicPHS
        The symbolic PHS to export.
    include_dynamics : bool
        If True, include the expanded dynamics ẋ = (J−R)∇H + g·u.

    Returns
    -------
    str
        LaTeX string (suitable for \\input{} in a document).
    """
    lines = []
    name_tex = sphs.name.replace("_", r"\_")
    lines.append(f"\\subsection{{{name_tex}}}")

    if sphs.description:
        lines.append(f"\n{sphs.description}\n")

    # --- States and inputs ---
    states_tex = ", ".join(sp_latex(s) for s in sphs.states)
    inputs_tex = ", ".join(sp_latex(u) for u in sphs.inputs)
    lines.append("\\paragraph{State variables}")
    lines.append(f"$\\mathbf{{x}} = [{states_tex}]^\\top$, "
                 f"$n = {sphs.n_states}$\n")
    lines.append("\\paragraph{Port inputs}")
    lines.append(f"$\\mathbf{{u}} = [{inputs_tex}]^\\top$, "
                 f"$m = {sphs.n_inputs}$\n")

    # --- Hamiltonian ---
    h_tex = sp_latex(sphs.H)
    lines.append("\\paragraph{Hamiltonian}")
    lines.append("\\begin{equation}")
    lines.append(f"  {_boxed_eq('H(\\mathbf{x}) = ' + h_tex)}"
                 if len(h_tex) > 80
                 else f"  H(\\mathbf{{x}}) = {h_tex}")
    lines.append("\\end{equation}\n")

    # --- Gradient ---
    grad_entries = [sp_latex(sphs.grad_H[i]) for i in range(sphs.n_states)]
    grad_inner = ("\\nabla H = \\begin{bmatrix} "
                  + " \\\\ ".join(grad_entries)
                  + " \\end{bmatrix}")
    lines.append("\\paragraph{Gradient}")
    lines.append("\\begin{equation}")
    lines.append(f"  {_boxed_eq(grad_inner)}"
                 if sphs.n_states > 4
                 else f"  {grad_inner}")
    lines.append("\\end{equation}\n")

    # --- J matrix ---
    lines.append("\\paragraph{Interconnection matrix $J$ (skew-symmetric)}")
    lines.append("\\begin{equation}")
    j_inner = f"J = {_matrix_to_latex(sphs.J)}"
    lines.append(f"  {_boxed_eq(j_inner)}"
                 if sphs.n_states > 4
                 else f"  {j_inner}")
    lines.append("\\end{equation}\n")

    # --- R matrix ---
    lines.append("\\paragraph{Dissipation matrix $R$ (positive semi-definite)}")
    lines.append("\\begin{equation}")
    r_inner = f"R = {_matrix_to_latex(sphs.R)}"
    lines.append(f"  {_boxed_eq(r_inner)}"
                 if sphs.n_states > 4
                 else f"  {r_inner}")
    lines.append("\\end{equation}\n")

    # --- g matrix ---
    lines.append("\\paragraph{Input coupling matrix $g$}")
    lines.append("\\begin{equation}")
    lines.append(f"  g = {_matrix_to_latex(sphs.g)}")
    lines.append("\\end{equation}\n")

    # --- Q matrix (only if non-identity) ---
    if not sphs.Q.equals(sp.eye(sphs.n_states)):
        lines.append("\\paragraph{Energy-variable transformation $Q$}")
        lines.append("\\begin{equation}")
        lines.append(f"  Q = {_matrix_to_latex(sphs.Q)}")
        lines.append("\\end{equation}\n")

    # --- PHS dynamics ---
    lines.append("\\paragraph{Port-Hamiltonian dynamics}")
    lines.append("\\begin{equation}")
    lines.append(r"  \dot{\mathbf{x}} = (J - R)\, Q\, "
                 r"\nabla H(\mathbf{x}) + g\, \mathbf{u}")
    lines.append("\\end{equation}\n")

    if include_dynamics:
        lines.append("\\paragraph{Expanded dynamics}")
        dyn = sphs.dynamics
        dyn_lines = []
        for i, xi in enumerate(sphs.states):
            lhs = f"\\dot{{{sp_latex(xi)}}}"
            rhs = sp_latex(sp.simplify(dyn[i]))
            sep = " \\\\" if i < sphs.n_states - 1 else ""
            dyn_lines.append(f"  {lhs} &= {rhs}{sep}")
        align_body = "\n".join(dyn_lines)
        # Wrap in adjustbox to prevent overflow
        lines.append(r"\begin{equation*}")
        lines.append(r"\adjustbox{max width=\linewidth}{$\displaystyle")
        lines.append(r"\begin{aligned}")
        lines.append(align_body)
        lines.append(r"\end{aligned}")
        lines.append("$}")
        lines.append("\\end{equation*}\n")

    # --- Power balance ---
    lines.append("\\paragraph{Power balance (passivity)}")
    lines.append("\\begin{equation}")
    lines.append(r"  \dot{H} = \underbrace{-\nabla H^\top R\, \nabla H}"
                 r"_{\text{dissipation}} + "
                 r"\underbrace{\nabla H^\top g\, \mathbf{u}}"
                 r"_{\text{supply rate}} \leq "
                 r"\nabla H^\top g\, \mathbf{u}")
    lines.append("\\end{equation}\n")

    # Show explicit dissipation expression
    diss = sp.simplify(sphs.dissipation_rate)
    diss_tex = sp_latex(diss)
    lines.append("Dissipation rate:")
    lines.append("\\begin{equation}")
    lines.append(f"  {_boxed_eq(r'-\nabla H^\top R\, \nabla H = ' + diss_tex)}")
    lines.append("\\end{equation}\n")

    return "\n".join(lines)


def phs_collection_to_tex_document(
    components: List[SymbolicPHS],
    title: str = "Port-Hamiltonian System Models",
    author: str = "",
    output_path: Optional[str] = None,
) -> str:
    """Generate a complete standalone LaTeX document from multiple PHS components.

    Parameters
    ----------
    components : list of SymbolicPHS
        PHS models to include.
    title, author : str
        Document metadata.
    output_path : str, optional
        If given, write the .tex file to this path.

    Returns
    -------
    str
        Complete .tex document content.
    """
    doc = []
    doc.append(r"\documentclass[11pt,a4paper]{article}")
    doc.append(r"\usepackage{amsmath,amssymb,amsfonts}")
    doc.append(r"\usepackage{graphicx}")
    doc.append(r"\usepackage[export]{adjustbox}")
    doc.append(r"\usepackage[margin=2.5cm]{geometry}")
    doc.append(r"\usepackage{booktabs}")
    doc.append(f"\\title{{{title}}}")
    if author:
        doc.append(f"\\author{{{author}}}")
    doc.append(r"\date{\today}")
    doc.append("")
    doc.append(r"\begin{document}")
    doc.append(r"\maketitle")
    doc.append("")
    doc.append(r"\section{Port-Hamiltonian Component Models}")
    doc.append("")
    doc.append(
        r"Each component is described by its Port-Hamiltonian structure "
        r"$\dot{\mathbf{x}} = (J - R)\, Q\, \nabla H(\mathbf{x}) + g\, \mathbf{u}$, "
        r"where $J = -J^\top$ is the skew-symmetric interconnection matrix, "
        r"$R = R^\top \geq 0$ is the dissipation matrix, $H(\mathbf{x})$ is the "
        r"Hamiltonian (stored energy), and $g$ is the port coupling matrix."
    )
    doc.append("")

    items = components.values() if isinstance(components, dict) else components
    for sphs in items:
        doc.append(phs_to_latex(sphs))
        doc.append("")

    doc.append(r"\end{document}")

    content = "\n".join(doc)

    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(content)

    return content


def _matrix_to_latex(M: sp.Matrix) -> str:
    """Convert a SymPy Matrix to a LaTeX bmatrix string."""
    rows, cols = M.shape
    lines = []
    lines.append("\\begin{bmatrix}")
    for i in range(rows):
        entries = []
        for j in range(cols):
            entries.append(sp_latex(M[i, j]))
        sep = " \\\\" if i < rows - 1 else ""
        lines.append("  " + " & ".join(entries) + sep)
    lines.append("\\end{bmatrix}")
    return "\n".join(lines)

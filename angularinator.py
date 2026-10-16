#!/usr/bin/env python3
"""
angularinator.py -- angular reduction of Goldstone diagrams (goldstoninator.py), with C++ output for
ampsci, in the conventions of the Methods notes: Q^k, W^k = Q^k + P^k, and reduced matrix elements
<a m_a|t^k_q|b m_b> = (-1)^{j_a-m_a} (j_a k j_b; -m_a q m_b) <a||t^k||b>.

angular(term) reduces a term, or a sum of terms, to Q^k, W^k (from g~, and from each pair of
diagrams that differ by the exchange of the in-letters of one Coulomb integral: QW rather than WQ),
reduced matrix elements <p||t||r> (in the order written), 6j symbols, phases and [j] = 2j+1 factors
(letters stand for their j in these).  The angular part is done by building the network of 3j symbols
of the numerator (sum_k A^k_{pqrs} X^k_{pqrs} for each two-body integral, Wigner-Eckart for each
one-body one), closing it as the external lines require, and reducing it with the graphical rules
(loops, zero lines, bubbles, triangles, interchanges on longer cycles), whose phases were fitted
numerically and are checked against direct m-sums; three 6j symbols summed over an intermediate are
written as a 9j where they form one.  A delta_{ab} in the result constrains fixed symbols (letters'
j, ranks, k); the identifications of multipolarities collapse their sums silently.  ampsci_fullmsum(term) writes the same quantity
as a brute-force sum over every m, using only Coulomb::g_abcd and TensorOperator::fullME: the
independent check of the reduced form.

    import angularinator as ang
    r = ang.angular([r"g_vamn \\tilde g_nmav", r"g_vnab \\tilde g_banv"])
    r.tex(); r.equation()                          # LaTeX; typeset in Jupyter (macros=True: \\sixj)
    print(r.ampsci("de2"))                         # ampsci C++ of the reduced form
    print(ang.ampsci_fullmsum([...], name="de2_fullmsum"))   # the same, by brute force over every m
    gd.grid([...]).angular(); gd.diagram(...).ampsci(); fd.diagram(...).ampsci_fullmsum()

What the function returns, chosen from the open lines unless external= says otherwise:
    'average'  one valence line, no external field: the m-independent value, averaged,
               (1/[j_v]) sum_m <w m|X|v m>  (zero unless j_w = j_v)
    'reduced'  one valence line, one field t of rank k: the reduced matrix element <w||X^k||v>,
               projected with t.rme3js (Wigner-Eckart as above)
    'Ak'       two valence lines, no field: S^k in X = sum_k A^k_{xyvw} S^k_{xyvw}, the incoming letters
               v, w paired with the outgoing x, y in the order of the valence letters (as in
               goldstoninator), S^k = [k] sum_m A^k_{xyvw}(m) X(m); k is an argument
    'm'        any: the m-resolved value; the 2m of every external leg and the 2q of every field are
               arguments (tmi_v for the incoming leg v, tmo_w for the outgoing leg w, tq_t for field t).
               The reduced form (one valence line) is X_{j_1...} in the chain basis along the valence
               line, X = sum_{j_1..} prod_i (-1)^{j_{i-1}-m_{i-1}} (j_{i-1} k_i j_i; -m_{i-1} q_i m_i)
               X_{j_1..} (j_0 = w, j_n = v, the fields in the order written)
    'coupled'  one valence line, two fields T, S (in the order written): X_K in
               X = sum_K B_K X_K, B_K = sum_Q <k_T q_T k_S q_S|K Q> (-1)^{w-m_w} (w K v; -m_w Q m_v)
               (so X_K = sum_m B_K X); K is an argument
A field is a one-body vertex, unless ranks[name] == 0 (ranks={'V': 0} makes V a scalar operator);
ranks={'t': 'k'} (or r'\\kappa') names the rank of t (k by default for one field, k_t for several).
Arguments: the valence states (incoming, then outgoing), core and excited (the states the internal
letters run over), the QkTable (for the reduced form; the full-m one ignores it), one
TensorOperator per one-body name, one double per energy omega in the denominators, then k or the m's.
Two-body integrals must be g; one-body ones written t_{out,in}.  The signs and denominators are
goldstoninator's (keywords go to it: types, core, exc, val, omega).
No dependencies beyond the standard library (and goldstoninator.py).
"""

import itertools
import math
import re
import textwrap
from fractions import Fraction
from functools import lru_cache

import goldstoninator as gd

EXTERNALS = ("average", "reduced", "Ak", "m", "coupled")
INCLUDES = """#include "Angular/Wigner369j.hpp"
#include "Coulomb/CoulombIntegrals.hpp"
#include "Coulomb/QkTable.hpp"
#include "DiracOperator/TensorOperator.hpp"
#include "Wavefunction/DiracSpinor.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>
"""
_AK = """// A^k_abcd(m) of the Methods notes, g_abcd = sum_k A^k_abcd Q^k_abcd:
// (-1)^{k+m_a-m_b} sum_q (-1)^q (j_a k j_c; -m_a -q m_c)(j_b k j_d; -m_b q m_d)
const auto Ak = [](int kk, const DiracSpinor &a, const DiracSpinor &b,
                   const DiracSpinor &c, const DiracSpinor &d, int tma, int tmb,
                   int tmc, int tmd) {
  const int twoq = tmc - tma;
  if (tmb - tmd != twoq)
    return 0.0;
  return Angular::neg1pow_2(2 * kk + tma - tmb + twoq) *
         Angular::threej_2(a.twoj(), 2 * kk, c.twoj(), -tma, -twoq, tmc) *
         Angular::threej_2(b.twoj(), 2 * kk, d.twoj(), -tmb, twoq, tmd);
};"""
_QZERO = """// true if every Q^k_abcd vanishes by the angular selection rules (then g_abcd = 0 for all m)
const auto Qzero = [](const DiracSpinor &a, const DiracSpinor &b,
                      const DiracSpinor &c, const DiracSpinor &d) {
  const auto [kmin, kmax] = Coulomb::k_minmax_Q(a, b, c, d);
  return kmin > kmax;
};"""


class Code(str):
    """C++ source: shown as is when it is the output of a Jupyter cell; .save(path) writes it"""

    def __repr__(self):
        return str.__str__(self)

    def save(self, path):
        with open(path, "w") as f:
            f.write(self)
        return path


class Term:
    """one Goldstone diagram as read from a goldstoninator Diagram: sign; factors [(name, labels)] as
    written; roles [('out'|'in', ...)] of the written labels in the diagram; kind {letter: 'core' |
    'exc' | 'val'}; dens [(plus, minus, {omega symbol: coefficient})]; vins, vouts (the incoming and
    outgoing valence letters, paired in order); omegas (the symbol absorbed at each factor, or None)"""

    def __init__(self, d):
        self.dens = d.denominators()  # also sets d.sign
        self.sign = d.sign
        self.text = d._term_str()
        self.tex = d.tex()
        self.factors = [(n, tuple(g)) for n, g in zip(d.names, d.gs)]
        self.kind = {x: d.types[x] for _, g in self.factors for x in g}
        self.vins = [x for x, _ in d.vins]
        self.vouts = [x for x, _ in d.vouts]
        self.omegas = d.omegas()
        if len(set(self.vins)) < len(self.vins) or len(set(self.vouts)) < len(self.vouts):
            raise ValueError("two incoming (or two outgoing) valence lines with the same letter in %s: use "
                             "different letters (and pass the same state twice for equal states)" % self.text)
        self.roles = []
        for (n, g), f in zip(self.factors, d.flips):
            if len(g) == 2 and f:
                raise ValueError(
                    "write %s_{%s} as %s_{%s} (outgoing letter first): %s"
                    % (n, "".join(g), n, "".join(g[::-1]), self.text)
                )
            half = len(g) // 2
            self.roles.append(tuple("out" if (j < half) != bool(f) else "in" for j in range(len(g))))

    def dummies(self):
        """the internal (summed) letters, in order of appearance"""
        out = []
        for _, g in self.factors:
            for x in g:
                if self.kind[x] != "val" and x not in out:
                    out.append(x)
        return out


def terms(term, **kw):
    """the Terms of a term (string or list of integrals, g~ expanded), a list of terms, a goldstoninator
    Diagram or Grid, or a Feynman diagram (feynmanator, all its time orderings)"""
    return [Term(d) for d in _diagrams(term, **kw)]


def _diagrams(term, **kw):
    import feynmanator as fd

    if isinstance(term, gd.Grid):
        return list(term.diagrams)
    if isinstance(term, gd.Diagram):
        return [term]
    if isinstance(term, fd.Diagram):
        out = []
        for t in term.goldstone_terms():
            d = gd.Diagram(t, **kw)
            if d.order != list(range(len(d.verts))):
                raise ValueError("goldstoninator orders %r differently" % t)
            out.append(d)
        return out
    if gd.several(term):
        return [d for t in term for d in _diagrams(t, **kw)]
    if gd.has_tilde(term):
        return gd._expanded(term, layout=False, **kw)
    return [gd.Diagram(term, **kw)]


def _fields(t, ranks):
    """the names of the external fields of a term: one-body vertices that are not scalars"""
    return [n for n, g in t.factors if len(g) == 2 and ranks.get(n) != 0]


def external_of(ts, ranks=None, external=None):
    """(external, fields): how the open lines of the terms are treated (see the module docstring), and
    the names of the fields, checking that every term has the same open lines and fields"""
    ranks = ranks or {}
    for n, r in ranks.items():
        if r == 0:
            continue
        if not (isinstance(r, str) and re.fullmatch(r"\\?[A-Za-z][A-Za-z0-9_]*", r)):
            raise ValueError("ranks[%r] = %r: give 0 (a scalar operator) or a name for its rank, like 'k' "
                             "or r'\\kappa'" % (n, r))
        if _cpp_name(r) in RESERVED or (len(r) == 1 and any(r in t.kind for t in ts)):
            raise ValueError("ranks[%r] = %r clashes with a letter or a name of the generated code" % (n, r))
    named = [str(r) for r in ranks.values() if r != 0]
    if len({tex_sym(r) for r in named}) < len(named) or len({_cpp_name(r) for r in named}) < len(named):
        raise ValueError("the ranks of different fields need different names: %s" % named)
    t0 = ts[0]
    fields = _fields(t0, ranks)
    for t in ts[1:]:
        if (t.vins, t.vouts) != (t0.vins, t0.vouts):
            raise ValueError("the terms have different valence lines: %s and %s" % (t0.text, t.text))
        if sorted(_fields(t, ranks)) != sorted(fields):
            raise ValueError("the terms have different fields: %s and %s" % (t0.text, t.text))
    nval = len(t0.vins)
    if external is None:
        if nval == 1:
            external = ("average", "reduced", "m")[min(len(fields), 2)]
        else:
            external = "m" if fields else "Ak"
    if external not in EXTERNALS:
        raise ValueError("external must be one of %s, not %r" % (EXTERNALS, external))
    need = {"average": (1, 0), "reduced": (1, 1), "Ak": (2, 0), "coupled": (1, 2)}.get(external)
    if need and need != (nval, len(fields)):
        raise ValueError(
            "external=%r needs %d valence line(s) and %d field(s), the terms have %d and %d"
            % (external, need[0], need[1], nval, len(fields))
        )
    if external in ("m", "coupled") and len(set(fields)) != len(fields):
        raise ValueError("a field name appears twice in a term: give the vertices different names")
    for n in fields:
        if str(ranks.get(n, "")) and _cpp_name(ranks.get(n, "")) in ARGS.get(external, ()):
            raise ValueError("ranks[%r] = %r: %s is the argument of external=%r" % (n, ranks[n], ranks[n], external))
    return external, fields


def _omega_var(sym):
    """the C++ name of an energy: \\omega_t -> omega_t, \\omega_{T} -> omega_T, \\omega' -> omegap"""
    v = sym.replace("\\", "").replace("'", "p").replace("+", "plus").replace("-", "minus")
    v = re.sub(r"[^A-Za-z0-9]+", "_", v).strip("_") or "omega"
    return v if v[0].isalpha() else "omega_" + v


def _mvar(t, x, role):
    """the C++ name of the 2m of letter x at a slot of the given role"""
    if t.kind[x] != "val":
        return "tm_" + x
    return ("tmi_" if role == "in" else "tmo_") + x


def _den_cpp(plus, minus, om):
    out = []
    for c, s in [(1, "F%s.en()" % x) for x in plus] + [(-1, "F%s.en()" % x) for x in minus] + [
        (c, _omega_var(s)) for s, c in om.items()
    ]:
        a = abs(c)
        s = s if a == 1 else "%d.0 * %s" % (a, s)
        out.append(("- " if c < 0 else "+ ") + s)
    out = " ".join(out)
    return out[2:] if out.startswith("+ ") else "-" + out[2:]


def ampsci_fullmsum(term, name="term_fullmsum", external=None, ranks=None, includes=True, **kw):
    """C++ (for ampsci) of a function evaluating a term, or the sum of several, by brute force over
    every magnetic quantum number, from Coulomb::g_abcd and TensorOperator::fullME only; term as for
    terms(); external, ranks: see the module docstring; includes: start with the #includes; other
    keywords go to goldstoninator"""
    ts = terms(term, **kw)
    external, fields = external_of(ts, ranks, external)
    for t in ts:
        for n, g in t.factors:
            if len(g) == 4 and n != "g":
                raise ValueError("only Coulomb integrals g have a full-m form, not %s_%s" % (n, "".join(g)))
    t0 = ts[0]
    vin, vout = t0.vins[0], t0.vouts[0]
    body = ["double result = 0.0;"]
    if external == "average" and vin != vout:
        body += ["if (F%s.twoj() != F%s.twoj())" % (vout, vin), "  return 0.0;"]
    if external == "Ak":
        body += _AK.split("\n")
    if any(len(g) == 4 for t in ts for _, g in t.factors):
        body += _QZERO.split("\n")
    for t in ts:
        body.append("")
        body.append("// %s" % t.tex)
        body += _term_cpp(t, external, fields)
    if external == "average":
        body.append("return result / (F%s.twoj() + 1);" % vin)
    else:
        body.append("return result;")
    head = ("Generated by angularinator.ampsci_fullmsum: brute-force sum over every m, using "
            "Coulomb::g_abcd and TensorOperator::fullME only.  Returns %s, X the sum of the %d "
            "Goldstone term(s) below; internal core letters run over core, excited ones over excited."
            % (_returns(t0, external, fields), len(ts)))
    return _function(name, _params(ts, external, fields), body, head, includes)


def _params(ts, external, fields):
    """[(type, name)] of the arguments, the same for the full-m and the reduced form"""
    t0 = ts[0]
    params = [("const DiracSpinor &", "F" + x) for x in dict.fromkeys(t0.vins + t0.vouts)]
    params += [("const std::vector<DiracSpinor> &", "core"), ("const std::vector<DiracSpinor> &", "excited")]
    params += [("const Coulomb::QkTable &", "qk")]
    ops = dict.fromkeys(n for t in ts for n, g in t.factors if len(g) == 2)
    params += [("const DiracOperator::TensorOperator &", "op_" + n) for n in ops]
    # the energies in the order of the operators that absorb them (then any others)
    in_dens = {s for t in ts for _, _, om in t.dens for s in om}
    by_op = [sy for n in ops for t in ts for (m, g), sy in zip(t.factors, t.omegas) if m == n and sy]
    syms = dict.fromkeys([sy for sy in by_op if sy in in_dens] +
                         [s for t in ts for _, _, om in t.dens for s in om])
    omegas = {}
    for sy in syms:
        omegas.setdefault(_omega_var(sy), set()).add(sy)
    for w, ss in omegas.items():
        if len(ss) > 1:
            raise ValueError("the energies %s would share the C++ name %s: give them other symbols" % (sorted(ss), w))
    params += [("double ", w) for w in omegas]
    if external == "Ak":
        params.append(("int ", "k"))
    elif external == "coupled":
        params.append(("int ", "K"))
    elif external == "m":
        params += [("int ", "tmi_" + x) for x in t0.vins] + [("int ", "tmo_" + x) for x in t0.vouts]
        params += [("int ", "tq_" + n) for n in fields]
    return params


def _returns(t0, external, fields):
    vin, vout = t0.vins[0], t0.vouts[0]
    return {
        "average": "(1/[j_%s]) sum_m <%s m|X|%s m>, the m-independent value (averaged over m)" % (vin, vout, vin),
        "reduced": "the reduced matrix element <%s||X||%s> (rank of op_%s)" % (vout, vin, fields[0] if fields else ""),
        "Ak": "S^k in X = sum_k A^k_{%s} S^k_{%s} (S^k = [k] sum_m A^k X)"
        % ("".join(t0.vouts + t0.vins), "".join(t0.vouts + t0.vins)),
        "m": "the m-resolved matrix element of X, at the given 2m and 2q (the reduced form: sum over the "
             "chain basis along the valence line)",
        "coupled": "X_K in X = sum_K B_K X_K, the fields %s coupled to K: B_K = sum_Q <k_%s q_%s k_%s q_%s|K Q> "
                   "(-1)^{%s-m_%s} (%s K %s; -m_%s Q m_%s)" % (" x ".join(fields), fields[0], fields[0],
                   fields[-1], fields[-1], vout, vout, vout, vin, vout, vin) if len(fields) == 2 else "",
    }[external]


def _function(name, params, body, head, includes):
    """the C++ function: header comment, signature (unused arguments commented out), body"""
    text = "\n".join(body)
    params = [ty + (v if re.search(r"\b%s\b" % v, text) else "/*%s*/" % v) for ty, v in params]
    head = textwrap.wrap(head, width=88, initial_indent="// ", subsequent_indent="// ")
    sig = "double %s(" % name
    sig += (",\n" + " " * len(sig)).join(params) + ")"
    out = (INCLUDES + "\n" if includes else "") + "\n".join(head) + "\n" + sig + " {\n"
    out += "\n".join(("  " + l) if l else "" for l in body) + "\n}\n"
    return Code(out)


def _term_cpp(t, external, fields):
    """the C++ lines of one term's contribution, added to result.  A check that would skip the term
    before any loop is opened puts the term inside do { ... } while (false), where continue leaves it"""
    lines, depth, wrap = [], 1, [False]
    emit = lambda s: lines.append("  " * depth + s)

    def skip_if(cond):
        emit("if (%s)" % cond)
        emit("  continue;")
        wrap[0] = wrap[0] or depth == 1

    def ready(cs, have):
        out = [c for c in cs if c[-1] <= have]
        for c in out:
            cs.remove(c)
        return out

    # selection-rule checks on the states, each as soon as its letters are fixed
    checks = []
    for n, g in t.factors:
        st = ", ".join("F" + x for x in g)
        checks.append(("Qzero(%s)" % st if len(g) == 4 else "op_%s.isZero(%s)" % (n, st), set(g)))
    bound = set(t.vins + t.vouts)
    for c, _ in ready(checks, bound):
        skip_if(c)
    for x in t.dummies():
        emit("for (const auto &F%s : %s) {" % (x, "core" if t.kind[x] == "core" else "excited"))
        depth += 1
        bound.add(x)
        for c, _ in ready(checks, bound):
            skip_if(c)
    nstate = depth

    if t.dens:
        emit("const double inv_den = 1.0 / (%s);" % " * ".join("(%s)" % _den_cpp(*d) for d in t.dens))
    emit("double msum = 0.0;")

    # the external m (and q) loops
    vin, vout = t.vins[0], t.vouts[0]
    weight = None

    def loop(var, j):
        nonlocal depth
        emit("for (int %s = -%s; %s <= %s; %s += 2) {" % (var, j, var, j, var))
        depth += 1

    if external == "average":
        loop("tmi_" + vin, "F%s.twoj()" % vin)
        emit("const int tmo_%s = tmi_%s;" % (vout, vin))
    elif external == "reduced":
        f = fields[0]
        loop("tmi_" + vin, "F%s.twoj()" % vin)
        loop("tq", "2 * op_%s.rank()" % f)
        emit("const int tmo_%s = tmi_%s + tq;" % (vout, vin))
        skip_if("std::abs(tmo_%s) > F%s.twoj()" % (vout, vout))
        emit("const double w3j = op_%s.rme3js(F%s, F%s, tmi_%s, tq);" % (f, vout, vin, vin))
        skip_if("w3j == 0.0")
        weight = "w3j"
    elif external == "coupled":
        f1, f2 = fields
        loop("tmi_" + vin, "F%s.twoj()" % vin)
        loop("tq_" + f1, "2 * op_%s.rank()" % f1)
        loop("tq_" + f2, "2 * op_%s.rank()" % f2)
        emit("const int tQ = tq_%s + tq_%s;" % (f1, f2))
        emit("const int tmo_%s = tmi_%s + tQ;" % (vout, vin))
        skip_if("std::abs(tmo_%s) > F%s.twoj() || std::abs(tQ) > 2 * K" % (vout, vout))
        emit("const double wC = Angular::neg1pow_2(2 * op_%s.rank() - 2 * op_%s.rank() + tQ) * "
             "std::sqrt(2 * K + 1) * Angular::threej_2(2 * op_%s.rank(), 2 * op_%s.rank(), 2 * K, tq_%s, tq_%s, -tQ) *"
             % (f1, f2, f1, f2, f1, f2))
        emit("                  Angular::neg1pow_2(F%s.twoj() - tmo_%s) * Angular::threej_2(F%s.twoj(), 2 * K, F%s.twoj(), -tmo_%s, tQ, tmi_%s);"
             % (vout, vout, vout, vin, vout, vin))
        skip_if("wC == 0.0")
        weight = "wC"
    elif external == "Ak":
        (v, w), (x, y) = t.vins, t.vouts
        loop("tmi_" + v, "F%s.twoj()" % v)
        loop("tmi_" + w, "F%s.twoj()" % w)
        loop("tmo_" + x, "F%s.twoj()" % x)
        emit("const int tmo_%s = tmi_%s + tmi_%s - tmo_%s;" % (y, v, w, x))
        skip_if("std::abs(tmo_%s) > F%s.twoj()" % (y, y))
        emit("const double wA = (2 * k + 1) * Ak(k, F%s, F%s, F%s, F%s, tmo_%s, tmo_%s, tmi_%s, tmi_%s);"
             % (x, y, v, w, x, y, v, w))
        skip_if("wA == 0.0")
        weight = "wA"

    # the factors, each as soon as its m's are fixed; internal m loops in a greedy order
    def qvar(n):
        if n not in fields:
            return "0"
        return "tq" if external == "reduced" else "tq_" + n  # m and coupled: tq_<name>

    facs = []
    for k, ((n, g), roles) in enumerate(zip(t.factors, t.roles)):
        ms = [_mvar(t, x, r) for x, r in zip(g, roles)]
        st = ", ".join("F" + x for x in g)
        if len(g) == 4:
            expr = "Coulomb::g_abcd(%s, %s)" % (st, ", ".join(ms))
        else:
            expr = "op_%s.fullME(%s, %s, %s)" % (n, st, ", ".join(ms), qvar(n))
        facs.append(("f%d" % (k + 1), expr, set(ms)))
    have = {_mvar(t, x, "in") for x in t.vins} | {_mvar(t, x, "out") for x in t.vouts}
    names = []

    def place():
        for f, expr, _ in ready(facs, have):
            emit("const double %s = %s;" % (f, expr))
            skip_if("%s == 0.0" % f)
            names.append(f)

    place()
    todo = t.dummies()
    while todo:
        x = max(todo, key=lambda y: (sum(1 for f in facs if f[2] <= have | {"tm_" + y}), -todo.index(y)))
        todo.remove(x)
        loop("tm_" + x, "F%s.twoj()" % x)
        have.add("tm_" + x)
        place()
    emit("msum += %s;" % " * ".join(([weight] if weight else []) + names))
    while depth > nstate:
        depth -= 1
        emit("}")
    emit("result %s= msum%s;" % ("+" if t.sign > 0 else "-", " * inv_den" if t.dens else ""))
    while depth > 1:
        depth -= 1
        emit("}")
    if wrap[0]:
        return ["do {"] + lines + ["} while (false);"]
    return [l[2:] for l in lines]


# ----------------------------------------------------------------------------------------------
# Wigner symbols, numerically (for the internal checks); every argument is twice the value
# ----------------------------------------------------------------------------------------------
_fact = math.factorial


def _triangle2(a, b, c):
    return (a + b + c) % 2 == 0 and abs(a - b) <= c <= a + b


def _delta2(a, b, c):
    return Fraction(
        _fact((a + b - c) // 2) * _fact((a - b + c) // 2) * _fact((b + c - a) // 2),
        _fact((a + b + c) // 2 + 1),
    )


@lru_cache(maxsize=None)
def threej2(j1, j2, j3, m1, m2, m3):
    """the 3j symbol (j1 j2 j3; m1 m2 m3), Racah's formula"""
    if m1 + m2 + m3 or not _triangle2(j1, j2, j3):
        return 0.0
    if any(abs(m) > j or (j + m) % 2 for j, m in ((j1, m1), (j2, m2), (j3, m3))):
        return 0.0
    pre = _delta2(j1, j2, j3)
    for j, m in ((j1, m1), (j2, m2), (j3, m3)):
        pre *= _fact((j + m) // 2) * _fact((j - m) // 2)
    s = Fraction(0)
    for k in range((j1 + j2 - j3) // 2 + 1):
        d = (k, (j3 - j2 + m1) // 2 + k, (j3 - j1 - m2) // 2 + k, (j1 + j2 - j3) // 2 - k,
             (j1 - m1) // 2 - k, (j2 + m2) // 2 - k)
        if min(d) >= 0:
            s += Fraction((-1) ** k, math.prod(_fact(x) for x in d))
    return (-1) ** ((j1 - j2 - m3) // 2) * math.sqrt(pre) * float(s)


@lru_cache(maxsize=None)
def sixj2(j1, j2, j3, j4, j5, j6):
    """the 6j symbol {j1 j2 j3; j4 j5 j6}, Racah's formula"""
    tri = ((j1, j2, j3), (j1, j5, j6), (j4, j2, j6), (j4, j5, j3))
    if not all(_triangle2(*t) for t in tri):
        return 0.0
    pre = math.prod(_delta2(*t) for t in tri)
    a = [sum(t) // 2 for t in tri]
    b = [(j1 + j2 + j4 + j5) // 2, (j2 + j3 + j5 + j6) // 2, (j3 + j1 + j6 + j4) // 2]
    s = Fraction(0)
    for t in range(max(a), min(b) + 1):
        s += Fraction((-1) ** t * _fact(t + 1),
                      math.prod(_fact(t - x) for x in a) * math.prod(_fact(x - t) for x in b))
    return math.sqrt(pre) * float(s)


@lru_cache(maxsize=None)
def ninej2(a, b, c, d, e, f, g, h, i):
    """the 9j symbol {a b c; d e f; g h i} = sum_x (-1)^{2x} [x] {a b c; f i x}{d e f; b x h}{g h i; x a d}"""
    lo = max(abs(a - i), abs(d - h), abs(b - f))
    hi = min(a + i, d + h, b + f)
    return sum((-1) ** x * (x + 1) * sixj2(a, b, c, f, i, x) * sixj2(d, e, f, b, x, h) * sixj2(g, h, i, x, a, d)
               for x in range(lo, hi + 1, 2))


def sixj_forms(t):
    """the 24 equal forms of a 6j symbol {j1 j2 j3; j4 j5 j6} (column permutations, and swapping
    upper and lower entries in two columns)"""
    cols = [(t[0], t[3]), (t[1], t[4]), (t[2], t[5])]
    out = []
    for perm in itertools.permutations(cols):
        for flip in ((0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)):
            cs = [(c[1], c[0]) if f else c for c, f in zip(perm, flip)]
            out.append(tuple(c[0] for c in cs) + tuple(c[1] for c in cs))
    return out


def recognise_9j(f):
    """replace sum_x [x] (-1)^{phase} 6j 6j 6j by a 9j where the three 6j symbols are those of a 9j
    and x appears nowhere else (its phase must then be x-independent over the sum)"""
    for x in list(f.sums):
        six = [t for t in f.sixj if x in t]
        implied = {frozenset(tr) for t in six for tr in _six_triads(t)}
        if len(six) != 3 or f.power.get(x, 0) != 1 or any(x in t and frozenset(t) not in implied for t in f.triads):
            continue
        if any(t.count(x) != 1 for t in six):
            continue
        c = f.phase.get(x, 0)
        if Fraction(c).denominator != 1 or c % 2:
            continue  # (-1)^{c x} must be constant over the sum
        nine = None
        for s1, s2, s3 in itertools.permutations(six):
            for A in sixj_forms(s1):
                if A[5] != x:
                    continue
                a, b, c_, f_, i_ = A[:5]
                for B in sixj_forms(s2):
                    if B[4] != x or B[2] != f_ or B[3] != b:
                        continue
                    d, e, h = B[0], B[1], B[5]
                    for C in sixj_forms(s3):
                        if C[3] == x and C[1] == h and C[2] == i_ and C[4] == a and C[5] == d:
                            nine = (a, b, c_, d, e, f_, C[0], h, i_)
                            break
                    if nine:
                        break
                if nine:
                    break
            if nine:
                break
        if not nine:
            continue
        # sum_x [x] (-1)^{c x} 6j6j6j = (-1)^{c x} (-1)^{2x} 9j, both constant over the sum
        const = Fraction(c, 2) + 1 if f.half[x] else 0
        f.phase.pop(x, None)
        f.add_phase({}, const)
        f.power.pop(x)
        f.sums.remove(x)
        f.triads = [t for t in f.triads if x not in t]
        for t in six:
            f.sixj.remove(t)
        f.ninej.append(nine)
    return f


# ----------------------------------------------------------------------------------------------
# angular graphs: networks of 3j symbols in standard form
# ----------------------------------------------------------------------------------------------
class Factors:
    """the factors a reduction collects: phase (exponent of -1, {symbol: coefficient}, None for the
    constant), power ({symbol: p} for [j]^p), sixj (6-tuples of symbols), triads (triangle
    conditions not implied by a 6j), deltas ([(kept, dropped)], every identification, also the ones
    that collapse a sum), sums (the new summed symbols), subst ({dropped: kept}), zero (True when a
    selection rule kills the graph)"""

    def __init__(self):
        self.phase, self.power, self.sixj, self.deltas, self.sums, self.subst = {}, {}, [], [], [], {}
        self.triads, self.ninej = [], []
        self.zero = False

    def add_phase(self, terms, const=0):
        """add sum c s + const to the exponent of -1; terms: [(symbol, c)] (a symbol may repeat) or
        {symbol: c}"""
        for s, c in (terms.items() if isinstance(terms, dict) else terms):
            self.phase[s] = self.phase.get(s, 0) + Fraction(c)
        self.phase[None] = self.phase.get(None, 0) + Fraction(const)

    def add_power(self, s, p):
        self.power[s] = self.power.get(s, 0) + Fraction(p)

    def rename(self, old, new, collapse=True):
        """replace a symbol everywhere: after an identification (collapse: a sum over old goes), or
        only a new name (collapse=False: old stays summed, as new)"""
        if old in self.phase:
            self.phase[new] = self.phase.get(new, 0) + self.phase.pop(old)
        if old in self.power:
            self.power[new] = self.power.get(new, 0) + self.power.pop(old)
        self.sixj = [tuple(new if x == old else x for x in t) for t in self.sixj]
        self.triads = [tuple(new if x == old else x for x in t) for t in self.triads]
        self.ninej = [tuple(new if x == old else x for x in t) for t in self.ninej]
        self.sums = [s for s in self.sums if s != old] if collapse else [new if s == old else s for s in self.sums]
        self.subst = {k: (new if v == old else v) for k, v in self.subst.items()}
        self.subst[old] = new


class Graph:
    """A network of 3j symbols in standard form.  nodes {id: [(line, end)] * 3}, the entries of the 3j
    in order, end '+' for the entry +m of the line, '-' for -m; sym {line: its angular momentum
    symbol}; half {symbol: True for a half-integer}; kind {symbol: 'fixed' (a letter's j, a field's
    rank, 0), 'summed' (a multipolarity) or 'inter' (an intermediate of the reduction)}.  A line with
    both ends has one '+' and one '-' end and is summed over its m with the weight (-1)^{j-m}; a line
    with one end is open.  factors collects what the reduction takes out."""

    def __init__(self):
        self.nodes, self.sym = {}, {}
        self.half, self.kind = {"0": False}, {"0": "fixed"}
        self.factors = Factors()
        self.factors.half = self.half
        self._id = itertools.count()

    def symbol(self, s, half, kind):
        self.half[s], self.kind[s] = half, kind
        return s

    def line(self, s):
        l = next(self._id)
        self.sym[l] = s
        return l

    def node(self, *entries):
        n = next(self._id)
        self.nodes[n] = list(entries)
        return n

    def ends(self, l):
        """{end: (node, position)} of a line"""
        out = {}
        for n, es in self.nodes.items():
            for i, (x, e) in enumerate(es):
                if x == l:
                    out[e] = (n, i)
        return out

    def j(self, l):
        return self.sym[l]

    def copy(self):
        g = Graph()
        g.nodes = {n: list(es) for n, es in self.nodes.items()}
        g.sym, g.half, g.kind = dict(self.sym), dict(self.half), dict(self.kind)
        f, h = self.factors, g.factors
        h.phase, h.power, h.subst = dict(f.phase), dict(f.power), dict(f.subst)
        h.sixj, h.deltas, h.sums, h.zero = list(f.sixj), list(f.deltas), list(f.sums), f.zero
        h.triads, h.ninej = list(f.triads), list(f.ninej)
        h.half = g.half
        g._id = itertools.count(max(list(self.nodes) + list(self.sym) + [0]) + 1)
        return g

    # -- the two elementary moves ---------------------------------------------------------------
    def reverse(self, l):
        """swap the '+' and '-' ends of an internal line: (-1)^{2j}"""
        for n, es in self.nodes.items():
            self.nodes[n] = [(x, {"+": "-", "-": "+"}[e]) if x == l else (x, e) for x, e in es]
        self.factors.add_phase({self.sym[l]: 2})

    def arrange(self, n, entries):
        """put the entries of node n in the given order (a permutation of them): an odd permutation
        gives (-1)^{j1+j2+j3}"""
        es = self.nodes[n]
        perm = [es.index(e) for e in entries]
        if sorted(perm) != [0, 1, 2]:
            raise ValueError("not a permutation of node %s: %s -> %s" % (n, es, entries))
        odd = sum(1 for i in range(3) for k in range(i + 1, 3) if perm[i] > perm[k]) % 2
        if odd:
            sj = {}
            for x, _ in es:
                sj[self.sym[x]] = sj.get(self.sym[x], 0) + 1
            self.factors.add_phase(sj)
        self.nodes[n] = list(entries)

    def end_at(self, l, n, want):
        """make the end of line l at node n be want ('+' or '-'), reversing l if needed"""
        es = self.nodes[n]
        if (l, want) not in es:
            self.reverse(l)

    # -- numerical value -------------------------------------------------------------------------
    def value(self, vals, ext=None):
        """the value for symbol values vals ({symbol: 2j}), the open lines at the 2m in ext ({line: 2m})"""
        ext = ext or {}
        lines = sorted({x for es in self.nodes.values() for x, _ in es})
        vals = dict(vals, **{"0": 0})
        tj = {l: vals[self.sym[l]] for l in lines}
        internal = [l for l in lines if len(self.ends(l)) == 2]
        at = {l: [n for n, es in self.nodes.items() if any(x == l for x, _ in es)] for l in lines}
        m = dict(ext)
        for l in lines:  # a dangling zero line has m = 0
            if l not in internal and tj[l] == 0:
                m.setdefault(l, 0)
        order = sorted(internal, key=lambda l: -len(at[l]))

        def node_val(n):
            es = self.nodes[n]
            return threej2(*[tj[x] for x, _ in es], *[(m[x] if e == "+" else -m[x]) for x, e in es])

        def rec(i):
            if i == len(order):
                return 1.0
            l = order[i]
            total = 0.0
            for ml in range(-tj[l], tj[l] + 1, 2):
                m[l] = ml
                f = (-1) ** ((tj[l] - ml) // 2)
                for n in set(at[l]):
                    if all(x in m for x, _ in self.nodes[n]):
                        f *= node_val(n)
                        if f == 0.0:
                            break
                if f != 0.0:
                    total += f * rec(i + 1)
            del m[l]
            return total

        done = [n for n in self.nodes if all(x in m for x, _ in self.nodes[n])]
        pre = math.prod(node_val(n) for n in done)
        return pre * rec(0) if pre else 0.0

    # -- reduction ---------------------------------------------------------------------------------
    def identify(self, l, k):
        """the delta_{j_l j_k} of a rule: line k takes the symbol of line l (or the other way round,
        keeping the more fixed one: a letter's j or a field's rank, then a multipolarity, then an
        intermediate); recorded in factors.deltas; returns the symbol kept"""
        a, b = self.sym[l], self.sym[k]
        if a == b:
            return a
        if self.half[a] != self.half[b]:
            self.factors.zero = True
            return a
        rank = {"fixed": 0, "summed": 1, "inter": 2}
        if rank[self.kind[b]] < rank[self.kind[a]] or (b == "0" and a != "0"):
            a, b = b, a
        self.factors.deltas.append((a, b))
        self.factors.rename(b, a)
        for x in self.sym:
            if self.sym[x] == b:
                self.sym[x] = a
        return a

    def _other(self, l, n):
        """(node, end) of the end of line l that is not at node n (or at the other position of n)"""
        es = [(m, e) for m, ents in self.nodes.items() for x, e in ents if x == l]
        rest = [me for me in es if me[0] != n] or es[1:]
        return rest[0]

    def _replace(self, n, old, new):
        self.nodes[n] = [new if e == old else e for e in self.nodes[n]]

    def _junction(self, n, z):
        """remove the zero line z at node n: (j_p j_q 0) joins p and q, delta_{pq}/sqrt[p]"""
        (p, _), (q, _) = [e for e in self.nodes[n] if e[0] != z]
        if len(self.ends(p)) == 1 or len(self.ends(q)) == 1:
            # another dangling zero line here: all three are zero, (0 0 0; 0 0 0) = 1
            self.identify(z, p)
            self.identify(z, q)
            del self.nodes[n]
            return
        if p == q:  # the zero line on a loop: sum_m (-1)^{j-m} (j j 0; -m m 0) = (-1)^{2j} sqrt[j]
            self.arrange(n, [(p, "-"), (p, "+"), next(e for e in self.nodes[n] if e[0] == z)])
            self.factors.add_phase({self.sym[p]: 2})
            self.factors.add_power(self.sym[p], Fraction(1, 2))
            del self.nodes[n]
            return
        self.end_at(p, n, "+")
        self.end_at(q, n, "-")
        self.arrange(n, [(p, "+"), (q, "-"), next(e for e in self.nodes[n] if e[0] == z)])
        keep = self.identify(p, q)
        self.factors.add_power(keep, Fraction(-1, 2))
        qn, _ = self._other(q, n)
        del self.nodes[n]
        self._replace(qn, (q, "+"), (p, "+"))

    def _tadpole(self, n, l):
        """sum_m (-1)^{j-m} (j j k; -m m .) = delta_{k0} (-1)^{2j} sqrt[j]"""
        k = next(e for e in self.nodes[n] if e[0] != l)
        self.arrange(n, [(l, "-"), (l, "+"), k])
        self.factors.add_phase({self.sym[l]: 2})
        self.factors.add_power(self.sym[l], Fraction(1, 2))
        self.identify(self.line("0"), k[0])  # j_k = 0
        del self.nodes[n]

    def _theta(self, a, b):
        """two 3j's on the same three lines: 1 (A all '-', B all '+', same order)"""
        lines = [x for x, _ in self.nodes[a]]
        for x in lines:
            self.end_at(x, a, "-")
        self.arrange(a, [(x, "-") for x in lines])
        self.arrange(b, [(x, "+") for x in lines])
        self.factors.triads.append(tuple(self.sym[x] for x in lines))
        del self.nodes[a], self.nodes[b]

    def _bubble(self, A, B, e1, e2):
        """A = (e1-, e2-, a-), B = (e1+, e2+, b+): delta_{ab}/[a], a and b joined"""
        a = next(x for x, _ in self.nodes[A] if x not in (e1, e2))
        b = next(x for x, _ in self.nodes[B] if x not in (e1, e2))
        for x in (e1, e2, a):
            self.end_at(x, A, "-")
        self.end_at(b, B, "+")
        self.arrange(A, [(e1, "-"), (e2, "-"), (a, "-")])
        self.arrange(B, [(e1, "+"), (e2, "+"), (b, "+")])
        self.factors.triads.append((self.sym[e1], self.sym[e2], self.sym[a]))
        keep = self.identify(a, b)
        self.factors.add_power(keep, -1)
        dn, _ = self._other(b, B)
        del self.nodes[A], self.nodes[B]
        self._replace(dn, (b, "-"), (a, "-"))

    def _triangle(self, A, B, C, l3, l1, l2):
        """A = (a-, l3-, l2+), B = (b-, l1-, l3+), C = (c-, l2-, l1+) [l3: A-B, l1: B-C, l2: C-A]:
        (-1)^{a+b+c} {a b c; l1 l2 l3} (a-, b-, c-)"""
        a = next(x for x, _ in self.nodes[A] if x not in (l3, l2))
        b = next(x for x, _ in self.nodes[B] if x not in (l1, l3))
        c = next(x for x, _ in self.nodes[C] if x not in (l2, l1))
        self.end_at(l3, A, "-")
        self.end_at(l1, B, "-")
        self.end_at(l2, C, "-")
        self.end_at(a, A, "-")
        self.end_at(b, B, "-")
        self.end_at(c, C, "-")
        self.arrange(A, [(a, "-"), (l3, "-"), (l2, "+")])
        self.arrange(B, [(b, "-"), (l1, "-"), (l3, "+")])
        self.arrange(C, [(c, "-"), (l2, "-"), (l1, "+")])
        S = self.sym
        self.factors.add_phase([(S[a], 1), (S[b], 1), (S[c], 1)])
        self.factors.sixj.append((S[a], S[b], S[c], S[l1], S[l2], S[l3]))
        del self.nodes[A], self.nodes[B], self.nodes[C]
        self.node((a, "-"), (b, "-"), (c, "-"))

    def _interchange(self, A, B, e, a1, b1):
        """A = (a1-, a2-, e-), B = (b1-, b2-, e+): sum_x [x] (-1)^{a2+b1-e-x} {a1 a2 e; b2 b1 x}
        (a1-, b1-, x-)(a2-, b2-, x+)"""
        a2 = next(x for x, _ in self.nodes[A] if x not in (a1, e))
        b2 = next(x for x, _ in self.nodes[B] if x not in (b1, e))
        self.end_at(e, A, "-")
        for x in (a1, a2):
            self.end_at(x, A, "-")
        for x in (b1, b2):
            self.end_at(x, B, "-")
        self.arrange(A, [(a1, "-"), (a2, "-"), (e, "-")])
        self.arrange(B, [(b1, "-"), (b2, "-"), (e, "+")])
        S = self.sym
        x = "x#%d" % (len([k for k in self.kind if self.kind[k] == "inter"]) + 1)
        self.symbol(x, self.half[S[a1]] != self.half[S[b1]], "inter")
        f = self.factors
        f.sums.append(x)
        f.add_power(x, 1)
        f.add_phase([(S[a2], 1), (S[b1], 1), (S[e], -1), (x, -1)])
        f.sixj.append((S[a1], S[a2], S[e], S[b2], S[b1], x))
        del self.nodes[A], self.nodes[B]
        xl = self.line(x)
        self.node((a1, "-"), (b1, "-"), (xl, "-"))
        self.node((a2, "-"), (b2, "-"), (xl, "+"))

    def _cycle(self):
        """the shortest cycle of length >= 2: ([nodes], [lines]) with lines[i] joining nodes[i] and
        nodes[i+1] (cyclically)"""
        link = {}  # line -> (u, v)
        for n, es in self.nodes.items():
            for x, _ in es:
                link.setdefault(x, []).append(n)
        link = {x: ns for x, ns in link.items() if len(ns) == 2 and ns[0] != ns[1]}
        best = None
        for l, (u, v) in sorted(link.items()):
            prev, seen, todo = {u: None}, {u}, [u]
            while todo and v not in seen:
                nxt = []
                for n in todo:
                    for x, (p, q) in sorted(link.items()):
                        if x == l or n not in (p, q):
                            continue
                        m = q if n == p else p
                        if m not in seen:
                            seen.add(m)
                            prev[m] = (n, x)
                            nxt.append(m)
                todo = nxt
            if v not in seen:
                continue
            nodes, lines, m = [v], [], v
            while prev[m] is not None:
                n, x = prev[m]
                lines.append(x)
                nodes.append(n)
                m = n
            nodes, lines = nodes[::-1], lines[::-1]  # u ... v
            cyc = (nodes, lines + [l])
            if best is None or len(cyc[1]) < len(best[1]):
                best = cyc
        return best

    def reduce(self):
        """reduce a closed graph to factors (6j symbols, phases, [j] powers, deltas, sums over
        intermediates); the graph is emptied"""
        while self.nodes and not self.factors.zero:
            loop = [(n, x) for n, es in self.nodes.items() for x, e in es
                    if e == "+" and (x, "-") in es]
            if loop:
                self._tadpole(*loop[0])
                continue
            zero = [(n, x) for n, es in self.nodes.items() for x, _ in es if self.sym[x] == "0"]
            if zero:
                self._junction(*zero[0])
                continue
            theta = [(a, b) for a in self.nodes for b in self.nodes if a < b
                     and len({x for x, _ in self.nodes[a]}) == 3
                     and {x for x, _ in self.nodes[a]} == {x for x, _ in self.nodes[b]}]
            if theta:
                self._theta(*theta[0])
                continue
            cyc = self._cycle()
            if cyc is None:
                raise ValueError("cannot reduce the graph %s" % self.nodes)
            nodes, lines = cyc
            if len(lines) == 2:
                self._bubble(nodes[0], nodes[1], lines[0], lines[1])
            elif len(lines) == 3:
                self._triangle(nodes[0], nodes[1], nodes[2], lines[0], lines[1], lines[2])
            else:
                # interchange on the edge nodes[0]-nodes[1], pairing the two cycle lines
                self._interchange(nodes[0], nodes[1], lines[0], lines[-1], lines[1])
        return self.factors


def factors_value(f, vals):
    """the numerical value of reduction factors, for symbol values vals ({symbol: 2j}), summing
    over the intermediates"""
    vals = dict(vals, **{"0": 0})
    if f.zero or any(vals[a] != vals[b] for a, b in f.deltas):
        return 0.0
    top = 2 * sum(v for v in vals.values()) + 2

    def term(v):
        if not all(_triangle2(*[v[x] for x in t]) for t in f.triads):
            return 0.0
        e2 = sum(c * v[s] for s, c in f.phase.items() if s is not None) + 2 * f.phase.get(None, 0)
        if e2.denominator != 1 or e2 % 2:
            raise ValueError("phase exponent %s is not an integer" % (e2 / 2))
        out = float((-1) ** int(e2 // 2))
        for s, p in f.power.items():
            out *= (v[s] + 1) ** float(p)
        for t in f.sixj:
            out *= sixj2(*[v[x] for x in t])
            if out == 0.0:
                break
        for t in f.ninej:
            out *= ninej2(*[v[x] for x in t])
        return out

    def rec(i, v):
        if i == len(f.sums):
            return term(v)
        x = f.sums[i]
        half = f.half[x]
        return sum(rec(i + 1, dict(v, **{x: tx})) for tx in range(1 if half else 0, top, 2))

    return rec(0, vals)


# ----------------------------------------------------------------------------------------------
# the angular graph of a Goldstone term
# ----------------------------------------------------------------------------------------------
def network(t, external, fields, ranks=None):
    """(graph, info) for a Term: the closed angular graph of its numerator, in which every two-body
    integral g_{pqrs} (in the orientation of the diagram) is sum_k A^k_{pqrs} X^k_{pqrs} and every
    one-body t_{pr} is (-1)^{j_p-m_p} (j_p K j_r; -m_p q m_r) <p||t||r>, closed as external says
    ('average', 'reduced' or 'Ak'; see the module docstring).  The value of the graph (with the
    phases and powers it already holds in graph.factors) is the coefficient of
    prod_i X^{k_i} prod <p||t||r> in the result.  info: 'multipoles' [symbol per two-body factor,
    None for one-body ones], 'ranks' {field name: symbol}, 'k' (the symbol of S^k in Ak mode),
    'integrals' [(name, labels in diagram orientation)]"""
    ranks = ranks or {}
    g = Graph()
    for x, kind in t.kind.items():
        g.symbol(x, True, "fixed")
    line, out_leg, in_leg = {}, {}, {}
    for x in t.vins:
        in_leg[x] = g.line(x)
    for x in t.vouts:
        out_leg[x] = g.line(x)
    for x in t.dummies():
        line[x] = g.line(x)
    bra = lambda x: (out_leg[x] if t.kind[x] == "val" else line[x], "-")
    ket = lambda x: (in_leg[x] if t.kind[x] == "val" else line[x], "+")
    info = {"multipoles": [], "ranks": {}, "integrals": []}
    nk = 0
    for (n, lab), roles in zip(t.factors, t.roles):
        outs = [x for x, r in zip(lab, roles) if r == "out"]
        ins = [x for x, r in zip(lab, roles) if r == "in"]
        if len(lab) == 4:
            nk += 1
            k = g.symbol("k#%d" % nk, False, "summed")
            kl = g.line(k)
            (p, q), (r, s) = outs, ins
            g.node(bra(p), (kl, "-"), ket(r))
            g.node(bra(q), (kl, "+"), ket(s))
            g.factors.add_phase([(p, -1), (q, -1)], 1)  # (-1)^{1-j_p-j_q}
            info["multipoles"].append(k)
            info["integrals"].append((n, (p, q, r, s)))
        else:
            (p,), (r,) = outs, ins
            info["multipoles"].append(None)
            info["integrals"].append((n, (p, r)))
            if n in fields:
                K = info["ranks"].get(n) or g.symbol("K#" + n, False, "fixed")
                info["ranks"][n] = K
                info.setdefault("field_lines", {})[n] = Kl = g.line(K)
                g.node(bra(p), (Kl, "+"), ket(r))
            else:  # a scalar: rank 0
                g.node(bra(p), (g.line("0"), "+"), ket(r))
    if external == "average":
        (v,), (w,) = t.vins, t.vouts
        vl, wl = in_leg[v], out_leg[w]
        n = next(m for m, es in g.nodes.items() if (vl, "+") in es)
        g._replace(n, (vl, "+"), (wl, "+"))
        keep = g.identify(wl, vl) if v != w else v
        g.factors.add_power(keep, -1)
    elif external == "reduced":
        (v,), (w,) = t.vins, t.vouts
        K = info["ranks"][fields[0]]
        Kl = next(x for x, s in g.sym.items() if s == K)
        g.node((out_leg[w], "+"), (Kl, "-"), (in_leg[v], "-"))
        g.factors.add_phase({}, 1)  # the projection is minus this graph
    elif external == "Ak":
        (v, w), (x, y) = t.vins, t.vouts
        k = g.symbol("K#out", False, "fixed")
        kl = g.line(k)
        g.node((out_leg[x], "+"), (kl, "+"), (in_leg[v], "-"))
        g.node((out_leg[y], "+"), (kl, "-"), (in_leg[w], "-"))
        g.factors.add_power(k, 1)
        g.factors.add_phase([(x, -1), (y, -1)], 1)  # [k] (-1)^{1-j_x-j_y}
        info["k"] = k
    elif external == "m" and len(t.vins) == 1 and fields:
        # the chain basis along the valence line, fields in the given order:
        # B = prod_i (-1)^{j_{i-1}-m_{i-1}} (j_{i-1} K_i j_i; -m_{i-1} q_i m_i), j_0 = w, j_n = v;
        # X_{j_1...} = prod [j_i] sum_m B X = (-1)^n prod [j_i] x (the graph closed with the
        # projectors (j_{i-1}+, K_i-, j_i-))
        (v,), (w,) = t.vins, t.vouts
        prev, chain = (out_leg[w], "+"), []
        for i, fname in enumerate(fields):
            Kl = info["field_lines"][fname]
            if i < len(fields) - 1:
                x = g.symbol("j#%d" % (i + 1), True, "fixed")
                xl = g.line(x)
                g.node(prev, (Kl, "-"), (xl, "-"))
                g.factors.add_power(x, 1)
                prev = (xl, "+")
                chain.append(x)
            else:
                g.node(prev, (Kl, "-"), (in_leg[v], "-"))
        g.factors.add_phase({}, len(fields))
        info["chain"] = chain
    elif external == "coupled" and len(t.vins) == 1 and len(fields) == 2:
        # X_K = sum_m B_K X = -sqrt[K] (-1)^{k1-k2+K} x (the graph closed with (k1-, k2-, K+) and (w+, K-, v-))
        (v,), (w,) = t.vins, t.vouts
        K = g.symbol("K#cpl", False, "fixed")
        Kl = g.line(K)
        k1, k2 = (info["ranks"][f] for f in fields)
        g.node((info["field_lines"][fields[0]], "-"), (info["field_lines"][fields[1]], "-"), (Kl, "+"))
        g.node((out_leg[w], "+"), (Kl, "-"), (in_leg[v], "-"))
        g.factors.add_power(K, Fraction(1, 2))
        g.factors.add_phase([(k1, 1), (k2, -1), (K, 1)], 1)
        info["k"] = K
    else:
        raise ValueError("no angular graph for external=%r with %d valence line(s) and %d field(s)"
                         % (external, len(t.vins), len(fields)))
    return g, info


# ----------------------------------------------------------------------------------------------
# the reduced form: phases, direct + exchange -> W, terms, LaTeX
# ----------------------------------------------------------------------------------------------
GREEK = ("mu", "lambda", "nu", "rho", "sigma", "tau")  # names of the Coulomb multipolarities
TEX = {"mu": r"\mu", "lambda": r"\lambda", "nu": r"\nu", "rho": r"\rho", "sigma": r"\sigma", "tau": r"\tau"}


def canonical_phase(phase, half, order=()):
    """([(sign, symbol)], const): the exponent of -1 written as +-symbols plus a constant (0, 1/2,
    1 or 3/2), equal mod 2 to sum_s c_s s + c0: integer symbols first, then the half-integer ones
    with alternating signs (a - b + c ...), in the given order (then alphabetical)"""
    const = Fraction(phase.get(None, 0))
    odd = []
    for s, c in phase.items():
        if s is None or s == "0" or c == 0:
            continue
        if Fraction(c).denominator != 1:
            raise ValueError("phase coefficient %s of %s is not an integer" % (c, s))
        if half[s]:
            const += Fraction(c, 2)  # j = J + 1/2
        if c % 2:
            odd.append(s)
    if const.denominator != 1:
        raise ValueError("internal error: the phase %s is not real" % {k: str(v) for k, v in phase.items()})
    key = lambda s: (order.index(s) if s in order else len(order), s)
    ints = sorted((s for s in odd if not half[s]), key=key)
    halfs = sorted((s for s in odd if half[s]), key=key)
    terms = [(1, s) for s in ints] + [(1 - 2 * (i % 2), s) for i, s in enumerate(halfs)]
    const -= Fraction(sum(sg for sg, s in terms if half[s]), 2)  # +-J = +-j -+ 1/2
    return terms, const % 2


def _tilde_terms(term, **kw):
    """[(Term, tilde)]: the terms, each g~ kept as one term (its direct diagram, tilde the indices of
    the antisymmetrised integrals)"""
    if isinstance(term, (gd.Grid, gd.Diagram)) or not (gd.several(term) or gd.has_tilde(term)):
        return [(t, frozenset()) for t in terms(term, **kw)]
    if gd.several(term):
        return [x for t in term for x in _tilde_terms(t, **kw)]
    d = gd._expanded(term, layout=False, **kw)[0]  # also checks the signs of the exchanges
    return [(Term(d), frozenset(i for i, t in enumerate(gd.parse_tilde(term)[2]) if t))]


def _canon(t, tilde, swap=None):
    """(key, parity): the key is equal for terms that are the same diagram whatever the names of their
    internal letters (renamed to placeholders by kind), with g_pqrs = g_qpsr, each g~ (W) in the
    orientation that makes the key smallest; parity: (-1)^(number of g~ turned round, g~_pqsr =
    -g~_pqrs).  swap: a two-body factor written with its in-letters swapped.  The denominators are
    part of the key"""
    facs = []
    for k, ((n, lab), roles) in enumerate(zip(t.factors, t.roles)):
        outs = tuple(x for x, r in zip(lab, roles) if r == "out")
        ins = tuple(x for x, r in zip(lab, roles) if r == "in")
        if k == swap:
            ins = ins[::-1]
        facs.append((n, outs, ins, k in tilde))
    turn = [k for k, (n, outs, ins, w) in enumerate(facs) if w and len(outs) == 2]
    core = [x for x in t.dummies() if t.kind[x] == "core"]
    exc = [x for x in t.dummies() if t.kind[x] != "core"]
    best = None
    for pc in itertools.permutations(range(len(core))):
        for pe in itertools.permutations(range(len(exc))):
            rel = {x: "#c%d" % i for x, i in zip(core, pc)}
            rel.update({x: "#e%d" % i for x, i in zip(exc, pe)})
            r = lambda xs: tuple(rel.get(x, x) for x in xs)
            dens = tuple(sorted((tuple(sorted(r(p))), tuple(sorted(r(m))), tuple(sorted(om.items())))
                                for p, m, om in t.dens))
            for flips in itertools.product((0, 1), repeat=len(turn)):
                key = []
                for k, (n, outs, ins, w) in enumerate(facs):
                    if len(outs) == 2:
                        if k in turn and flips[turn.index(k)]:
                            ins = ins[::-1]
                        a, b = r(outs) + r(ins), r(outs[::-1]) + r(ins[::-1])
                        key.append((n, min(a, b), w))
                    else:
                        key.append((n, r(outs) + r(ins), w))
                full = (tuple(sorted(key)), dens)
                if best is None or full < best[0]:
                    best = (full, (-1) ** sum(flips))
    return best


def pair_exchange(items):
    """merge each direct-exchange pair of terms (the same diagram with the in-letters of one two-body
    integral swapped, and the opposite sign) into one term with g~ there (so W), trying the later
    integrals first (QW rather than WQ), until no pair is left"""
    items = list(items)
    while True:
        keys = [_canon(t, tl) for t, tl in items]
        hit = None
        for i, (t, tl) in enumerate(items):
            for v in reversed(range(len(t.factors))):
                if v in tl or len(t.factors[v][1]) != 4 or t.factors[v][0] != "g":
                    continue
                key, par = _canon(t, tl, swap=v)
                j = next((j for j, (k, _) in enumerate(keys) if j != i and k == key), None)
                if j is not None:
                    if items[j][0].sign * keys[j][1] != -t.sign * par:
                        raise ValueError("%s and its exchange %s have the same sign" % (t.text, items[j][0].text))
                    hit = (i, j, v)
                    break
            if hit:
                break
        if not hit:
            return items
        i, j, v = hit
        items[i] = (items[i][0], items[i][1] | {v})
        del items[j]


class RTerm:
    """one term of the reduced form: sign; f (the angular Factors, symbols renamed); integrals
    [(X, labels, k)] (X 'Q', 'W' or the name of another two-body line, labels in the diagram
    orientation, k its multipolarity symbol); onebody [(name, (p, r))] (reduced matrix elements
    <p||name||r>); dens (goldstoninator's); kind {letter: 'core' | 'exc' | 'val'}; dummies (the
    letters summed over); multipoles (the summed multipolarity symbols); fixed {symbol: value}
    (multipolarities fixed to '0', a field's rank or k); text (the Goldstone term)"""

    def __init__(self, t, tilde, external, fields, ranks):
        g, info = network(t, external, fields, ranks)
        f = g.reduce()
        self.t = t
        self.sign, self.dens, self.kind, self.text = t.sign, t.dens, dict(t.kind), t.text
        self.dummies = t.dummies()
        self.order = list(dict.fromkeys(t.vins + t.vouts)) + self.dummies
        # display names, none equal to a letter, a user's rank name or another: Coulomb multipolarities
        # mu, lambda, ... (k if alone), field ranks k (k_t), the A^k output k, the coupled K,
        # intermediates J (J_1, ...)
        ranks = ranks or {}
        ks = [k for k in info["multipoles"] if k]
        alive = list(dict.fromkeys(f.subst.get(k, k) for k in ks))
        summed = [k for k in alive if g.kind.get(k) == "summed"]
        args = ARGS.get(external, set())
        used_raw, used_tex, used_cpp = set(), set(), set(RESERVED)

        def take(c, cpp=True):
            used_raw.add(c)
            used_tex.add(tex_sym(c))
            if cpp:
                used_cpp.add(_cpp_name(c))

        for x in t.kind:  # letters: plain in LaTeX (F<x> in C++)
            used_raw.add(x)
            used_tex.add(tex_sym(x))
        for r in ranks.values():
            if r != 0:
                take(str(r))
        names = {}

        def pick(*cands, arg=False):
            """the first candidate whose name, LaTeX and C++ name are all free (arg: the symbol is a
            C++ argument, k or K, whatever it is called)"""
            for c in cands:
                if c in used_raw or tex_sym(c) in used_tex:
                    continue
                if not arg and (_cpp_name(c) in used_cpp or _cpp_name(c) in args):
                    continue
                take(c, cpp=not arg)
                return c
            raise ValueError("no free name among %s" % (cands,))

        rank_syms = info["ranks"]
        for fname, K in rank_syms.items():
            if fname in ranks:
                names[K] = str(ranks[fname])
            else:
                names[K] = pick(*(["k"] if len(rank_syms) == 1 else []) + ["k_" + fname, "K_" + fname, "k_%s0" % fname])
        if info.get("k"):
            names[info["k"]] = (pick("k", "K", "k_0", "k_1", arg=True) if external == "Ak"
                                else pick("K", "K_0", "K_1", arg=True))
        for i, c in enumerate(info.get("chain", [])):
            names[c] = pick("j_%d" % (i + 1), "l_%d" % (i + 1), "x_%d" % (i + 1))
        lone = len(summed) == 1 and not rank_syms and external != "Ak"
        for k in summed:
            names[k] = pick(*(["k"] if lone else []) + list(GREEK) + ["k_%d" % i for i in range(1, 9)])
        inters = list(f.sums)
        for i, x in enumerate(inters):
            names[x] = pick(*(["J"] if len(inters) == 1 else []) + ["J_%d" % j for j in range(i + 1, 30)])
        # the identifications of fixed symbols (letters, ranks, k, chain j's) are constraints of the
        # result; those of multipolarities and intermediates only collapse their sums
        self.constraints = [(names.get(a, a), names.get(b, b)) for a, b in f.deltas if g.kind.get(b) == "fixed"]
        # rename through temporary names, so that no symbol can take a name that is still in use
        tmp = {}
        for i, (old, new) in enumerate(names.items()):
            if old != new:
                tk = "\x00%d" % i
                g.half[tk], g.kind[tk] = g.half[old], g.kind[old]
                f.rename(old, tk, collapse=False)
                tmp[tk] = new
        for tk, new in tmp.items():
            g.half[new], g.kind[new] = g.half[tk], g.kind[tk]
            f.rename(tk, new, collapse=False)
        res = lambda k: names.get(f.subst.get(k, k), f.subst.get(k, k))
        recognise_9j(f)
        self.f = f
        self.half = g.half
        self.multipoles = [names[k] for k in summed]
        self.inters = [names[x] for x in inters if names[x] in f.sums]
        self.fixed = {}
        self.integrals, self.onebody = [], []
        for idx, ((n, lab), k) in enumerate(zip(info["integrals"], info["multipoles"])):
            if k:
                X = "W" if idx in tilde else ("Q" if n == "g" else n)
                kk = res(k)
                if kk not in self.multipoles:
                    self.fixed[kk] = kk
                self.integrals.append((X, lab, kk))
            else:
                self.onebody.append((n, lab))
        self.ranks = {fn: names.get(K, K) for fn, K in info["ranks"].items()}
        self.k_out = names.get(info["k"]) if info.get("k") else None
        # C++ names: the A^k output and the coupled K are the arguments k and K
        self.cpp = {s_: _cpp_name(s_) for s_ in set(names.values())}
        if self.k_out:
            self.cpp[self.k_out] = "k" if external == "Ak" else "K"
        self.chain = [names[c] for c in info.get("chain", [])]
        self.fields = list(fields)
        f.sixj = [self.nice_sixj(t) for t in f.sixj]

    def nice_sixj(self, t):
        """the form of a 6j with the valence letters, then the external ranks, in the top row (as
        {w v k; mu lambda b}), then the other letters, multipolarities, intermediates"""
        def rank(x):
            k = self.kind.get(x)
            if k == "val":
                return (0, self.order.index(x))
            if x in self.ranks.values() or x == self.k_out:
                return (1, x)
            if x in self.chain:
                return (2, x)
            if k:
                return (3, self.order.index(x) if x in self.order else 99)
            if x in self.multipoles or x in self.fixed:
                return (4, self.multipoles.index(x) if x in self.multipoles else 99)
            return (5, x)
        return min(sixj_forms(t), key=lambda form: [rank(x) for x in form])

    def letter_deltas(self):
        """the constraints delta_{ab} of the result (identifications of fixed symbols: letters'
        j, ranks, k, chain j's; the identifications of summed symbols collapse their sums)"""
        return list(self.constraints)

    def tex(self, eps=None, macros=False):
        """the term as LaTeX (letters stand for their j in phases, [j] and 6j symbols)"""
        f = self.f
        sym = tex_sym
        terms, c = canonical_phase(f.phase, self.half, list(self.multipoles) + self.order)
        sign = self.sign
        if c == 1:  # (-1)^1 goes into the sign
            sign, c = -sign, Fraction(0)
        out = "+" if sign > 0 else "-"
        sums = ("".join(self.dummies) and r"\sum_{%s}" % "".join(self.dummies))
        sums += "".join(r"\sum_{%s}" % sym(s) for s in self.multipoles + self.inters)
        num, den = [], []
        if terms or c:
            e = ""
            for sg, s in terms:
                e += ("-" if sg < 0 else "+" if e else "") + sym(s)
            if c:
                e += {Fraction(1): "+1", Fraction(1, 2): "+1/2", Fraction(3, 2): "-1/2"}[c] if e else \
                    {Fraction(1): "1", Fraction(1, 2): "1/2", Fraction(3, 2): "-1/2"}[c]
            num.append("(-1)^{%s}" % e)
        for s, p in sorted(f.power.items(), key=lambda sp: str(sp[0])):
            if p == 0 or s == "0":
                continue
            b = "[%s]" % sym(s)
            if abs(p) != 1:
                b = (b + "^{%s}" % abs(p)) if abs(p).denominator == 1 else r"\sqrt{%s}" % b if abs(p) == Fraction(1, 2) else "%s^{%s}" % (b, abs(p))
            (num if p > 0 else den).append(b)
        dsym = lambda x: "j_%s" % x if self.kind.get(x) else sym(x)
        for a, b in self.letter_deltas():
            a, b = (b, a) if a == "0" else (a, b)
            num.append(r"\delta_{%s %s}" % (dsym(a), dsym(b)))
        six = ""
        for t in f.sixj:
            t = [sym(x) for x in t]
            six += (r"\sixj{%s}{%s}{%s}{%s}{%s}{%s}" % tuple(t) if macros else
                    r"\begin{Bmatrix}%s&%s&%s\\%s&%s&%s\end{Bmatrix}" % tuple(t))
        for t in f.ninej:
            t = [sym(x) for x in t]
            six += r"\begin{Bmatrix}%s&%s&%s\\%s&%s&%s\\%s&%s&%s\end{Bmatrix}" % tuple(t)
        rad = []
        for X, lab, k in self.integrals:
            rad.append("%s^{%s}_{%s}" % (X, sym(k), "".join(lab)))
        for n, (p, r) in self.onebody:
            rad.append(r"\redmatel{%s}{%s}{%s}" % (p, n, r) if macros else r"\langle %s\|%s\|%s\rangle" % (p, n, r))
        coef = ("\\frac{%s}{%s}" % ("".join(num) or "1", "".join(den))) if den else "".join(num)
        eps = gd.EPS if eps is None else eps
        dens = gd.Diagram._den_tex(self.dens, eps) if self.dens else ""
        body = r"\,".join(rad)
        body = r"\frac{%s}{%s}" % (body, dens) if dens else body
        return out + sums + coef + six + body


class Reduced:
    """the angular reduction of a term or a sum of terms: terms [RTerm], external, fields;
    .tex(), .equation() (typeset in Jupyter), .ampsci(name) (C++ for ampsci)"""

    def __init__(self, rterms, external, fields, source, source_terms):
        self.terms, self.external, self.fields, self.source = rterms, external, fields, source
        self.source_terms = source_terms  # all the Goldstone terms (for the signature)

    def tex(self, eps=None, macros=False):
        """the sum as LaTeX, one term per line; macros=True uses \\sixj and \\redmatel"""
        return "\n".join(([self.chain_tex()] if self.external == "m" else []) + [t.tex(eps, macros) for t in self.terms])

    def chain_tex(self):
        """m mode: <w m_w|X|v m_v> in terms of X_{j_1...} (the chain basis along the valence line)"""
        t0 = self.terms[0]
        (v,), (w,) = t0.t.vins, t0.t.vouts
        js = [w] + [tex_sym(j) for j in t0.chain] + [v]
        ms = ["m_%s" % w] + ["m_%d" % (i + 1) for i in range(len(t0.chain))] + ["m_%s" % v]
        out = r"\langle %s\,m_%s|X|%s\,m_%s\rangle = %s" % (w, w, v, v, "".join(r"\sum_{%s}" % j for j in js[1:-1]))
        for i, f in enumerate(self.fields):
            k = tex_sym(t0.ranks[f])
            out += r"(-1)^{%s-%s}\begin{pmatrix}%s&%s&%s\\-%s&q_{%s}&%s\end{pmatrix}" % (
                js[i], ms[i], js[i], k, js[i + 1], ms[i], f, ms[i + 1])
        return out + "X_{%s}" % ",".join(js[1:-1]) + r",\quad X_{%s}=" % ",".join(js[1:-1])

    def equation(self, eps=None):
        rows = ([self.chain_tex()] if self.external == "m" else []) + ["&" + t.tex(eps) for t in self.terms]
        return gd.Equation("\\begin{aligned}\n%s\n\\end{aligned}" % " \\\\\n".join(rows))

    def _repr_latex_(self):
        return self.equation()._repr_latex_()


def angular(term, external=None, ranks=None, pair=True, **kw):
    """the angular reduction (a Reduced) of a term, a list of terms (g~ allowed), a goldstoninator
    Diagram or Grid, or a Feynman diagram: every numerator in terms of Q^k (W^k for g~ and for each
    direct-exchange pair when pair=True), reduced matrix elements, 6j symbols, phases and [j]
    factors; external and ranks as for ampsci_fullmsum (ranks={'t': 'k'} names t's rank k,
    ranks={'V': 0} makes V a scalar); other keywords go to goldstoninator"""
    items = _tilde_terms(term, **kw)
    ts = [t for t, _ in items]
    external, fields = external_of(ts, ranks, external)
    if external in ("m", "coupled") and (len(ts[0].vins) != 1 or not fields):
        raise ValueError("the reduced form for external='m' needs one valence line and a field")
    if pair:
        items = pair_exchange(items)
    return Reduced([RTerm(t, tl, external, fields, ranks) for t, tl in items], external, fields, term,
                   terms(term, **kw))


# ----------------------------------------------------------------------------------------------
# the reduced form as C++ for ampsci
# ----------------------------------------------------------------------------------------------
CPP = {"lambda": "lam", "sigma": "sig"}  # C++ names of the multipolarities, where they differ
_WZERO = """// true if every W^k_abcd vanishes by the angular selection rules
const auto Wzero = [](const DiracSpinor &a, const DiracSpinor &b,
                      const DiracSpinor &c, const DiracSpinor &d) {
  const auto [kmin, kmax] = Coulomb::k_minmax_W(a, b, c, d);
  return kmin > kmax;
};"""


RESERVED = {"result", "asum", "inv_den", "msum", "qk", "core", "excited", "tq", "tQ", "wC", "wA", "w3j", "Xj",
            "Qzero", "Wzero", "Ak", "int", "double", "const", "auto", "for", "if"}  # used by the generated C++
ARGS = {"Ak": {"k"}, "coupled": {"K"}}  # C++ arguments of each external mode


def tex_sym(s):
    """the LaTeX of a symbol: mu -> \\mu, kappa -> \\mathrm{kappa}, k_T, \\kappa as they are"""
    s = str(s)
    return TEX.get(s, s if len(s) == 1 or "_" in s or s.startswith("\\") else r"\mathrm{%s}" % s)


def _cpp_name(s):
    """the C++ name of a symbol: mu, lam, k_T -> kT, \\kappa -> kappa"""
    return CPP.get(s, s.replace("\\", "").replace("_", ""))


def _rterm_cpp(rt, acc="result", declared=()):
    """the C++ lines of one reduced term, added to acc (declared: fields whose ranks are declared
    outside)"""
    lines, depth, wrap = [], 1, [False]
    emit = lambda s: lines.append("  " * depth + s)
    f = rt.f

    def skip_if(cond):
        emit("if (%s)" % cond)
        emit("  continue;")
        wrap[0] = wrap[0] or depth == 1

    def tj(s):
        """C++ expression for 2j of a symbol"""
        if s == "0":
            return "0"
        if rt.kind.get(s):
            return "F%s.twoj()" % s
        if s in rt.inters or s in rt.chain:
            return "t" + _cpp_name(s)
        return "2 * %s" % rt.cpp.get(s, _cpp_name(s))

    # the ranks of the fields (in m mode they are set outside, for the chain)
    for field, K in rt.ranks.items():
        if field not in declared:
            emit("const int %s = op_%s.rank();" % (rt.cpp.get(K, _cpp_name(K)), field))

    # the states, with the selection rules as soon as their letters are fixed
    checks = []
    for X, lab, _ in rt.integrals:
        if X not in ("Q", "W"):
            raise ValueError("no ampsci form for the integral %s_%s" % (X, "".join(lab)))
        checks.append(("%szero(%s)" % (X, ", ".join("F" + x for x in lab)), set(lab)))
    for n, lab in rt.onebody:
        checks.append(("op_%s.isZero(%s)" % (n, ", ".join("F" + x for x in lab)), set(lab)))
    for a, b in rt.letter_deltas():  # letters as soon as their states are fixed, the rest at once
        checks.append(("%s != %s" % (tj(a), tj(b)), {x for x in (a, b) if rt.kind.get(x)}))
    vals = {x for x, k in rt.kind.items() if k == "val"}

    def ready(have):
        out = [c for c in checks if c[1] <= have]
        for c in out:
            checks.remove(c)
        return out

    for c, _ in ready(vals):
        skip_if(c)
    have = set(vals)
    for x in rt.dummies:
        emit("for (const auto &F%s : %s) {" % (x, "core" if rt.kind[x] == "core" else "excited"))
        depth += 1
        have.add(x)
        for c, _ in ready(have):
            skip_if(c)
    nstate = depth
    if rt.dens:
        emit("const double inv_den = 1.0 / (%s);" % " * ".join("(%s)" % _den_cpp(*d) for d in rt.dens))
    emit("double asum = 0.0;")

    # the multipolarities: fixed ones, then loops over the summed ones (each within the selection
    # rules of its integrals), then the intermediates of the reduction
    for k in rt.multipoles:
        kv = rt.cpp.get(k, _cpp_name(k))
        uses = [(X, lab) for X, lab, kk in rt.integrals if kk == k]
        mins, maxs = [], []
        for i, (X, lab) in enumerate(uses):
            emit("const auto [%s_min%d, %s_max%d] = Coulomb::k_minmax_%s(%s);"
                 % (kv, i, kv, i, X, ", ".join("F" + x for x in lab)))
            mins.append("%s_min%d" % (kv, i))
            maxs.append("%s_max%d" % (kv, i))
        lo = mins[0] if len(mins) == 1 else "std::max({%s})" % ", ".join(mins)
        hi = maxs[0] if len(maxs) == 1 else "std::min({%s})" % ", ".join(maxs)
        emit("for (int %s = %s; %s <= %s; ++%s) {" % (kv, lo, kv, hi, kv))
        depth += 1
        for X, lab in uses:
            if X == "Q":
                skip_if("!Coulomb::Qk_abcd_SR(%s, %s)" % (kv, ", ".join("F" + x for x in lab)))
    known = set(rt.kind) | set(rt.multipoles) | set(rt.fixed) | set(rt.chain) | set(rt.ranks.values()) | {"0"}
    for x in rt.inters:
        tri = next((t for six in f.sixj for t in _six_triads(six) if x in t
                    and len([y for y in t if y != x and y in known]) == 2), None)
        if tri is None:
            raise ValueError("no range for the intermediate %s" % x)
        p, q = [y for y in tri if y != x]
        tx = "t" + _cpp_name(x)
        emit("for (int %s = std::abs(%s - %s); %s <= %s + %s; %s += 2) {"
             % (tx, tj(p), tj(q), tx, tj(p), tj(q), tx))
        depth += 1
        known.add(x)
    for t in f.triads:
        skip_if("!Angular::triangle(%s)" % ", ".join(tj(x) for x in t))

    # the value
    terms, c = canonical_phase(f.phase, rt.half, list(rt.multipoles) + rt.order)
    sign = rt.sign
    if c == 1:  # (-1)^1 goes into the sign
        sign, c = -sign, Fraction(0)
    parts = []
    if terms or c:
        e2 = " ".join(("- " if sg < 0 else "+ ") + tj(s) for sg, s in terms)
        if c:
            e2 += " + %d" % int(2 * c)
        e2 = e2[2:] if e2.startswith("+ ") else "-" + e2[2:]
        parts.append("Angular::neg1pow_2(%s)" % e2)
    num, den = [], []
    for s, p in f.power.items():
        if p == 0 or s == "0":
            continue
        b = "(%s + 1.0)" % tj(s)
        a = abs(p)
        if a.denominator == 2:
            b = "std::sqrt(%s)" % (b if a == Fraction(1, 2) else "std::pow(%s, %d)" % (b, a.numerator))
        elif a != 1:
            b = "std::pow(%s, %d)" % (b, a)
        (num if p > 0 else den).append(b)
    if num or den:
        parts.append("%s%s" % (" * ".join(num) or "1.0", (" / (%s)" % " * ".join(den)) if den else ""))
    for six in f.sixj:
        parts.append("Angular::sixj_2(%s)" % ", ".join(tj(x) for x in six))
    for nine in f.ninej:
        parts.append("Angular::ninej_2(%s)" % ", ".join(tj(x) for x in nine))
    for X, lab, k in rt.integrals:
        parts.append("qk.%s(%s, %s)" % (X, rt.cpp.get(k, _cpp_name(k)) if k != "0" else "0", ", ".join("F" + x for x in lab)))
    for n, (p_, r) in rt.onebody:
        parts.append("op_%s.reducedME(F%s, F%s)" % (n, p_, r))
    emit("asum += %s;" % " *\n".join(parts).replace("\n", "\n" + "  " * depth + "        "))
    while depth > nstate:
        depth -= 1
        emit("}")
    emit("%s %s= asum%s;" % (acc, "+" if sign > 0 else "-", " * inv_den" if rt.dens else ""))
    while depth > 1:
        depth -= 1
        emit("}")
    if wrap[0]:
        return ["do {"] + lines + ["} while (false);"]
    if any(l.startswith("  const ") for l in lines):  # declarations of its own: a block
        return ["{"] + lines + ["}"]
    return [l[2:] for l in lines]


def _six_triads(t):
    j1, j2, j3, j4, j5, j6 = t
    return [(j1, j2, j3), (j1, j5, j6), (j4, j2, j6), (j4, j5, j3)]


def _reduced_ampsci(self, name="term_reduced", includes=True, eps=None):
    """C++ (for ampsci) of a function evaluating the reduced form, with the same signature and
    result as ampsci_fullmsum of the same terms"""
    ts = [rt.t for rt in self.terms]
    body = ["double result = 0.0;"]
    if any(X == "Q" for rt in self.terms for X, _, _ in rt.integrals):
        body += _QZERO.split("\n")
    if any(X == "W" for rt in self.terms for X, _, _ in rt.integrals):
        body += _WZERO.split("\n")
    if self.external == "m":
        body += _chain_cpp(self, eps)
    else:
        for rt in self.terms:
            body.append("")
            body.append("// %s" % rt.tex(eps))
            body += _rterm_cpp(rt)
    body.append("return result;")
    head = ("Generated by angularinator: the angular-reduced form (Q^k, W^k from the QkTable, reduced "
            "matrix elements, 6j symbols).  Returns %s, X the sum of the %d term(s) below; internal core "
            "letters run over core, excited ones over excited." % (_returns(ts[0], self.external, self.fields), len(ts)))
    return _function(name, _params(self.source_terms, self.external, self.fields), body, head, includes)


def _chain_cpp(self, eps):
    """m mode: result = sum over the chain j_1 ... of B(m) X_{j_1...}"""
    t0 = self.terms[0]
    (v,), (w,) = t0.t.vins, t0.t.vouts
    lines, depth = [], 0
    emit = lambda s: lines.append("  " * depth + s)
    ranks = t0.ranks
    for f in self.fields:
        emit("const int %s = op_%s.rank();" % (t0.cpp.get(ranks[f], _cpp_name(ranks[f])), f))
    prev_tj, prev_tm = "F%s.twoj()" % w, "tmo_%s" % w
    emit("// B = prod_i (-1)^{j_{i-1}-m_{i-1}} (j_{i-1} k_i j_i; -m_{i-1} q_i m_i), j_0 = %s, j_n = %s" % (w, v))
    chain = t0.chain
    for i, f in enumerate(self.fields):
        K, tq = t0.cpp.get(ranks[f], _cpp_name(ranks[f])), "tq_" + f
        last = i == len(self.fields) - 1
        tj = "F%s.twoj()" % v if last else "t" + _cpp_name(chain[i])
        tm = "tm_chain%d" % (i + 1)
        if not last:
            emit("for (int %s = std::abs(%s - 2 * %s); %s <= %s + 2 * %s; %s += 2) {"
                 % (tj, prev_tj, K, tj, prev_tj, K, tj))
            depth += 1
        emit("const int %s = %s - %s;" % (tm, prev_tm, tq))
        emit("const double B%d = %sAngular::neg1pow_2(%s - %s) * Angular::threej_2(%s, 2 * %s, %s, -%s, %s, %s);"
             % (i + 1, "B%d * " % i if i else "", prev_tj, prev_tm, prev_tj, K, tj, prev_tm, tq, tm))
        prev_tj, prev_tm = tj, tm
    emit("if (B%d == 0.0 || %s != tmi_%s)" % (len(self.fields), prev_tm, v))
    emit("  %s;" % ("continue" if chain else "return 0.0"))
    emit("double Xj = 0.0;")
    for rt in self.terms:
        emit("")
        emit("// %s" % rt.tex(eps))
        for l in _rterm_cpp(rt, acc="Xj", declared=self.fields):
            emit(l)
    emit("result += B%d * Xj;" % len(self.fields))
    while depth > 0:
        depth -= 1
        emit("}")
    return lines


Reduced.ampsci = _reduced_ampsci

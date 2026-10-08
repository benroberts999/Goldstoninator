#!/usr/bin/env python3
"""
goldstoninator.py -- Goldstone (time-ordered) diagrams and energy denominators from the numerator of
an MBPT term.

A term is a product of integrals.  A two-body integral g_{pqrs} = <pq|g|rs> (electron r -> p and
s -> q) is a vertical interaction line whose first end absorbs r and emits p and whose second end
absorbs s and emits q.  A one-body integral h_{pr} = <p|h|r> is a single vertex (marker) absorbing r
and emitting p; it stands for an external field (or any effective one-body operator) that absorbs an
energy omega_h.  Letters are classified as core (hole), excited (particle) or valence, and the diagram
follows from the numerator alone:

  * every non-valence letter appears once in an "out" slot (p or q) and once in an "in" slot (r or s);
    its fermion line runs from the out-slot end to the in-slot end (arrow in that direction:
    forward in time for particles, backward for holes);
  * a particle is created (out slot) before it is annihilated (in slot); a hole is created when the
    electron leaves the core state (in slot) and filled later (out slot): these inequalities order
    the vertices in time;
  * the valence letters give the external legs: in slot = incoming, out slot = outgoing; one or two
    valence lines (two incoming and two outgoing letters: an effective two-body interaction).  With
    two, the initial energy is eps_v + eps_w and the outgoing letters belong to the incoming ones in
    the order of the valence letters (v with x, w with y); the exchange diagram, whose open lines
    connect them the other way round, gets an extra minus sign.
Antisymmetrised integrals are not accepted: expand g~_pqrs = g_pqrs - g_pqsr first.

Time runs left to right, the incoming valence line enters at the top left; along the valence line
particle lines are horizontal and hole lines slope down to the left (closed loops are placed freely,
a bubble is a lens).  If some integrals are
written in the opposite convention (g_{rspq}) this is detected: the reading in which the fewest
integrals are flipped is used (flipping all of them is the time-reversed diagram).

Energy denominators (one per gap between successive vertices) follow from the same picture: the
initial energy eps_v, plus the energies omega absorbed at the vertices before the gap, plus the energies
of the hole lines crossing the gap minus those of the particle lines, the external valence legs counting
as particles where they cross.  The final valence energy never appears: energy conservation
eps_final = eps_initial + sum(omega) removes it (every one-body vertex is taken as absorption, so e.g.
h_na g_wavn and g_wnva h_an give (eps_a - eps_n + omega_h) and (eps_a - eps_n - omega_h)); when no
vertex absorbs an energy, eps_w is kept.  Each denominator is written with the valence energy first and
positive (flipping the overall sign if needed), followed by the +/- omega terms; the sign is
(-1)^(hole lines + closed loops), times the flips.

Usage (see goldstoninator.ipynb):
    import goldstoninator as gd
    d = gd.diagram('g_vbms g_asnb g_mnva')     # also 'g_{vbms} ...', 'g[v,b,m,s] ...', [('v','b','m','s'), ...]
    d                                           # shown inline in Jupyter
    d.tex()                                     # '+\\frac{g_{vbms}\\,g_{asnb}\\,g_{mnva}}{(\\varepsilon_{va}-...)...}'
    d.equation()                                # the same, typeset when it is the output of a Jupyter cell
    d.tikz()                                    # TikZ source of the picture (print it, or d.save('fig.tikz'))
    d.save('fig.svg'); d.save('fig.pdf'); d.save('fig.tex')   # .tex: standalone TikZ document
    gd.grid([...terms...], ncols=3)             # several diagrams side by side; .tex()/.equation() give the sum
    gd.diagram('S_vbms h_na g_vavn', styles={'S': 'dotted', 'h': 'dot'}, pad=0.5)
    gd.diagram('h_na g_wavn')                   # one-body vertex h absorbs omega_h: (eps_a - eps_n + omega_h)
    gd.diagram('h_na g_wavn', omega=False)      # static field: no omega (then eps_w is kept where it occurs)
    gd.diagram('h_na S_wavn', omega={'h': r'\\omega', 'S': True})   # symbol per name; True: omega_{name}
    gd.diagram('g_vamn g_mnva', labels=False)   # the picture without the orbital labels
    gd.diagram('g_vamn g_mnva', scale=60, font=12)   # a bigger picture (pixels per unit) with 12 pt labels
No dependencies beyond the standard library.

Letter classes (override with core='abcd', val='v', exc='...' or types=dict(a='core', ...)):
    core (holes) a b c d e f;  valence v w x y;  every other letter is excited (a particle)
Line styles for two-body names: wavy (default for g), doublewavy (default for X), dashed (default
otherwise), dotted, double, solid; 'double-wavy' and 'double wavy' are read as doublewavy.
Markers for one-body names: x (default; 'cross' is the same), dot, circle, square.
styles={...} on diagram() or grid() sets them for one picture, gd.STYLES['t'] = 'dot' for the session.
Energy symbols in tex(): EPS (orbital energies) and OMEGA (absorbed energies, subscripted by the name).

feynmanator.py draws Feynman diagrams (G, Q and polarisation lines, external legs) with the drawing
code of this file (class Picture).
"""

import itertools
import math
import random
import re
from typing import List, Sequence, Tuple, Union, overload

CORE = "abcdef"
EXC = "mnrspqtu"
VAL = "vwxy"
STYLES = {"g": "wavy", "X": "doublewavy"}  # interaction name -> line style (two-body) or marker (one-body)
DEFAULT_STYLE = "dashed"
DEFAULT_MARKER = "x"
DOUBLE = 0.045  # half the distance between the two strokes of a double or doublewavy line
EPS = r"\varepsilon"  # orbital-energy symbol in tex()
OMEGA = r"\omega"  # symbol of the energy absorbed at a vertex, subscripted by the vertex name
TIKZ_PREAMBLE = "\\documentclass[tikz,border=2pt]{standalone}\n\\usetikzlibrary{decorations.pathmorphing}\n"

# ----------------------------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------------------------
_G_RE = re.compile(
    r"(\\widetilde\s*|\\tilde\s*)?(?<![A-Za-z\\])([A-Za-z]+)\s*(?:_\{\s*([A-Za-z]{4}|[A-Za-z]{2})\s*\}"
    r"|_([A-Za-z]{4}|[A-Za-z]{2})(?![A-Za-z])"
    r"|[\[(]\s*([A-Za-z])\s*,\s*([A-Za-z])(?:\s*,\s*([A-Za-z])\s*,\s*([A-Za-z]))?\s*[\])])"
)


def _factors(text):
    """[(name, labels), ...] for every integral in a string; labels has 4 (two-body) or 2 entries"""
    out = []
    for m in _G_RE.finditer(text):
        if m.group(1):
            raise ValueError(
                "antisymmetrised integral in %r: expand g~_pqrs = g_pqrs - g_pqsr first"
                % text
            )
        if m.group(3) or m.group(4):
            lab = tuple(m.group(3) or m.group(4))
        else:
            lab = tuple(x for x in m.group(5, 6, 7, 8) if x)
        out.append((m.group(2), lab))
    return out


def parse_term(term):
    """term -> (names, [labels, ...]).  A string ('g_{vbms} S_{asnb} h_{na}', 'g_vbms ...',
    'g[v,b,m,s] ...') or a list of items: 'g_abcd' strings, 4- or 2-letter strings, or tuples
    (p,q,r,s) / (name,p,q,r,s) / (name,p,r)."""
    if isinstance(term, str):
        fac = _factors(term)
        if not fac:
            raise ValueError("no integrals found in %r" % term)
        return [n for n, _ in fac], [lab for _, lab in fac]
    names, gs = [], []
    for g in term:
        if isinstance(g, str) and len(g) not in (2, 4):
            fac = _factors(g)
            if len(fac) != 1:
                raise ValueError("cannot read integral %r" % g)
            names.append(fac[0][0])
            gs.append(fac[0][1])
        else:
            g = tuple(g)
            if len(g) in (3, 5):
                names.append(g[0])
                gs.append(g[1:])
            elif len(g) in (2, 4):
                names.append("g" if len(g) == 4 else "h")
                gs.append(g)
            else:
                raise ValueError("an integral needs two or four labels, got %r" % (g,))
    return names, gs


def _types(types=None, core=CORE, exc=EXC, val=VAL):
    t = {x: "core" for x in core}
    t.update({x: "exc" for x in exc})
    t.update({x: "val" for x in val})
    if types:
        t.update(types)
    return t


# ----------------------------------------------------------------------------------------------
# picture: SVG, PDF and TikZ output of a geometry; base of Diagram here and in feynmanator.py
# ----------------------------------------------------------------------------------------------
class Picture:
    """A drawing built from the primitives of self._geometry(), a dict (diagram units, y up):
        int      [(a, b, style)]     straight line a -- b; style wavy, doublewavy, dashed, dotted, double or solid
        bubble   [(a, c, b, style)]  quadratic Bezier arc a -- b with control point c
        loop     [(c, r, style)]     circle of radius r about c
        arc      [(c, r, t0, t1, style)]  circular arc about c, counterclockwise from angle t0 to t1
        lens     [(a, c1, b, c2, fill)]  the region between the arcs a -- b with control points c1 and c2,
                                     filled: hatched, crosshatched or shaded (grey); drawn under the lines
        straight [(a, b)]            plain solid line (the external legs)
        marker   [(p, style)]        x (cross), dot, circle or square at p
        arrows   [(p, d)]            arrowhead at p pointing along the unit vector d
        labels   [(text, p)]         label centred at p: a letter, or a little LaTeX math (Greek letters,
                                     subscripts, primes, + and -: \\varepsilon_{v}-\\omega), in every output
        bbox     [x0, y0, x1, y1]
    svg(), pdf(), tikz() and save() draw it; _repr_svg_ shows it inline in Jupyter.  scale is the
    size of the picture (pixels per diagram unit in the SVG and PDF), font the size of the labels in
    points (None: 0.4 of a unit, so that they scale with the picture; in TikZ, the document font)."""

    scale = 40.0
    font = None

    def font_size(self, scale):
        """the label font size in pixels at the given scale"""
        return self.font or 0.4 * scale

    def font_units(self):
        """the label font size in diagram units (at the picture's own scale)"""
        return self.font / self.scale if self.font else 0.4

    def _geometry(self):
        raise NotImplementedError

    def _term_str(self) -> str:
        return ""

    def size(self, scale=None):
        scale = scale or self.scale
        x0, y0, x1, y1 = self._geometry()["bbox"]
        return (x1 - x0) * scale, (y1 - y0) * scale

    # ------------------------------------------------------------------------------------------
    # SVG
    # ------------------------------------------------------------------------------------------
    def svg_body(self, scale=40.0, ox=0.0, oy=0.0):
        g = self._geometry()
        x0, y0, x1, y1 = g["bbox"]
        T = lambda p: (
            ox + (p[0] - x0) * scale,
            oy + (y1 - p[1]) * scale,
        )  # y up -> svg y down
        line = (
            lambda a, b, extra="": '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="black" stroke-width="1.2"%s/>'
            % (T(a) + T(b) + (extra,))
        )
        poly = (
            lambda pts: '<polyline fill="none" stroke="black" stroke-width="1.2" points="%s"/>'
            % " ".join("%.1f,%.1f" % T(p) for p in pts)
        )
        el = []
        for a, c1, b, c2, fill in g.get("lens", []):  # under everything else
            if fill == "shaded":
                el.append(
                    '<path d="M%.1f,%.1f Q%.1f,%.1f %.1f,%.1f Q%.1f,%.1f %.1f,%.1f Z" fill="#%s" stroke="none"/>'
                    % (T(a) + T(c1) + T(b) + T(c2) + T(a) + ("%02x" % round(255 * _GREY) * 3,))
                )
            else:
                for p, q in _lens_hatch(a, c1, b, c2, fill):
                    el.append(
                        '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="black" stroke-width="0.7"/>'
                        % (T(p) + T(q))
                    )
        for a, b, style in g["int"]:
            if style == "wavy":
                el.append(poly(_wave(a, b)))
            elif style == "doublewavy":
                for sgn in (1, -1):
                    el.append(poly(_wave(a, b, shift=DOUBLE * sgn)))
            elif style == "double":
                for sgn in (1, -1):
                    el.append(line(*_offset(a, b, DOUBLE * sgn)))
            else:
                el.append(line(a, b, _SVG_DASH.get(style, "")))
        for p, style in g["marker"]:
            px, py = T(p)
            r = 0.11 * scale
            if style == "dot":
                el.append(
                    '<circle cx="%.1f" cy="%.1f" r="%.1f" fill="black"/>'
                    % (px, py, 0.7 * r)
                )
            elif style == "circle":
                el.append(
                    '<circle cx="%.1f" cy="%.1f" r="%.1f" fill="white" stroke="black" stroke-width="1.2"/>'
                    % (px, py, r)
                )
            elif style == "square":
                el.append(
                    '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="white" stroke="black" stroke-width="1.2"/>'
                    % (px - r, py - r, 2 * r, 2 * r)
                )
            else:
                el.append(
                    '<path d="M%.1f,%.1f L%.1f,%.1f M%.1f,%.1f L%.1f,%.1f" stroke="black" stroke-width="1.4"/>'
                    % (px - r, py - r, px + r, py + r, px - r, py + r, px + r, py - r)
                )
        for a, b in g["straight"]:
            el.append(line(a, b))
        for a, c, b, style in g["bubble"]:
            if style == "wavy":
                el.append(poly(_wavy(*_bezier(a, c, b))))
            elif style == "doublewavy":
                for sgn in (1, -1):
                    el.append(poly(_wavy(*_bezier(a, c, b), shift=DOUBLE * sgn)))
            elif style == "double":
                for sgn in (1, -1):
                    el.append(poly(_offset_pts(*_bezier(a, c, b), DOUBLE * sgn)))
            else:
                el.append(
                    '<path d="M%.1f,%.1f Q%.1f,%.1f %.1f,%.1f" fill="none" stroke="black" stroke-width="1.2"%s/>'
                    % (T(a) + T(c) + T(b) + (_SVG_DASH.get(style, ""),))
                )
        for c, r, style in g["loop"]:
            if style == "wavy":
                el.append(poly(_wavy(*_circle(c, r))))
            elif style == "doublewavy":
                for sgn in (1, -1):
                    el.append(poly(_wavy(*_circle(c, r), shift=DOUBLE * sgn)))
            elif style == "double":
                for sgn in (1, -1):
                    el.append(
                        '<circle cx="%.1f" cy="%.1f" r="%.1f" fill="none" stroke="black" stroke-width="1.2"/>'
                        % (T(c) + ((r + DOUBLE * sgn) * scale,))
                    )
            else:
                el.append(
                    '<circle cx="%.1f" cy="%.1f" r="%.1f" fill="none" stroke="black" stroke-width="1.2"%s/>'
                    % (T(c) + (r * scale, _SVG_DASH.get(style, "")))
                )
        for c, r, t0, t1, style in g.get("arc", []):
            pts, tans = _carc(c, r, t0, t1)
            if style == "wavy":
                el.append(poly(_wavy(pts, tans)))
            elif style == "doublewavy":
                for sgn in (1, -1):
                    el.append(poly(_wavy(pts, tans, shift=DOUBLE * sgn)))
            elif style == "double":
                for sgn in (1, -1):
                    el.append(poly(_offset_pts(pts, tans, DOUBLE * sgn)))
            else:
                el.append(
                    '<polyline fill="none" stroke="black" stroke-width="1.2"%s points="%s"/>'
                    % (
                        _SVG_DASH.get(style, ""),
                        " ".join("%.1f,%.1f" % T(p) for p in pts),
                    )
                )
        for p, d in g["arrows"]:
            el.append(
                '<polygon fill="black" points="%s"/>'
                % " ".join("%.1f,%.1f" % T(q) for q in _arrowhead(p, d))
            )
        for p, d in g.get("openarrows", []):  # unfilled head: a V, stroked
            tip, l, r = _arrowhead(p, d)
            el.append(
                '<polyline fill="none" stroke="black" stroke-width="1.2" points="%s"/>'
                % " ".join("%.1f,%.1f" % T(q) for q in (l, tip, r))
            )
        for lab, p in g["labels"]:
            el.append(_svg_label(lab, *T(p), self.font_size(scale)))
        return "\n".join(el)

    def svg(self, scale=None):
        scale = scale or self.scale
        W, H = self.size(scale)
        return _svg_wrap(self.svg_body(scale), W, H)

    def _repr_svg_(self):
        return self.svg()

    # ------------------------------------------------------------------------------------------
    # PDF (written directly; vector graphics, Times-Italic labels)
    # ------------------------------------------------------------------------------------------
    def pdf_body(self, scale=40.0, ox=0.0, oy=0.0):
        g = self._geometry()
        x0, y0, x1, y1 = g["bbox"]
        T = lambda p: (ox + (p[0] - x0) * scale, oy + (p[1] - y0) * scale)
        f = lambda p: "%.2f %.2f" % T(p)
        el = ["1.2 w 0 J 0 j"]
        poly = lambda pts: "%s m %s S" % (
            f(pts[0]),
            " ".join(f(p) + " l" for p in pts[1:]),
        )
        dashed = lambda cmd, style: (
            "%s %s [] 0 d 0 J" % (_PDF_DASH[style], cmd) if style in _PDF_DASH else cmd
        )
        for a, c1, b, c2, fill in g.get("lens", []):  # under everything else
            if fill == "shaded":
                q1, q2 = _q2c(a, c1, b)
                q3, q4 = _q2c(b, c2, a)
                el.append(
                    "%.2f g %s m %s %s %s c %s %s %s c f 0 g"
                    % (_GREY, f(a), f(q1), f(q2), f(b), f(q3), f(q4), f(a))
                )
            else:
                segs = _lens_hatch(a, c1, b, c2, fill)
                if segs:
                    el.append(
                        "0.7 w %s S 1.2 w"
                        % " ".join("%s m %s l" % (f(p), f(q)) for p, q in segs)
                    )
        for a, b, style in g["int"]:
            if style == "wavy":
                el.append(poly(_wave(a, b)))
            elif style == "doublewavy":
                for sgn in (1, -1):
                    el.append(poly(_wave(a, b, shift=DOUBLE * sgn)))
            elif style == "double":
                for sgn in (1, -1):
                    a2, b2 = _offset(a, b, DOUBLE * sgn)
                    el.append("%s m %s l S" % (f(a2), f(b2)))
            else:
                el.append(dashed("%s m %s l S" % (f(a), f(b)), style))
        for p, style in g["marker"]:
            px, py = T(p)
            r = 0.11 * scale
            if style == "dot":
                el.append(_pdf_circle(px, py, 0.7 * r) + " f")
            elif style == "circle":
                el.append(_pdf_circle(px, py, r) + " S")
            elif style == "square":
                el.append("%.2f %.2f %.2f %.2f re S" % (px - r, py - r, 2 * r, 2 * r))
            else:
                el.append(
                    "1.4 w %.2f %.2f m %.2f %.2f l %.2f %.2f m %.2f %.2f l S 1.2 w"
                    % (px - r, py - r, px + r, py + r, px - r, py + r, px + r, py - r)
                )
        for a, b in g["straight"]:
            el.append("%s m %s l S" % (f(a), f(b)))
        for a, c, b, style in g["bubble"]:  # quadratic -> cubic Bezier
            if style == "wavy":
                el.append(poly(_wavy(*_bezier(a, c, b))))
            elif style == "doublewavy":
                for sgn in (1, -1):
                    el.append(poly(_wavy(*_bezier(a, c, b), shift=DOUBLE * sgn)))
            elif style == "double":
                for sgn in (1, -1):
                    el.append(poly(_offset_pts(*_bezier(a, c, b), DOUBLE * sgn)))
            else:
                c1, c2 = _q2c(a, c, b)
                el.append(
                    dashed("%s m %s %s %s c S" % (f(a), f(c1), f(c2), f(b)), style)
                )
        for c, r, style in g["loop"]:
            if style == "wavy":
                el.append(poly(_wavy(*_circle(c, r))))
            elif style == "doublewavy":
                for sgn in (1, -1):
                    el.append(poly(_wavy(*_circle(c, r), shift=DOUBLE * sgn)))
            elif style == "double":
                for sgn in (1, -1):
                    el.append(_pdf_circle(*T(c), (r + DOUBLE * sgn) * scale) + " S")
            else:
                el.append(dashed(_pdf_circle(*T(c), r * scale) + " S", style))
        for c, r, t0, t1, style in g.get("arc", []):
            pts, tans = _carc(c, r, t0, t1)
            if style == "wavy":
                el.append(poly(_wavy(pts, tans)))
            elif style == "doublewavy":
                for sgn in (1, -1):
                    el.append(poly(_wavy(pts, tans, shift=DOUBLE * sgn)))
            elif style == "double":
                for sgn in (1, -1):
                    el.append(poly(_offset_pts(pts, tans, DOUBLE * sgn)))
            else:
                el.append(dashed(poly(pts), style))
        for p, d in g["arrows"]:
            q = [T(v) for v in _arrowhead(p, d)]
            el.append("%.2f %.2f m %.2f %.2f l %.2f %.2f l f" % (q[0] + q[1] + q[2]))
        for p, d in g.get("openarrows", []):
            tip, l, r = [T(v) for v in _arrowhead(p, d)]
            el.append("%.2f %.2f m %.2f %.2f l %.2f %.2f l S" % (l + tip + r))
        for lab, p in g["labels"]:
            el.append(_pdf_label(lab, *T(p), self.font_size(scale)))
        return "\n".join(el)

    def pdf(self, scale=None):
        scale = scale or self.scale
        W, H = self.size(scale)
        return _pdf_wrap(self.pdf_body(scale), W, H)

    # ------------------------------------------------------------------------------------------
    # TikZ (coordinates in diagram units, 1 unit = `unit` cm; the same geometry as the SVG and PDF)
    # ------------------------------------------------------------------------------------------
    def tikz_body(self, unit=1.0):
        """TikZ drawing commands (no tikzpicture environment); wavy lines use the snake decoration of
        \\usetikzlibrary{decorations.pathmorphing}"""
        g = self._geometry()
        C = lambda p: "(%s,%s)" % (_num(p[0]), _num(p[1]))
        el = []
        for a, c1, b, c2, fill in g.get("lens", []):  # under everything else
            if fill == "shaded":
                q1, q2 = _q2c(a, c1, b)
                q3, q4 = _q2c(b, c2, a)
                el.append(
                    "\\fill[black!%d] %s .. controls %s and %s .. %s .. controls %s and %s .. %s -- cycle;"
                    % (round(100 * (1 - _GREY)), C(a), C(q1), C(q2), C(b), C(q3), C(q4), C(a))
                )
            else:
                segs = _lens_hatch(a, c1, b, c2, fill)
                if segs:
                    el.append(
                        "\\draw[line width=%spt] %s;"
                        % (_num(0.5 * unit), " ".join("%s -- %s" % (C(p), C(q)) for p, q in segs))
                    )
        for a, b, style in g["int"]:
            el.append("\\draw%s %s -- %s;" % (_tikz_opt(style, unit), C(a), C(b)))
        for p, style in g["marker"]:
            r = 0.11
            if style == "dot":
                el.append("\\fill %s circle (%s);" % (C(p), _num(0.7 * r)))
            elif style == "circle":
                el.append("\\draw[fill=white] %s circle (%s);" % (C(p), _num(r)))
            elif style == "square":
                el.append(
                    "\\draw[fill=white] %s rectangle %s;"
                    % (C((p[0] - r, p[1] - r)), C((p[0] + r, p[1] + r)))
                )
            else:
                el.append(
                    "\\draw[line width=%spt] %s -- %s %s -- %s;"
                    % (
                        _num(unit),
                        C((p[0] - r, p[1] - r)),
                        C((p[0] + r, p[1] + r)),
                        C((p[0] - r, p[1] + r)),
                        C((p[0] + r, p[1] - r)),
                    )
                )
        for a, b in g["straight"]:
            el.append("\\draw %s -- %s;" % (C(a), C(b)))
        for a, c, b, style in g["bubble"]:  # quadratic -> cubic Bezier
            c1, c2 = _q2c(a, c, b)
            el.append(
                "\\draw%s %s .. controls %s and %s .. %s;"
                % (_tikz_opt(style, unit), C(a), C(c1), C(c2), C(b))
            )
        for c, r, style in g["loop"]:
            el.append(
                "\\draw%s %s circle (%s);" % (_tikz_opt(style, unit), C(c), _num(r))
            )
        for c, r, t0, t1, style in g.get("arc", []):
            el.append(
                "\\draw%s %s arc[start angle=%s, end angle=%s, radius=%s];"
                % (
                    _tikz_opt(style, unit),
                    C((c[0] + r * math.cos(t0), c[1] + r * math.sin(t0))),
                    _num(math.degrees(t0)),
                    _num(math.degrees(t1)),
                    _num(r),
                )
            )
        for p, d in g["arrows"]:
            el.append(
                "\\fill %s -- %s -- %s -- cycle;"
                % tuple(C(q) for q in _arrowhead(p, d))
            )
        for p, d in g.get("openarrows", []):
            tip, l, r = _arrowhead(p, d)
            el.append("\\draw %s -- %s -- %s;" % (C(l), C(tip), C(r)))
        for lab, p in g["labels"]:
            el.append("\\node%s at %s {$%s$};" % (_tikz_font(self.font), C(p), lab))
        return "\n".join(el)

    def tikz(self, unit=1.0, standalone=False):
        """the picture as TikZ source: a tikzpicture to \\input (1 diagram unit = `unit` cm), or with
        standalone=True a complete document.  Needs \\usetikzlibrary{decorations.pathmorphing}.
        """
        body = "%% %s\n%s\n%s\n\\end{tikzpicture}" % (
            self._term_str(),
            _tikz_begin(unit),
            self.tikz_body(unit),
        )
        return _tikz_standalone(body) if standalone else body

    def save(self, path, scale=None, unit=1.0):
        """write .svg, .pdf, .tikz (tikzpicture, to \\input) or .tex (standalone TikZ document)"""
        return _save(self, path, scale, unit)


# ----------------------------------------------------------------------------------------------
# diagram: orientation, time order, lines, denominators
# ----------------------------------------------------------------------------------------------
class Diagram(Picture):
    """Attributes: verts (out1, out2, in1, in2) with out2 = in2 = None for a one-body vertex; vertex k
    has ends (k,0) and, for two-body, (k,1); names; flips; order (earliest first); level[k];
    lines [(label, 'exc'|'core', src_end, dst_end)]; vins, vouts [(label, end)], one or two each in the
    order of the valence letters (vin, vout: the first); x[end] = row after layout;
    sign after denominators()/tex().

    omega: which vertices absorb an energy (entering the denominators after them) and its tex symbol.
    True (default): every one-body vertex absorbs OMEGA_{name}; False/None: none (static fields);
    a string: that symbol at every one-body vertex; a dict name -> True | symbol | False for the
    vertices named (any vertex, also two-body ones), the others as by default."""

    LEG = 0.7

    def __init__(
        self,
        term,
        types=None,
        core=CORE,
        exc=EXC,
        val=VAL,
        styles=None,
        pad=0.35,
        omega=True,
        labels=True,
        scale=None,
        font=None,
    ):
        self.names, self.gs = parse_term(term)
        self.scale = scale or Picture.scale
        self.font = font
        self.styles = dict(STYLES)
        self.styles.update(canon_styles(styles))
        self.pad = pad
        self.omega = omega
        self.labels = labels
        self.valorder = val
        self.types = _types(types, core, exc, val)
        for x in {x for g in self.gs for x in g}:
            self.types.setdefault(x, "exc")  # any other letter is excited
        self._orient()
        self._lines()
        self._order()
        self.open = {
            x for path in self._paths() for x in path
        }  # on a valence line: drawing rule
        self.x = {}  # row of every vertex end, set by layout()

    def _term_str(self):
        return " ".join(n + "_" + "".join(g) for n, g in zip(self.names, self.gs))

    @staticmethod
    def _ends(vert):
        """[(side, out, in), ...] of a vertex"""
        o1, o2, i1, i2 = vert
        return [(0, o1, i1)] + ([(1, o2, i2)] if o2 is not None else [])

    def _valid(self, verts):
        ins, outs = {}, {}
        vin = vout = 0
        for k, v in enumerate(verts):
            for _, o, i in self._ends(v):
                if self.types[i] == "val":
                    vin += 1
                elif i in ins:
                    return False
                else:
                    ins[i] = k
                if self.types[o] == "val":
                    vout += 1
                elif o in outs:
                    return False
                else:
                    outs[o] = k
        return vin == vout and vin in (1, 2) and set(ins) == set(outs)

    def _orient(self):
        as_vert = lambda g, f: (
            ((g[1], None, g[0], None) if f else (g[0], None, g[1], None))
            if len(g) == 2
            else ((g[2], g[3], g[0], g[1]) if f else (g[0], g[1], g[2], g[3]))
        )
        sols = []
        for flips in itertools.product([0, 1], repeat=len(self.gs)):
            verts = [as_vert(g, f) for f, g in zip(flips, self.gs)]
            if self._valid(verts):
                sols.append((sum(flips), flips, verts))
        if not sols:
            raise ValueError(
                "not a valid Goldstone numerator: %s (each non-valence letter must appear once as "
                "an out index and once as an in index, one or two valence letters in and as many out)"
                % self._term_str()
            )
        sols.sort(key=lambda s: s[0])
        self.flips, self.verts = sols[0][1], sols[0][2]

    def _lines(self):
        out_end, in_end = {}, {}
        vin, vout = [], []
        for k, v in enumerate(self.verts):
            for side, o, i in self._ends(v):
                if self.types[i] == "val":
                    vin.append((i, (k, side)))
                else:
                    in_end[i] = (k, side)
                if self.types[o] == "val":
                    vout.append((o, (k, side)))
                else:
                    out_end[o] = (k, side)
        order = lambda le: (
            (
                self.valorder.index(le[0])
                if le[0] in self.valorder
                else len(self.valorder)
            ),
            le[0],
        )
        self.vins = sorted(
            vin, key=order
        )  # one or two incoming and as many outgoing valence
        self.vouts = sorted(
            vout, key=order
        )  # legs (see _valid), in the order of the valence letters
        self.vin, self.vout = self.vins[0], self.vouts[0]  # the first of each
        self.lines = [(x, self.types[x], out_end[x], in_end[x]) for x in out_end]

    def _order(self):
        n = len(self.verts)
        before = set()
        for x, kind, (ko, _), (ki, _) in self.lines:
            if ko != ki:
                before.add(
                    (ko, ki) if kind == "exc" else (ki, ko)
                )  # particle: out before in; hole: in before out
        orders = [
            o
            for o in itertools.permutations(range(n))
            if all(o.index(a) < o.index(b) for a, b in before)
        ]
        if not orders:
            raise ValueError("no consistent time ordering for " + self._term_str())
        self.order = list(orders[0])
        self.level = {k: i for i, k in enumerate(self.order)}

    # ------------------------------------------------------------------------------------------
    def _nxt(self):
        """letter -> the letter emitted at the end where it is absorbed (along the fermion lines)"""
        nxt = {}
        for v in self.verts:
            for _, o, i in self._ends(v):
                nxt[i] = o
        return nxt

    def _paths(self):
        """the letters of the open (valence) fermion lines, one list per incoming letter, from it to
        the outgoing letter"""
        nxt = self._nxt()
        paths = []
        for v, _ in self.vins:
            path = [v]
            while self.types[nxt[path[-1]]] != "val":
                path.append(nxt[path[-1]])
            paths.append(path + [nxt[path[-1]]])
        return paths

    def _crossed(self):
        """1 if the open lines pair the outgoing valence letters with the incoming ones the other way
        round than the order of the valence letters (the exchange diagram), else 0"""
        want = [w for w, _ in self.vouts]
        perm = [want.index(path[-1]) for path in self._paths()]
        return (
            sum(
                1
                for i in range(len(perm))
                for j in range(i + 1, len(perm))
                if perm[i] > perm[j]
            )
            % 2
        )

    def _loops(self):
        """number of closed fermion loops"""
        nxt = self._nxt()
        seen = {x for path in self._paths() for x in path}
        loops = 0
        for x in nxt:
            if x in seen or self.types[x] == "val":
                continue
            loops += 1
            while x not in seen:
                seen.add(x)
                x = nxt[x]
        return loops

    def omegas(self):
        """the tex symbol of the energy absorbed at each vertex (None where nothing is absorbed)"""
        spec = self.omega
        out = []
        for n, v in zip(self.names, self.verts):
            one = v[1] is None
            s = spec.get(n, one) if isinstance(spec, dict) else (spec if one else False)
            if s is True:
                s = OMEGA + ("_%s" % n if len(n) == 1 else "_{%s}" % n)
            out.append(s if isinstance(s, str) and s else None)
        return out

    def denominators(self):
        """[(plus, minus, omega), ...] per gap between successive vertices: the labels whose orbital
        energies enter with + and - sign (valence energy first and positive) and {symbol: coefficient}
        of the energies absorbed at the vertices.  The initial energy is that of the incoming valence
        letter(s); the final valence energy is removed with eps_final = eps_initial + sum(omega)
        whenever some vertex absorbs an energy (with two valence lines, once both outgoing legs
        exist).  Sets self.sign.
        """
        lv = self.level
        ins = [(v, lv[k]) for v, (k, _) in self.vins]
        outs = [(w, lv[k]) for w, (k, _) in self.vouts]
        absorbed = sorted(
            ((k, s) for k, s in enumerate(self.omegas()) if s), key=lambda ks: lv[ks[0]]
        )
        out, flips = [], 0
        for i in range(len(self.verts) - 1):
            plus, minus = [], []
            for x, kind, (ko, _), (ki, _) in self.lines:
                if kind == "exc" and lv[ko] <= i < lv[ki]:
                    minus.append(x)
                if kind == "core" and lv[ki] <= i < lv[ko]:
                    plus.append(x)
            val = {}  # the initial energy: the incoming valence letters ...
            for v, kv in ins:
                val[v] = val.get(v, 0) + 1
                if kv > i:  # ... minus an incoming leg while it still travels
                    val[v] -= 1
            created = [
                w for w, kw in outs if kw <= i
            ]  # ... minus the outgoing legs once created:
            done = len(created) == len(outs)
            if (
                done and absorbed
            ):  #     all of them, as eps_final = eps_initial + sum(omega) ...
                for v, _ in ins:
                    val[v] -= 1
            else:  #     ... or each by its own energy when nothing is absorbed (or some are missing)
                for w in created:
                    val[w] = val.get(w, 0) - 1
            om = {}
            for k, s in absorbed:  # + omega once absorbed, - omega inside eps_final
                c = (1 if lv[k] <= i else 0) - (1 if done else 0)
                om[s] = om.get(s, 0) + c
            om = {s: c for s, c in om.items() if c}
            vplus = [x for x, c in val.items() if c > 0]
            vminus = [x for x, c in val.items() if c < 0]
            if vminus and not vplus:  # valence energy first and positive
                plus, minus, vplus, vminus = minus, plus, vminus, vplus
                om = {s: -c for s, c in om.items()}
                flips += 1
            if not (plus or minus or vplus or vminus or om):
                raise ValueError(
                    "vanishing energy denominator: %s is not a linked diagram"
                    % self._term_str()
                )
            out.append((vplus + sorted(plus), vminus + sorted(minus), om))
        nh = sum(1 for _, kind, _, _ in self.lines if kind == "core")
        self.sign = (-1) ** (nh + self._loops() + flips + self._crossed())
        return out

    @staticmethod
    def _den_tex(dens, eps):
        """the energy denominators as LaTeX: (eps_{plus}-eps_{minus}+omega...)(...)"""
        E = lambda ls: eps + ("_{%s}" % "".join(ls) if len(ls) > 1 else "_%s" % ls[0])
        out = []
        for p, m, om in dens:
            terms = (
                ([(1, E(p))] if p else [])
                + ([(-1, E(m))] if m else [])
                + [(c, s) for s, c in om.items()]
            )
            if terms[0][0] < 0:  # lead with a positive term when there is one
                j = next((j for j, (c, _) in enumerate(terms) if c > 0), None)
                if j is not None:
                    terms.insert(0, terms.pop(j))
            s = ""
            for c, t in terms:
                s += (
                    ("-" if c < 0 else "+" if s else "")
                    + ("" if abs(c) == 1 else str(abs(c)))
                    + t
                )
            out.append("(%s)" % s)
        return "".join(out)

    def tex(self, eps=None):
        """the term as LaTeX: sign, numerator as given, energy denominators"""
        eps = EPS if eps is None else eps
        dens = self.denominators()
        num = r"\,".join(
            "%s_{%s}" % (n, "".join(g)) for n, g in zip(self.names, self.gs)
        )
        sign = "+" if self.sign > 0 else "-"
        return sign + (
            r"\frac{%s}{%s}" % (num, self._den_tex(dens, eps)) if dens else num
        )

    def equation(self, eps=None):
        """tex() as an Equation: typeset when it is the output of a Jupyter cell, str()/print() give the source"""
        return Equation(self.tex(eps))

    # ------------------------------------------------------------------------------------------
    # layout: a row for every vertex end, minimising crossings (x = row, y = time; rotated when drawn)
    # ------------------------------------------------------------------------------------------
    def _legs(self, x):
        """[(a, b, letter, incoming)] of the external legs, incoming ones first"""
        ymin, ymax = 0.0, float(len(self.verts) - 1)
        legs = []
        for letter, e in self.vins:
            p = (float(x[e]), float(self.level[e[0]]))
            legs.append(((p[0], ymin - self.LEG), p, letter, True))
        for letter, e in self.vouts:
            p = (float(x[e]), float(self.level[e[0]]))
            legs.append((p, (p[0], ymax + self.LEG), letter, False))
        return legs

    def _segments(self, x):
        P = lambda e: (float(x[e]), float(self.level[e[0]]))
        segs = [
            (P((k, 0)), P((k, 1))) for k, v in enumerate(self.verts) if v[1] is not None
        ]
        seen = set()
        for _, kind, a, b in self.lines:
            key = frozenset([a, b])
            if a != b and key not in seen:
                seen.add(key)
                segs.append((P(a), P(b)))
        segs += [(a, b) for a, b, _, _ in self._legs(x)]
        return segs

    def _score(self, x):
        segs = self._segments(x)
        ends = {(float(x[e]), float(self.level[e[0]])) for e in x}
        bad = cross = 0
        for i, (a, b) in enumerate(segs):
            for p in ends:
                if p != a and p != b and _on_segment(p, a, b):
                    bad += 1
            for c, d in segs[i + 1 :]:
                if _proper_cross(a, b, c, d):
                    cross += 1
                elif _collinear_overlap(a, b, c, d):
                    bad += 1
        val_x = sum(x[e] for _, e in self.vins + self.vouts)
        travel = sum(abs(x[a] - x[b]) for _, _, a, b in self.lines)
        wavy = sum(
            abs(x[(k, 0)] - x[(k, 1)]) - 1
            for k, v in enumerate(self.verts)
            if v[1] is not None
        )
        width = max(x.values()) - min(x.values())
        rule = 0  # valence line: particles level, holes down to the left (closed loops are free)
        for (
            letter,
            kind,
            a,
            b,
        ) in self.lines:  # a: emitted (out slot), b: absorbed (in slot)
            if letter in self.open:
                if kind == "exc":
                    rule += abs(x[a] - x[b])
                else:  # hole: created at b (earlier), filled at a (later): b at least a row lower
                    rule += max(0, x[a] + 1 - x[b])
        return (
            100 * bad
            + 10 * cross
            + 8 * rule
            + 1.0 * val_x
            + 0.3 * travel
            + 0.3 * wavy
            + 0.2 * width
        )

    def layout(self, nrows=None, max_eval=40000, seed=0):
        """choose the row of every vertex end, minimising _score().  Without nrows, one row more than
        there are vertices is tried first, and more rows whenever the best arrangement still has a
        line through a vertex or on top of another line (each such fault costs 100)"""
        n = len(self.verts)
        rows = [nrows] if nrows else list(range(n + 1, n + 5))
        best = self._layout(rows[0], max_eval, seed)
        for r in rows[1:]:
            if best[0] < 100:
                break
            best = self._layout(r, max_eval, seed)
        self.score, self.x = best
        return self

    def _layout(self, nrows, max_eval, seed):
        """(score, x) of the best arrangement on nrows rows"""
        n = len(self.verts)
        two = [(a, b) for a in range(nrows) for b in range(nrows) if a != b]
        one = [(a, None) for a in range(nrows)]
        k0, side = self.vin[1]  # incoming valence line in row 0 (top)
        per_vertex = [
            [o for o in (two if v[1] is not None else one) if k != k0 or o[side] == 0]
            for k, v in enumerate(self.verts)
        ]
        total = 1
        for o in per_vertex:
            total *= len(o)
        if total <= max_eval:
            cands = itertools.product(*per_vertex)
        else:
            rng = random.Random(seed)
            cands = (tuple(rng.choice(o) for o in per_vertex) for _ in range(max_eval))
        best = (math.inf, {})
        for choice in cands:
            x = {}
            for k, (a, b) in enumerate(choice):
                x[(k, 0)] = a
                if b is not None:
                    x[(k, 1)] = b
            s = self._score(x)
            if s < best[0]:
                best = (s, x)
        if total > max_eval:  # local improvement
            s, x = best
            improved = True
            while improved:
                improved = False
                for e in list(x):
                    if e == self.vin[1]:
                        continue
                    for c in range(nrows):
                        if c == x[e] or c == x.get((e[0], 1 - e[1])):
                            continue
                        y = dict(x)
                        y[e] = c
                        sy = self._score(y)
                        if sy < s:
                            s, x, improved = sy, y, True
            best = (s, x)
        return best

    # ------------------------------------------------------------------------------------------
    # geometry (time -> x, row 0 -> top), shared by the SVG and PDF output
    # ------------------------------------------------------------------------------------------
    def _geometry(self):
        if not self.x:
            self.layout()
        x = self.x
        P = lambda e: (float(x[e]), float(self.level[e[0]]))
        g = {
            "int": [],
            "marker": [],
            "straight": [],
            "bubble": [],
            "loop": [],
            "labels": [],
            "arrows": [],
        }
        extent = []

        def label(
            text, p, *room
        ):  # a label at p, with the room it takes (only when labels are on)
            if self.labels:
                g["labels"].append((text, p))
                extent.extend(room or (p,))

        for k, v in enumerate(self.verts):
            if v[1] is not None:
                g["int"].append(
                    (
                        P((k, 0)),
                        P((k, 1)),
                        self.styles.get(self.names[k], DEFAULT_STYLE),
                    )
                )
            else:
                p = P((k, 0))
                g["marker"].append((p, self.styles.get(self.names[k], DEFAULT_MARKER)))
                extent += [(p[0] - 0.2, p[1] - 0.2), (p[0] + 0.2, p[1] + 0.2)]
        pts = [P(e) for e in x]
        extent += pts
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        pairs = {}
        for x_, kind, a, b in self.lines:
            pairs.setdefault(frozenset([a, b]), []).append((x_, kind, a, b))
        for key, members in pairs.items():
            if len(key) == 1:  # line from an end to itself
                (e,) = key
                p = P(e)
                c = (p[0] + 0.25, p[1])
                g["loop"].append((c, 0.25, "solid"))
                label(members[0][0], (c[0] + 0.42, c[1]))
                g["arrows"].append(((c[0], c[1] + 0.25), (-1.0, 0.0)))
                extent += [
                    (c[0] + (0.55 if self.labels else 0.3), c[1] + 0.3),
                    (c[0], c[1] - 0.3),
                ]
            elif len(members) == 1:
                x_, kind, a, b = members[0]
                pa, pb = P(a), P(b)
                g["straight"].append((pa, pb))
                d = _unit(pa, pb)
                mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
                g["arrows"].append((mid, d))
                nrm = (-d[1], d[0])
                c1 = (mid[0] + 0.2 * nrm[0], mid[1] + 0.2 * nrm[1])
                c2 = (mid[0] - 0.2 * nrm[0], mid[1] - 0.2 * nrm[1])
                lab = c1 if _dist(c1, (cx, cy)) >= _dist(c2, (cx, cy)) else c2
                label(x_, lab)
            else:  # two lines between the same ends: bubble
                p1, p2 = sorted(P(e) for e in key)
                dc = _unit(p1, p2)
                nrm = (-dc[1], dc[0])
                for i, (x_, kind, a, b) in enumerate(members):
                    pa, pb = P(a), P(b)
                    d = _unit(pa, pb)
                    bulge = (
                        0.8 * min(_dist(pa, pb), 1.0) * (1 if i % 2 == 0 else -1)
                    )  # peak 0.4: clear of the next row
                    mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
                    g["bubble"].append(
                        (
                            pa,
                            (mid[0] + bulge * nrm[0], mid[1] + bulge * nrm[1]),
                            pb,
                            "solid",
                        )
                    )
                    top = (mid[0] + 0.5 * bulge * nrm[0], mid[1] + 0.5 * bulge * nrm[1])
                    g["arrows"].append((top, d))
                    sgn = 1 if bulge > 0 else -1
                    lab = (top[0] + 0.2 * sgn * nrm[0], top[1] + 0.2 * sgn * nrm[1])
                    label(x_, lab)
        for a, b, letter, incoming in self._legs(x):
            g["straight"].append((a, b))
            g["arrows"].append((((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), (0.0, 1.0)))
            extent += [a, b]
            if incoming:
                label(letter, (a[0] - 0.2, a[1] + 0.35), (a[0] - 0.2, a[1]))
            else:
                label(letter, (b[0] - 0.2, b[1] - 0.35), (b[0] - 0.2, b[1]))
        R = lambda p: (p[1], -p[0])  # rotate: time -> x (left to right), row 0 -> top
        g["int"] = [(R(a), R(b), st) for a, b, st in g["int"]]
        g["marker"] = [(R(p), st) for p, st in g["marker"]]
        g["straight"] = [(R(a), R(b)) for a, b in g["straight"]]
        g["bubble"] = [(R(a), R(c), R(b), st) for a, c, b, st in g["bubble"]]
        g["loop"] = [(R(c), r, st) for c, r, st in g["loop"]]
        g["arrows"] = [(R(p), R(d)) for p, d in g["arrows"]]
        g["openarrows"] = [(R(p), R(d)) for p, d in g.get("openarrows", [])]
        g["labels"] = [(l, R(p)) for l, p in g["labels"]]
        extent = [R(p) for p in extent]
        xs = [p[0] for p in extent]
        ys = [p[1] for p in extent]
        g["bbox"] = [
            min(xs) - self.pad,
            min(ys) - self.pad,
            max(xs) + self.pad,
            max(ys) + self.pad,
        ]
        return g


# ----------------------------------------------------------------------------------------------
# several diagrams side by side
# ----------------------------------------------------------------------------------------------
class Grid:
    def __init__(self, diagrams, ncols=3, scale=None, gap=6.0):
        self.diagrams, self.ncols, self.gap = diagrams, ncols, gap
        self.scale = scale or (diagrams[0].scale if diagrams else Picture.scale)

    def _place(self):
        """[(diagram, ox, oy_top, w, h)], total W, H  (y measured downwards from the top)"""
        sizes = [d.size(self.scale) for d in self.diagrams]
        nc = max(1, min(self.ncols, len(self.diagrams)))
        rows = [sizes[i : i + nc] for i in range(0, len(sizes), nc)]
        colw = [
            max((r[c][0] for r in rows if c < len(r)), default=0) for c in range(nc)
        ]
        rowh = [max(s[1] for s in r) for r in rows]
        W = sum(colw) + self.gap * (nc + 1)
        H = sum(rowh) + self.gap * (len(rows) + 1)
        out = []
        for i, d in enumerate(self.diagrams):
            r, c = divmod(i, nc)
            ox = self.gap + sum(colw[:c]) + self.gap * c
            oy = self.gap + sum(rowh[:r]) + self.gap * r
            out.append((d, ox, oy, sizes[i][0], sizes[i][1]))
        return out, W, H

    def svg(self):
        placed, W, H = self._place()
        return _svg_wrap(
            "\n".join(d.svg_body(self.scale, ox, oy) for d, ox, oy, w, h in placed),
            W,
            H,
        )

    def _repr_svg_(self):
        return self.svg()

    def pdf(self):
        placed, W, H = self._place()
        return _pdf_wrap(
            "\n".join(
                d.pdf_body(self.scale, ox, H - oy - h) for d, ox, oy, w, h in placed
            ),
            W,
            H,
        )

    def tex(self, eps=None):
        """the sum of the terms as LaTeX (each term carries its own sign), one term per line"""
        return "\n".join(d.tex(eps) for d in self.diagrams)

    def equation(self, eps=None):
        """the sum as an Equation (aligned, one term per line): typeset as the output of a Jupyter cell"""
        return Equation(
            "\\begin{aligned}\n%s\n\\end{aligned}"
            % " \\\\\n".join("&" + d.tex(eps) for d in self.diagrams)
        )

    def tikz(self, unit=1.0, standalone=False):
        """the grid as one tikzpicture (each diagram in a shifted scope), or a standalone document"""
        placed, W, H = self._place()
        parts = []
        for d, ox, oy, w, h in placed:
            x0, y0, x1, y1 = d._geometry()["bbox"]
            parts.append(
                "%% %s\n\\begin{scope}[shift={(%s,%s)}]\n%s\n\\end{scope}"
                % (
                    d._term_str(),
                    _num(ox / self.scale - x0),
                    _num(-oy / self.scale - y1),
                    _indent(d.tikz_body(unit)),
                )
            )
        body = "%s\n%s\n\\end{tikzpicture}" % (_tikz_begin(unit), "\n".join(parts))
        return _tikz_standalone(body) if standalone else body

    def save(self, path, unit=1.0):
        """write .svg, .pdf, .tikz (tikzpicture, to \\input) or .tex (standalone TikZ document)"""
        return _save(self, path, None, unit)


class Equation(str):
    """LaTeX source that Jupyter typesets when it is the output of a cell (str()/print() give the source)"""

    def __repr__(self):
        return str.__str__(self)

    def _repr_latex_(self):
        return "$\\displaystyle %s$" % self

    def _repr_markdown_(self):
        return "$$%s$$" % self


def several(term):
    """whether term is a list of terms (strings or diagrams) rather than one term (a string, or a list
    of tuples)"""
    return (
        isinstance(term, (list, tuple))
        and bool(term)
        and all(isinstance(t, (str, Picture)) for t in term)
    )


@overload
def diagram(term: Union[str, Diagram], **kw) -> Diagram: ...
@overload
def diagram(term: Sequence[tuple], **kw) -> Diagram: ...
@overload
def diagram(term: List[Union[str, Picture]], **kw) -> Grid: ...


def diagram(term, **kw):
    """the diagram of one term (string or list of integrals), or, for a list of terms, a Grid of
    them as grid() gives (ncols=, gap=); keywords: styles, pad, omega, labels, scale, font,
    core/exc/val, types"""
    if several(term):
        return grid(term, **kw)
    kw.pop("ncols", None)
    kw.pop("gap", None)
    if isinstance(term, Diagram):
        return term
    return Diagram(term, **kw).layout()


def grid(terms, ncols=3, scale=None, gap=6.0, **kw):
    """diagrams of several terms side by side (shown inline in Jupyter; .tex(), .equation(), .tikz(),
    .save('x.svg'/'x.pdf'/'x.tikz'/'x.tex')); keywords as for diagram()"""
    return Grid([diagram(t, scale=scale, **kw) for t in terms], ncols=ncols, scale=scale, gap=gap)


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _save(obj, path, scale, unit):
    if path.endswith(".svg"):
        data = obj.svg() if scale is None else obj.svg(scale)
    elif path.endswith(".pdf"):
        data = obj.pdf() if scale is None else obj.pdf(scale)
    elif path.endswith(".tikz"):
        data = obj.tikz(unit) + "\n"
    elif path.endswith(".tex"):
        data = obj.tikz(unit, standalone=True)
    else:
        raise ValueError("use a .svg, .pdf, .tikz or .tex file name")
    if isinstance(data, bytes):
        with open(path, "wb") as f:
            f.write(data)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
    return path


def _num(x):
    """a coordinate for TikZ: two decimals, no trailing zeros, no -0"""
    s = ("%.2f" % x).rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _indent(text, by="  "):
    return "\n".join(by + line for line in text.split("\n"))


def _tikz_begin(unit):
    return "\\begin{tikzpicture}[x=%scm,y=%scm,line width=%spt]" % (
        _num(unit),
        _num(unit),
        _num(0.85 * unit),
    )


def _tikz_standalone(body):
    return TIKZ_PREAMBLE + "\\begin{document}\n" + body + "\n\\end{document}\n"


def _svg_wrap(body, W, H):
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="%.0f" height="%.0f" viewBox="0 0 %.0f %.0f">\n'
        '<rect width="100%%" height="100%%" fill="white"/>\n%s\n</svg>' % (W, H, W, H, body)
    )


def _pdf_wrap(body, W, H):
    """a one-page PDF (bytes) with the given content stream, on a white page"""
    body = "1 g 0 0 %.2f %.2f re f 0 g\n%s" % (W, H, body)
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.2f %.2f] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R%s >> >> >>"
        % (W, H, " /F2 6 0 R" if "/F2" in body else ""),
        "<< /Length %d >>\nstream\n%s\nendstream" % (len(body.encode("latin-1")), body),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Times-Italic >>",
    ] + (["<< /Type /Font /Subtype /Type1 /BaseFont /Symbol >>"] if "/F2" in body else [])
    out = b"%PDF-1.4\n"
    offsets = []
    for i, o in enumerate(objs):
        offsets.append(len(out))
        out += ("%d 0 obj\n%s\nendobj\n" % (i + 1, o)).encode("latin-1")
    xref = len(out)
    out += ("xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)).encode("latin-1")
    out += "".join("%010d 00000 n \n" % off for off in offsets).encode("latin-1")
    out += (
        "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objs) + 1, xref)
    ).encode("latin-1")
    return out


def _pdf_circle(cx, cy, r):
    k = 0.5523 * r
    return (
        "%.2f %.2f m %.2f %.2f %.2f %.2f %.2f %.2f c %.2f %.2f %.2f %.2f %.2f %.2f c "
        "%.2f %.2f %.2f %.2f %.2f %.2f c %.2f %.2f %.2f %.2f %.2f %.2f c h"
        % (
            cx + r,
            cy,
            cx + r,
            cy + k,
            cx + k,
            cy + r,
            cx,
            cy + r,
            cx - k,
            cy + r,
            cx - r,
            cy + k,
            cx - r,
            cy,
            cx - r,
            cy - k,
            cx - k,
            cy - r,
            cx,
            cy - r,
            cx + k,
            cy - r,
            cx + r,
            cy - k,
            cx + r,
            cy,
        )
    )


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _on_segment(p, a, b, eps=1e-9):
    return (
        abs(_cross(a, b, p)) < eps
        and min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def _proper_cross(a, b, c, d):
    d1, d2, d3, d4 = _cross(c, d, a), _cross(c, d, b), _cross(a, b, c), _cross(a, b, d)
    return d1 * d2 < 0 and d3 * d4 < 0


def _collinear_overlap(a, b, c, d, eps=1e-9):
    if abs(_cross(a, b, c)) > eps or abs(_cross(a, b, d)) > eps:
        return False
    ux, uy = b[0] - a[0], b[1] - a[1]
    L = ux * ux + uy * uy
    if L < eps:
        return False
    t = sorted([((p[0] - a[0]) * ux + (p[1] - a[1]) * uy) / L for p in (c, d)])
    return min(t[1], 1.0) - max(t[0], 0.0) > eps


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _unit(a, b):
    L = _dist(a, b)
    return ((b[0] - a[0]) / L, (b[1] - a[1]) / L) if L > 0 else (0.0, 1.0)


def _offset(a, b, h):
    d = _unit(a, b)
    n = (-d[1] * h, d[0] * h)
    return (a[0] + n[0], a[1] + n[1]), (b[0] + n[0], b[1] + n[1])


def _arrowhead(p, d, length=0.16, width=0.08):
    tip = (p[0] + 0.5 * length * d[0], p[1] + 0.5 * length * d[1])
    base = (p[0] - 0.5 * length * d[0], p[1] - 0.5 * length * d[1])
    n = (-d[1], d[0])
    return [
        tip,
        (base[0] + width * n[0], base[1] + width * n[1]),
        (base[0] - width * n[0], base[1] - width * n[1]),
    ]


_SVG_DASH = {
    "dashed": ' stroke-dasharray="6,4"',
    "dotted": ' stroke-dasharray="1.5,3.5" stroke-linecap="round"',
}
_PDF_DASH = {"dashed": "[6 4] 0 d", "dotted": "[0.1 3.5] 0 d 1 J"}
_GREY = 0.85  # the grey level of a shaded lens (0 black, 1 white)


def _tikz_font(font):
    """the node option for labels of the given size in points ('' for the document font)"""
    if not font:
        return ""
    return "[font=\\fontsize{%s}{%s}\\selectfont]" % (_num(font), _num(1.2 * font))


def _tikz_opt(style, unit):
    """the TikZ options of a line style: '[...]' or ''"""
    snake = "decorate,decoration={snake,amplitude=%smm,segment length=%smm}" % (
        _num(0.7 * unit),
        _num(2.5 * unit),
    )
    double = "double,double distance=%spt" % _num(1.8 * unit)
    opt = {
        "wavy": snake,
        "doublewavy": double + "," + snake,
        "dashed": "dashed",
        "dotted": "dotted",
        "double": double,
    }.get(style, "")
    return "[%s]" % opt if opt else ""


# the labels: a letter, or a little LaTeX math.  A label is split into pieces (text, kind, level):
# kind it (letters, italic), greek, rm (digits and the like, upright) or op (+ - = '), level 0, -1
# (subscript) or 1 (superscript).  TikZ gets the LaTeX itself; the SVG uses Unicode, the PDF the
# Symbol font (/F2) for Greek letters and operators
_GREEK = {  # LaTeX name: (Unicode, its code in the Symbol font)
    "alpha": ("α", "a"), "beta": ("β", "b"), "gamma": ("γ", "g"), "delta": ("δ", "d"),
    "epsilon": ("ε", "e"), "varepsilon": ("ε", "e"), "zeta": ("ζ", "z"), "eta": ("η", "h"),
    "theta": ("θ", "q"), "vartheta": ("ϑ", "J"), "iota": ("ι", "i"), "kappa": ("κ", "k"),
    "lambda": ("λ", "l"), "mu": ("μ", "m"), "nu": ("ν", "n"), "xi": ("ξ", "x"),
    "pi": ("π", "p"), "varpi": ("ϖ", "v"), "rho": ("ρ", "r"), "sigma": ("σ", "s"),
    "varsigma": ("ς", "V"), "tau": ("τ", "t"), "upsilon": ("υ", "u"), "phi": ("φ", "f"),
    "varphi": ("ϕ", "j"), "chi": ("χ", "c"), "psi": ("ψ", "y"), "omega": ("ω", "w"),
    "Gamma": ("Γ", "G"), "Delta": ("Δ", "D"), "Theta": ("Θ", "Q"), "Lambda": ("Λ", "L"),
    "Xi": ("Ξ", "X"), "Pi": ("Π", "P"), "Sigma": ("Σ", "S"), "Upsilon": ("Υ", "U"),
    "Phi": ("Φ", "F"), "Psi": ("Ψ", "Y"), "Omega": ("Ω", "W"),
}  # fmt: skip
_OPS = {  # operator: (Unicode, Symbol font code, width in em)
    "-": (" − ", " - ", 1.0),
    "+": (" + ", " + ", 1.0),
    "=": (" = ", " = ", 1.0),
    "'": ("′", "\xa2", 0.3),
}
_EM = {"it": 0.5, "greek": 0.55, "rm": 0.5}  # width of a character, in em


def _label_tokens(text):
    """[(text, kind, level)] of a label (see above); braces group, _ and ^ shift the level of the next
    character or group, \\, and spaces are dropped, an unknown \\command shows its name upright"""
    out, stack, level, pending, i = [], [], 0, None, 0

    def emit(piece, kind):
        nonlocal pending
        lvl = level if pending is None else pending
        pending = None
        if out and out[-1][1] == kind and out[-1][2] == lvl and kind in ("it", "rm"):
            out[-1] = (out[-1][0] + piece, kind, lvl)
        else:
            out.append((piece, kind, lvl))

    while i < len(text):
        ch = text[i]
        if ch == "\\":
            m = re.match(r"\\([A-Za-z]+|.)", text[i:])
            if m is None:  # a backslash at the very end
                break
            name = m.group(1)
            i += m.end()
            if name in _GREEK:
                emit(name, "greek")
            elif name.isalpha():
                emit(name, "rm")
        elif ch in "_^":
            lvl = -1 if ch == "_" else 1
            i += 1
            if i < len(text) and text[i] == "{":
                stack.append(level)
                level = lvl
                i += 1
            else:
                pending = lvl
        elif ch == "{":
            stack.append(level)
            i += 1
        elif ch == "}":
            level = stack.pop() if stack else 0
            i += 1
        else:
            i += 1
            if ch in _OPS:
                emit(ch, "op")
            elif ch.isalpha():
                emit(ch, "it")
            elif not ch.isspace():
                emit(ch, "rm")
    return out


def _label_em(tokens):
    """the width of a label in em (an estimate)"""
    return sum(
        (_OPS[t][2] if kind == "op" else _EM[kind] * (1 if kind == "greek" else len(t)))
        * (0.7 if lvl else 1)
        for t, kind, lvl in tokens
    )


def label_width(text, font=0.4):
    """the width a label takes in the picture (diagram units) at a font size of font units"""
    return font * _label_em(_label_tokens(text))


def _svg_label(text, px, py, fs):
    """an SVG text element for a label centred at (px, py), font size fs (px)"""
    tokens = _label_tokens(text)
    head = (
        '<text x="%.1f" y="%.1f" font-family="serif" font-style="italic" font-size="%.0f" '
        'text-anchor="middle" dominant-baseline="middle">' % (px, py, fs)
    )
    esc = lambda t: t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if all(kind == "it" and lvl == 0 for _, kind, lvl in tokens):  # a plain word
        return head + esc("".join(t for t, _, _ in tokens)) + "</text>"
    rise = {0: 0.0, -1: 0.25 * fs, 1: -0.45 * fs}  # svg y runs down
    parts, cur = [], 0
    for t, kind, lvl in tokens:
        attrs = ' dy="%.1f"' % (rise[lvl] - rise[cur]) if lvl != cur else ""
        cur = lvl
        if lvl:
            attrs += ' font-size="%.0f"' % (0.7 * fs)
        if kind in ("rm", "op"):
            attrs += ' font-style="normal"'
        shown = _GREEK[t][0] if kind == "greek" else _OPS[t][0] if kind == "op" else t
        parts.append("<tspan%s>%s</tspan>" % (attrs, esc(shown)))
    return head + "".join(parts) + "</text>"


def _pdf_label(text, px, py, fs):
    """the PDF text object of a label centred at (px, py), font size fs"""
    tokens = _label_tokens(text)
    esc = lambda t: t.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    x = px - _label_em(tokens) * fs / 2
    parts = ["BT /F1 %.1f Tf %.2f %.2f Td" % (fs, x, py - 0.2 * fs)]
    font, rise = ("F1", fs), 0.0
    for t, kind, lvl in tokens:
        f = ("F1" if kind == "it" else "F2", 0.7 * fs if lvl else fs)
        if f != font:
            parts.append("/%s %.1f Tf" % f)
            font = f
        r = {0: 0.0, -1: -0.25 * fs, 1: 0.4 * fs}[lvl]
        if r != rise:
            parts.append("%.1f Ts" % r)
            rise = r
        shown = _GREEK[t][1] if kind == "greek" else _OPS[t][1] if kind == "op" else t
        parts.append("(%s) Tj" % esc(shown))
    if rise:
        parts.append("0 Ts")
    return " ".join(parts) + " ET"


def canon_styles(styles):
    """a styles dict as given by the user, with 'double-wavy', 'double wavy', 'Wavy' -> 'doublewavy', 'wavy'"""
    return {
        k: v.replace("-", "").replace(" ", "").lower() if isinstance(v, str) else v
        for k, v in (styles or {}).items()
    }


def _q2c(a, c, b):
    """the two control points of the cubic Bezier curve equal to the quadratic one a -> b with control point c"""
    return (
        (a[0] + 2 / 3 * (c[0] - a[0]), a[1] + 2 / 3 * (c[1] - a[1])),
        (b[0] + 2 / 3 * (c[0] - b[0]), b[1] + 2 / 3 * (c[1] - b[1])),
    )


def _hatch(poly, spacing=0.1, angle=45.0):
    """the segments [(p, q)] of the parallel lines at the given angle (degrees), spacing apart (in
    absolute position, so adjacent regions hatch in phase), that lie inside the closed polygon poly"""
    d = (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
    n = (-d[1], d[0])
    s = [p[0] * n[0] + p[1] * n[1] for p in poly]  # across the lines
    t = [p[0] * d[0] + p[1] * d[1] for p in poly]  # along them
    out = []
    for k in range(math.floor(min(s) / spacing) + 1, math.ceil(max(s) / spacing)):
        s0 = k * spacing
        cross = sorted(
            t[i] + (s0 - s[i]) / (s[j] - s[i]) * (t[j] - t[i])
            for i, j in zip(range(len(poly)), list(range(1, len(poly))) + [0])
            if (s[i] < s0) != (s[j] < s0)
        )
        for t0, t1 in zip(cross[::2], cross[1::2]):
            out.append(
                (
                    (s0 * n[0] + t0 * d[0], s0 * n[1] + t0 * d[1]),
                    (s0 * n[0] + t1 * d[0], s0 * n[1] + t1 * d[1]),
                )
            )
    return out


def _lens_hatch(a, c1, b, c2, fill, spacing=0.1):
    """the hatching of the lens between the arcs a -> b with control points c1 and c2: lines at 45
    degrees, or both ways when fill is crosshatched"""
    poly = _bezier(a, c1, b, 60)[0] + _bezier(b, c2, a, 60)[0]
    angles = (45.0, -45.0) if fill == "crosshatched" else (45.0,)
    return [seg for ang in angles for seg in _hatch(poly, spacing, ang)]


def _bezier(a, c, b, npts=120):
    """points and unit tangents along the quadratic Bezier curve a -> b with control point c"""
    pts, tans = [], []
    for i in range(npts + 1):
        t = i / npts
        u = 1 - t
        pts.append(
            (
                u * u * a[0] + 2 * u * t * c[0] + t * t * b[0],
                u * u * a[1] + 2 * u * t * c[1] + t * t * b[1],
            )
        )
        dx, dy = 2 * u * (c[0] - a[0]) + 2 * t * (b[0] - c[0]), 2 * u * (
            c[1] - a[1]
        ) + 2 * t * (b[1] - c[1])
        L = math.hypot(dx, dy)
        tans.append((dx / L, dy / L) if L > 0 else (1.0, 0.0))
    return pts, tans


def _circle(c, r, npts=120):
    """points and unit tangents around the circle of radius r about c (counter-clockwise from the bottom)"""
    pts, tans = [], []
    for i in range(npts + 1):
        th = 2 * math.pi * i / npts - math.pi / 2
        pts.append((c[0] + r * math.cos(th), c[1] + r * math.sin(th)))
        tans.append((-math.sin(th), math.cos(th)))
    return pts, tans


def _carc(c, r, t0, t1, npts=None):
    """points and unit tangents along the circular arc about c from angle t0 to t1 (counterclockwise)"""
    npts = npts or max(12, int(24 * r * abs(t1 - t0)) + 1)
    pts, tans = [], []
    for i in range(npts + 1):
        th = t0 + (t1 - t0) * i / npts
        pts.append((c[0] + r * math.cos(th), c[1] + r * math.sin(th)))
        tans.append((-math.sin(th), math.cos(th)))
    return pts, tans


def _wavy(
    pts: Sequence[Tuple[float, float]],
    tans: Sequence[Tuple[float, float]],
    amp=0.07,
    wavelength=0.25,
    shift=0.0,
):
    """a wavy line along a curve: sinusoid in the arc length, displaced along the normal (by shift
    on top, for the two strokes of a double wavy line)"""
    k = 2 * math.pi / wavelength
    out, s = [], 0.0
    for i, (p, d) in enumerate(zip(pts, tans)):
        if i:
            s += _dist(pts[i - 1], p)
        h = shift + amp * math.sin(k * s)
        out.append((p[0] - d[1] * h, p[1] + d[0] * h))
    return out


def _offset_pts(pts, tans, h):
    """the curve displaced by h along its normal"""
    return [(p[0] - d[1] * h, p[1] + d[0] * h) for p, d in zip(pts, tans)]


def _wave(a, b, amp=0.07, wavelength=0.25, npts=120, shift=0.0):
    """a wavy straight line a -- b, displaced by shift along its normal"""
    L = _dist(a, b)
    d = _unit(a, b)
    n = (-d[1], d[0])
    k = 2 * math.pi / wavelength
    return [
        (
            a[0] + d[0] * t + n[0] * (shift + amp * math.sin(k * t)),
            a[1] + d[1] * t + n[1] * (shift + amp * math.sin(k * t)),
        )
        for t in (L * i / npts for i in range(npts + 1))
    ]

#!/usr/bin/env python3
"""
feynmanator.py -- Feynman diagrams from a product of propagators, drawn with the code of
goldstoninator.py (same SVG, PDF and TikZ output, same grid of several diagrams).

    import feynmanator as fd
    d = fd.diagram("v(1) G(1,2) w(2) Q(1,2)")   # exchange (Fock) self-energy
    d                                             # the picture, as the last line of a Jupyter cell
    print(d.tikz())                               # TikZ source;  d.save('fig.svg' / '.pdf' / '.tikz' / '.tex')
    fd.grid(["v(1) w(1) Q(1,2) G(2,2)", "v(1) G(1,2) w(2) Q(1,2)"], ncols=2)   # side by side

A term is a product of factors name(label) or name(label,label); a label (letters or digits) names a
vertex, any label will do.  The factors are
    G(1,2)       internal line (Green's function): solid, no arrow.  G(1,1) is a closed loop (tadpole)
    Gex(1,2)     the excited part of G: a line with two arrowheads;  Pa(1,2): the core projector |a><a|: double line.
                 Both count as fermion lines for the layout (straight line, loops), like G
    Q(1,2)       Coulomb line: wavy, bowing gently outwards (curved=False draws them all straight)
                 vertical=('Qi', ...) forces the lines of those names to run vertically (needs straight=False
                 when they join two vertices of the fermion line)
    Qs(1,2)      a Coulomb line drawn straight
    Qu(1,2), Qd(1,2)  a Coulomb line bent upwards or downwards (left or right when it is vertical),
                 for when the side chosen automatically is not the one wanted
    PI(1,2)      polarisation loop = G(1,2) G(2,1): two solid arcs (a bubble); also Pi, \\Pi
    v(1), w(2)   external lines (names v w x y): solid with an arrow.  The first one written comes in
                 at the top left, the second goes out at the right; at most one of each
    T(1)         any other name with one label: a marker at that vertex (a cross; styles={'T': 'dot'})
    S(1,2)       any other name with two labels: a dashed line (styles={'S': 'dotted'})
Nothing is drawn at a plain vertex and there are no labels.  Line styles: wavy, dashed, dotted, double,
solid, arrow and arrows (solid with one filled or two open arrowheads at the middle, pointing from the first label to
the second; styles={'G': 'double'} for dressed propagators); markers: x (= cross), dot, circle, square.

Layout: the vertices go on a small grid, with the incoming vertex at the top left, the outgoing one rightmost
and the fermion line (the G lines from one to the other) straight, left to right (straight=False frees
it).  Placements are scored (lines through a vertex, overlapping and crossing lines, length, bends,
size, tilted bubbles) and the best one is drawn; lines between the same two vertices are bent into arcs
on alternate sides, so lines never coincide.  A closed loop of G lines runs counterclockwise in the
direction of propagation (G(a,b) goes from b to a): writing a loop the other way round mirrors it, which
puts a vertex on the other side of the loop.  A loop of three or more G lines is drawn round: the
circle through its vertices (a rounded ring for more than three), so an insertion sits on the curve.
diagram(term).layout(cols=, rows=) sets the grid.
No dependencies beyond the standard library (and goldstoninator.py).
"""

import itertools
import math
import re

import goldstoninator as gd

EXT = "vwxy"  # names of the external lines
GREEN = "G"  # the internal (fermion) line
FERMIONS = (GREEN, "Gex", "Pa")  # names laid out as fermion lines: G, its excited part, the core projector
COULOMB = "Q"  # the Coulomb line
COULOMB_STRAIGHT = "Qs"  # a Coulomb line drawn straight
COULOMB_UP = "Qu"  # a Coulomb line bent upwards (to the left when it is vertical)
COULOMB_DOWN = "Qd"  # ... downwards (to the right)
COULOMBS = (COULOMB, COULOMB_STRAIGHT, COULOMB_UP, COULOMB_DOWN)
ARROWS = {"arrow": 1, "arrows": 2}  # line styles drawn solid with one filled or two open arrowheads at the middle
POLAR = ("PI", "Pi", "\\Pi")  # names of the polarisation loop
STYLES = {
    GREEN: "solid",
    "Gex": "arrows",
    "Pa": "double",
    COULOMB: "wavy",
    COULOMB_STRAIGHT: "wavy",
    COULOMB_UP: "wavy",
    COULOMB_DOWN: "wavy",
}  # name -> line style (two labels) or marker (one label)

_F_RE = re.compile(
    r"(\\?[A-Za-z]+)\s*\(\s*([A-Za-z0-9]+)\s*(?:,\s*([A-Za-z0-9]+)\s*)?\)"
)


def parse_term(term):
    """[(name, label) or (name, label, label), ...] from a string or a list of tuples"""
    if not isinstance(term, str):
        return [tuple(str(x) for x in f) for f in term]
    rest = _F_RE.sub(" ", term).replace("*", " ")
    if rest.strip():
        raise ValueError("cannot read %r in %r" % (rest.split()[0], term))
    out = [
        (m.group(1),) + tuple(x for x in m.groups()[1:] if x)
        for m in _F_RE.finditer(term)
    ]
    if not out:
        raise ValueError("no factors in %r" % term)
    return out


# ----------------------------------------------------------------------------------------------
class Diagram(gd.Picture):
    """Attributes: factors; vertices (labels, in order of appearance); edges [(name, u, v, style)];
    markers {vertex: style}; ext [(name, vertex)], incoming first; vin, vout (vertex or None);
    chain (the vertices of the fermion line, vin to vout along G lines, kept on one row when
    straight=True); loops (closed loops of G lines, in the direction of propagation, drawn
    counterclockwise); pos {vertex: (col, row)} and score after layout()."""

    LEG = 0.7  # length of the external legs
    LOOP = 0.3  # radius of a closed loop

    def __init__(
        self, term, styles=None, ext=EXT, pad=0.35, straight=True, curved=True, vertical=()
    ):
        self.factors = parse_term(term)
        self.vertical = set(vertical)  # names of lines that must run vertically
        self.styles = dict(STYLES)
        self.styles.update(styles or {})
        self.pad = pad
        self.vertices, self.edges, self.markers, self.ext = [], [], {}, []
        for name, *labels in self.factors:
            for l in labels:
                if l not in self.vertices:
                    self.vertices.append(l)
            if len(labels) == 1:
                if name in ext:
                    self.ext.append((name, labels[0]))
                else:
                    self.markers[labels[0]] = self.styles.get(name, gd.DEFAULT_MARKER)
            elif name in POLAR:
                st = self.styles.get(name, self.styles.get(GREEN, "solid"))
                self.edges += [(name, labels[0], labels[1], st)] * 2
            else:
                st = self.styles.get(name, gd.DEFAULT_STYLE)
                self.edges.append((name, labels[0], labels[1], st))
        if len(self.ext) > 2:
            raise ValueError(
                "at most two external lines (one in, one out): " + self._term_str()
            )
        self.vin = self.ext[0][1] if self.ext else None
        self.vout = self.ext[1][1] if len(self.ext) > 1 else None
        self.chain = _chain(self.edges, self.vin, self.vout) if straight else []
        self.loops = _loops(self.edges)
        self.curved = curved
        self.pos = {}

    def _term_str(self):
        return " ".join("%s(%s)" % (f[0], ",".join(f[1:])) for f in self.factors)

    # ------------------------------------------------------------------------------------------
    # the lines for given vertex positions (shared by the layout score and the drawing); pos may
    # hold only some of the vertices while the layout is being built
    # ------------------------------------------------------------------------------------------
    def _pieces(self, pos):
        """[(kind, data, style, points, ends, bent, dirn)]: kind/data = line (a, b) | arc (a, c, b), a
        Bezier arc with control point c | ring (c, r, t0, t1), a circular arc of a round loop |
        loop (c, r), a line from a vertex to itself; points approximate the line (for the crossing
        tests); ends are the vertex positions it is attached to; bent is the cost of a line that
        does not run straight because a vertex is in the way (100 for a fermion line, which should
        never bend, 8 otherwise); dirn = (position of the first label, of the second), None for a loop at
        one vertex"""
        P = lambda v: (float(pos[v][0]), float(pos[v][1]))
        pts = [P(v) for v in self.vertices if v in pos]
        cen = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        groups = {}
        for e in self.edges:
            if e[1] in pos and e[2] in pos:
                groups.setdefault(frozenset(e[1:3]), []).append(e)
        out = []
        rings = self._rings(pos, P)
        for key, members in groups.items():
            if (
                len(key) == 1
            ):  # closed loop(s) at one vertex, on the side away from the others
                (u,) = key
                p = P(u)
                d = _away(p, cen)
                for i, (name, _, _, st) in enumerate(members):
                    ang = math.atan2(d[1], d[0]) + (i - (len(members) - 1) / 2) * 1.2
                    c = (
                        p[0] + self.LOOP * math.cos(ang),
                        p[1] + self.LOOP * math.sin(ang),
                    )
                    out.append(
                        (
                            "loop",
                            (c, self.LOOP),
                            st,
                            gd._circle(c, self.LOOP, 12)[0],
                            (p,),
                            0,
                            None,
                        )
                    )
                continue
            a, b = sorted(P(v) for v in key)
            d = gd._unit(a, b)
            n = (-d[1], d[0])
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            far = _side(n, (mid[0] - cen[0], mid[1] - cen[1]))
            members = sorted(
                members, key=lambda e: e[0] not in FERMIONS
            )  # the fermion line goes straight
            m = len(members)
            if (
                m == 2 and members[0][0] == members[1][0]
            ):  # polarisation loop: a symmetric lens
                levels = [1, -1]
            else:
                levels = [0, 1, -1, 2, -2, 3, -3][:m]
            h = min(max(0.8 * gd._dist(a, b), 0.7), 1.2)
            for (name, u, v, st), k in zip(members, levels):
                dirn = (P(u), P(v))
                forced = name in (COULOMB_UP, COULOMB_DOWN)  # the side is prescribed
                up = _side(
                    n, (0.0, 0.0)
                )  # the sign that bulges upwards (left when vertical)
                side = (up if name == COULOMB_UP else -up) if forced else far
                if k != 0:
                    c, poly = _arc(
                        a, b, mid, n, side * abs(k) * h if forced else far * k * h
                    )
                    out.append(("arc", (a, c, b), st, poly, (a, b), 0, dirn))
                    continue
                through = _hits(
                    [a, b], pts, (a, b)
                )  # the straight line would pass a vertex
                if m == 1 and key in rings:  # a line of a round loop
                    c, r, t0, t1 = rings[key]
                    poly = _ring_pts(c, r, t0, t1, 8)
                    out.append(
                        (
                            "ring",
                            (c, r, t0, t1),
                            st,
                            poly,
                            (a, b),
                            100 if through else 0,
                            dirn,
                        )
                    )
                elif (
                    m == 1 and through
                ):  # bend it round the vertex, the higher the longer it is
                    hb = max(0.8, 0.35 * gd._dist(a, b))
                    if forced:
                        c, poly = _arc(a, b, mid, n, side * hb)
                    else:
                        arcs = [_arc(a, b, mid, n, s * hb) for s in (far, -far)]
                        c, poly = min(arcs, key=lambda cp: _hits(cp[1], pts, (a, b)))
                    out.append(
                        (
                            "arc",
                            (a, c, b),
                            st,
                            poly,
                            (a, b),
                            100 if name in FERMIONS else 8,
                            dirn,
                        )
                    )
                elif m == 1 and (
                    forced or (name == COULOMB and self.curved)
                ):  # photons bow outwards
                    c, poly = _arc(
                        a, b, mid, n, side * min(max(0.4 * gd._dist(a, b), 0.35), 0.7)
                    )
                    out.append(("arc", (a, c, b), st, poly, (a, b), 0, dirn))
                else:
                    out.append(("line", (a, b), st, [a, b], (a, b), 0, dirn))
        return out

    def _rings(self, pos, P):
        """the G lines of the closed loops with three or more vertices, drawn as arcs of a circle:
        {pair of vertices: (centre, radius, start angle, end angle)}.  Three vertices lie on the
        circle through them; with more, every line is an arc of the loop's mean radius bulging
        outwards (a rounded polygon)"""
        rings = {}
        for loop in self.loops:
            if not all(v in pos for v in loop):
                continue
            q = [P(v) for v in loop]
            cen = (sum(p[0] for p in q) / len(q), sum(p[1] for p in q) / len(q))
            circle = _circumcircle(q[0], q[1], q[2]) if len(q) == 3 else None
            if len(q) == 3 and circle is None:  # collinear: no circle
                continue
            for i in range(len(q)):
                a, b = q[i], q[(i + 1) % len(q)]
                if circle is not None:
                    c, r = circle
                    arc = _arc_avoiding(c, r, a, b, q[(i + 2) % 3])
                else:
                    L = gd._dist(a, b)
                    r = max(sum(gd._dist(p, cen) for p in q) / len(q), L / 2 + 1e-6)
                    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                    d = gd._unit(a, b)
                    nrm = (-d[1], d[0])
                    inward = _side(nrm, (cen[0] - mid[0], cen[1] - mid[1]))
                    k = math.sqrt(r * r - L * L / 4)
                    c = (mid[0] + inward * k * nrm[0], mid[1] + inward * k * nrm[1])
                    arc = _arc_avoiding(c, r, a, b, cen)
                rings[frozenset((loop[i], loop[(i + 1) % len(q)]))] = arc
        return rings

    def _legs(self, pos):
        """[(a, b)] of the external legs, drawn left to right"""
        legs = []
        vin, vout = self.vin, self.vout
        if vin is not None and vin in pos:
            x, y = pos[vin]
            legs.append(((x - self.LEG, float(y)), (float(x), float(y))))
        if vout is not None and vout in pos:
            x, y = pos[vout]
            legs.append(((float(x), float(y)), (x + self.LEG, float(y))))
        return legs

    # ------------------------------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------------------------------
    def _score(self, pos):
        """lower is better: 100 per line through a vertex or on top of another line, 10 per crossing,
        10 per pair of lines that touch or nearly touch, 100 per fermion line and 8 per other line bent
        round a vertex, the total length, 0.5 per unit of width and 0.75 per unit of height, 0.75 per
        diagonal line and 2.5 per line at any other angle, 3 per bubble that does not lie flat, 5 per
        Coulomb line that is neither horizontal nor vertical, 20 per closed loop of G lines not running
        counterclockwise, 5 per vertex where a line continues straight into a line of another kind,
        6 per unit the centre of a round loop is off the midpoint of its two attachment points
        """
        P = lambda v: (float(pos[v][0]), float(pos[v][1]))
        pts = [P(v) for v in self.vertices if v in pos]
        pieces = self._pieces(pos) + [
            ("leg", ab, "solid", list(ab), (ab[0] if ab[0] in pts else ab[1],), 0, None)
            for ab in self._legs(pos)
        ]
        bad = cross = bent = touch = 0
        for _, _, _, poly, ends, b, _ in pieces:
            bad += _hits(poly, pts, ends)
            bent += b
        segs = [
            [(poly[i], poly[i + 1]) for i in range(len(poly) - 1)]
            for _, _, _, poly, _, _, _ in pieces
        ]
        box = [
            (
                min(p[0] for p in poly),
                min(p[1] for p in poly),
                max(p[0] for p in poly),
                max(p[1] for p in poly),
            )
            for _, _, _, poly, _, _, _ in pieces
        ]
        ends_of = [set(e) for _, _, _, _, e, _, _ in pieces]
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                if (
                    box[i][2] + 0.15 < box[j][0]
                    or box[j][2] + 0.15 < box[i][0]
                    or box[i][3] + 0.15 < box[j][1]
                    or box[j][3] + 0.15 < box[i][1]
                ):
                    continue
                shared = ends_of[i] & ends_of[j]
                curved = (
                    len(segs[i]) > 1 or len(segs[j]) > 1
                )  # two straight lines cannot graze
                near = [  # segments next to a vertex the two lines share: they touch there anyway
                    [
                        any(min(gd._dist(q, p) for q in seg) < 0.35 for p in shared)
                        for seg in segs[k]
                    ]
                    for k in ((i, j) if curved else ())
                ]
                crossing = close = 0
                for si, (a, b) in enumerate(segs[i]):
                    for sj, (c, d) in enumerate(segs[j]):
                        if gd._proper_cross(a, b, c, d):
                            crossing += 1
                        elif gd._collinear_overlap(a, b, c, d):
                            bad += 1
                        elif (
                            curved
                            and not (near[0][si] or near[1][sj])
                            and min(a[0], b[0]) - 0.15 <= max(c[0], d[0])
                            and min(c[0], d[0]) - 0.15 <= max(a[0], b[0])
                            and min(a[1], b[1]) - 0.15 <= max(c[1], d[1])
                            and min(c[1], d[1]) - 0.15 <= max(a[1], b[1])
                            and _seg_gap(a, b, c, d) < 0.15
                        ):
                            close += 1  # grazing
                cross += crossing
                touch += not crossing and close > 0
        pairs = {}
        for name, u, v, _ in self.edges:
            if u != v and u in pos and v in pos:
                pairs.setdefault(frozenset((u, v)), set()).add(name)
        length = bends = tilted = askew = upright = 0.0
        for key, names in pairs.items():  # every connected pair once
            u, v = key
            dx, dy = abs(pos[u][0] - pos[v][0]), abs(pos[u][1] - pos[v][1])
            length += math.hypot(dx, dy)
            bends += (
                0.75 if dx == dy else 2.5 * (dx != 0 and dy != 0)
            )  # diagonal, or any other angle
            tilted += dy != 0 and any(nm in POLAR for nm in names)  # bubbles lie flat
            askew += (
                dx != 0 and dy != 0 and not names.isdisjoint(COULOMBS)
            )  # Coulomb lines run along an axis
            upright += dx != 0 and not names.isdisjoint(self.vertical)  # forced vertical lines
        at = (
            {}
        )  # vertex -> [(name, direction)] of the single straight lines meeting there
        for key, names in pairs.items():
            if (
                len(names) == 1
                and sum(frozenset(e[1:3]) == key for e in self.edges) == 1
            ):
                u, v = key
                (name,) = names
                d = gd._unit(P(u), P(v))
                at.setdefault(u, []).append((name, d))
                at.setdefault(v, []).append((name, (-d[0], -d[1])))
        through = 0  # a line continuing straight into a line of another kind looks like one line
        for lines in at.values():
            for i, (n1, d1) in enumerate(lines):
                for n2, d2 in lines[i + 1 :]:
                    through += n1 != n2 and d1[0] * d2[0] + d1[1] * d2[1] < -0.999
        offset = 0.0  # a round loop hangs centred between its two attachment points
        for loop in self.loops:
            if len(loop) < 3 or not all(v in pos for v in loop):
                continue
            q = [P(v) for v in loop]
            circle = _circumcircle(q[0], q[1], q[2]) if len(q) == 3 else None
            centre = (
                circle[0]
                if circle
                else (sum(p[0] for p in q) / len(q), sum(p[1] for p in q) / len(q))
            )
            hooks = [
                P(v)
                for v in loop
                if any(
                    (e[1] == v or e[2] == v)
                    and (e[0] not in FERMIONS or e[1] not in loop or e[2] not in loop)
                    for e in self.edges
                )
            ]
            if len(hooks) == 2:
                offset += gd._dist(
                    centre,
                    ((hooks[0][0] + hooks[1][0]) / 2, (hooks[0][1] + hooks[1][1]) / 2),
                )
        turned = 0
        for loop in self.loops:  # counterclockwise in the direction of propagation
            if all(v in pos for v in loop):
                area = sum(
                    pos[a][0] * pos[b][1] - pos[b][0] * pos[a][1]
                    for a, b in zip(loop, loop[1:] + loop[:1])
                )
                turned += area <= 0
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        return (
            100 * bad
            + 10 * cross
            + bent
            + 1.0 * length
            + 0.5 * (max(xs) - min(xs))
            + 0.75 * (max(ys) - min(ys))
            + 1.0 * bends
            + 3 * tilted
            + 5 * askew
            + 1000 * upright
            + 10 * touch
            + 6 * offset
            + 20 * turned
            + 5 * through
        )

    def layout(self, cols=None, rows=None, max_eval=10000):
        """put the vertices on a grid minimising _score().  Grids up to cols x rows are tried (by
        default up to n x about n/4 for n vertices), smallest first: every placement when there are at
        most max_eval of them, otherwise the fermion line is placed and the other vertices are added
        one at a time, each at its best cell; the best few placements are then improved by moving and
        swapping single vertices.  The incoming vertex is at the top left, the outgoing one rightmost and
        the fermion line (self.chain) on the top row, left to right."""
        n = len(self.vertices)
        vin, vout, chain = self.vin, self.vout, self.chain
        k = len(chain)
        cmax = cols or max(2, n)
        rmax = rows or max(
            2, (n + 2) // 4 + 1, 4 if self.loops else 0
        )  # a round loop needs room
        if (
            vin is not None and vin == vout
        ):  # in and out at one vertex: everything in a column
            rmax = max(rmax, n)
        grids = sorted(
            (
                (c, r)
                for c in ([cols] if cols else range(max(1, k), cmax + 1))
                for r in ([rows] if rows else range(1, rmax + 1))
                if c * r >= n and (c * r <= 2 * n + 2 or (cols and rows))
            ),
            key=lambda g: (g[0] * g[1], g[1]),
        )
        others = [v for v in self.vertices if v not in chain and v != vin]

        def ok(pos):
            xs = [p[0] for p in pos.values()]
            if min(xs) != 0 or (
                vin is not None and pos[vin] != (0, max(p[1] for p in pos.values()))
            ):
                return False
            if vout is not None and pos[vout][0] != max(xs):
                return False
            return all(
                pos[a][1] == pos[b][1] and pos[a][0] < pos[b][0]
                for a, b in zip(chain, chain[1:])
            )

        def heads(c, r):
            """placements of the fermion line (or of the incoming vertex alone): [(vertex, cell)]"""
            if chain:  # on the top row
                for xs in itertools.combinations(range(1, c), k - 1):
                    yield list(zip(chain, [(0, r - 1)] + [(x, r - 1) for x in xs]))
            elif vin is not None:
                yield [(vin, (0, r - 1))]
            else:
                yield []

        def greedy(head, cells):
            """the other vertices one at a time, in order of appearance, each at its best free cell"""
            pos = dict(head)
            for v in others:
                xmax = (
                    pos[vout][0] if vout is not None and vout in pos else cells[-1][0]
                )
                free = [c for c in cells if c not in pos.values() and c[0] <= xmax]
                if not free:  # no room left of the outgoing vertex
                    return None
                pos[v] = min(free, key=lambda c: self._score({**pos, v: c}))
            return pos

        scored = []
        for c, r in grids:
            cells = [(x, y) for x in range(c) for y in range(r)]
            hs = list(heads(c, r))
            free_n = len(cells) - (k if chain else (1 if vin is not None else 0))
            if len(hs) * math.perm(free_n, len(others)) <= max_eval:
                cands = (
                    dict(head + list(zip(others, rest)))
                    for head in hs
                    for rest in itertools.permutations(
                        [x for x in cells if x not in dict(head).values()], len(others)
                    )
                )
            else:
                cands = (greedy(head, cells) for head in hs)
            scored += [(self._score(p), p) for p in cands if p is not None and ok(p)]
        scored.sort(key=lambda t: t[0])
        if not scored:
            raise ValueError(
                "no layout found for %s: pass cols=/rows=" % self._term_str()
            )
        best = scored[0]
        cells = [(x, y) for x in range(cmax) for y in range(rmax)]
        for s, pos in scored[:8]:
            s, pos = self._improve(s, pos, cells, ok)
            if s < best[0]:
                best = (s, pos)
        self.score, self.pos = best
        return self

    def _improve(self, s, pos, cells, ok):
        """move one vertex to a free cell, or swap two vertices, while it lowers the score"""
        improved = True
        while improved:
            improved = False
            for v in self.vertices:
                for c in cells:
                    if c == pos[v]:
                        continue
                    q = dict(pos)
                    for u in pos:
                        if pos[u] == c:
                            q[u] = pos[v]
                    q[v] = c
                    if ok(q):
                        sq = self._score(q)
                        if sq < s - 1e-9:
                            s, pos, improved = sq, q, True
        return s, pos

    # ------------------------------------------------------------------------------------------
    def _geometry(self):
        if not self.pos:
            self.layout()
        pos = self.pos
        P = lambda v: (float(pos[v][0]), float(pos[v][1]))
        g = {
            k: []
            for k in (
                "int",
                "marker",
                "straight",
                "bubble",
                "arc",
                "loop",
                "labels",
                "arrows",
                "openarrows",
            )
        }
        extent = [P(v) for v in self.vertices]
        for kind, data, st, poly, _, _, dirn in self._pieces(pos):
            if st in ARROWS and dirn is not None:  # one filled head, or two open ones
                g["arrows" if ARROWS[st] == 1 else "openarrows"] += _heads(poly, dirn, ARROWS[st])
            if st in ARROWS:
                st = "solid"
            if kind == "line":
                a, b = data
                g["int"].append((a, b, st))
            elif kind == "arc":
                a, c, b = data
                g["bubble"].append((a, c, b, st))
                extent.append(
                    ((a[0] + 2 * c[0] + b[0]) / 4, (a[1] + 2 * c[1] + b[1]) / 4)
                )
            elif kind == "ring":
                g["arc"].append(data + (st,))
                extent += poly
            else:
                c, r = data
                g["loop"].append((c, r, st))
                extent += [(c[0] - r, c[1] - r), (c[0] + r, c[1] + r)]
        for a, b in self._legs(pos):
            g["straight"].append((a, b))
            g["arrows"].append((((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), (1.0, 0.0)))
            extent += [a, b]
        for v, st in self.markers.items():
            p = P(v)
            g["marker"].append((p, st))
            extent += [(p[0] - 0.2, p[1] - 0.2), (p[0] + 0.2, p[1] + 0.2)]
        xs = [p[0] for p in extent]
        ys = [p[1] for p in extent]
        g["bbox"] = [
            min(xs) - self.pad,
            min(ys) - self.pad,
            max(xs) + self.pad,
            max(ys) + self.pad,
        ]
        return g


def diagram(term, **kw):
    """the Feynman diagram of a term (string or list of tuples); keywords: styles, ext, pad, straight,
    curved, vertical"""
    return term if isinstance(term, Diagram) else Diagram(term, **kw).layout()


def grid(terms, ncols=3, scale=40.0, gap=6.0, **kw):
    """several diagrams side by side (inline in Jupyter; .tikz(), .save('x.svg'/'x.pdf'/'x.tikz'/'x.tex'))"""
    return gd.Grid([diagram(t, **kw) for t in terms], ncols=ncols, scale=scale, gap=gap)


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _chain(edges, vin, vout):
    """the fermion line: a path of fermion lines (G, Gex, Pa) from the incoming to the outgoing vertex ([] if none)"""
    if vin is None or vout is None:
        return []
    adj = {}
    for name, u, v, _ in edges:
        if name in FERMIONS and u != v:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

    def walk(path):
        if path[-1] == vout:
            return path
        for w in adj.get(path[-1], []):
            if w not in path:
                found = walk(path + [w])
                if found:
                    return found
        return []

    return walk([vin])


def _loops(edges):
    """the closed loops of fermion lines (G, Gex, Pa), each as its vertices in the direction of propagation (G(a,b)
    runs from b to a); a loop of two lines (a bubble) is left out, so is a line from a vertex to
    itself"""
    nxt, order = {}, {}
    for name, u, v, _ in edges:
        if name in FERMIONS and u != v:
            nxt.setdefault(v, []).append(u)
            order.setdefault(v, len(order))
            order.setdefault(u, len(order))
    loops = []

    def walk(path):
        for w in nxt.get(path[-1], []):
            if w == path[0]:
                if len(path) >= 3 and path not in loops:
                    loops.append(list(path))
            elif w not in path and order[w] > order[path[0]]:
                walk(path + [w])

    for start in nxt:
        walk([start])
    return loops


def _heads(poly, dirn, n):
    """[(point, direction)] of n arrowheads at the middle of a line (its points), pointing from the
    first label to the second (dirn)"""
    if gd._dist(poly[0], dirn[0]) > gd._dist(poly[-1], dirn[0]):
        poly = poly[::-1]
    L = [0.0]
    for p, q in zip(poly, poly[1:]):
        L.append(L[-1] + gd._dist(p, q))
    out = []
    for s in [L[-1] / 2] if n == 1 else [L[-1] / 2 - 0.1, L[-1] / 2 + 0.1]:
        k = max(1, min(len(L) - 1, next((i for i, l in enumerate(L) if l >= s), len(L) - 1)))
        p, q = poly[k - 1], poly[k]
        f = (s - L[k - 1]) / (L[k] - L[k - 1]) if L[k] > L[k - 1] else 0.0
        out.append(((p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1])), gd._unit(p, q)))
    return out


def _arc(a, b, mid, n, h):
    """control point and a 6-segment polyline of the arc a -> b bulging by h along the normal n"""
    c = (mid[0] + h * n[0], mid[1] + h * n[1])
    pts = []
    for i in range(7):
        t = i / 6
        u = 1 - t
        pts.append(
            (
                u * u * a[0] + 2 * u * t * c[0] + t * t * b[0],
                u * u * a[1] + 2 * u * t * c[1] + t * t * b[1],
            )
        )
    return c, pts


def _seg_dist(q, a, b):
    """distance from q to the segment a -- b"""
    ux, uy = b[0] - a[0], b[1] - a[1]
    L2 = ux * ux + uy * uy
    t = (
        0.0
        if L2 == 0
        else max(0.0, min(1.0, ((q[0] - a[0]) * ux + (q[1] - a[1]) * uy) / L2))
    )
    return math.hypot(q[0] - a[0] - t * ux, q[1] - a[1] - t * uy)


def _hits(poly, pts, ends, eps=0.3):
    """the number of points (other than the ends) that the polyline passes through or close to"""
    return sum(
        1
        for q in pts
        if q not in ends
        and any(_seg_dist(q, poly[i], poly[i + 1]) < eps for i in range(len(poly) - 1))
    )


def _circumcircle(p, q, s):
    """centre and radius of the circle through three points (None if they are collinear)"""
    d = 2 * (p[0] * (q[1] - s[1]) + q[0] * (s[1] - p[1]) + s[0] * (p[1] - q[1]))
    if abs(d) < 1e-9:
        return None
    p2, q2, s2 = p[0] ** 2 + p[1] ** 2, q[0] ** 2 + q[1] ** 2, s[0] ** 2 + s[1] ** 2
    cx = (p2 * (q[1] - s[1]) + q2 * (s[1] - p[1]) + s2 * (p[1] - q[1])) / d
    cy = (p2 * (s[0] - q[0]) + q2 * (p[0] - s[0]) + s2 * (q[0] - p[0])) / d
    return (cx, cy), math.hypot(p[0] - cx, p[1] - cy)


def _arc_avoiding(c, r, a, b, avoid):
    """the arc of the circle (c, r) from a to b that does not pass the direction of the point
    avoid: (c, r, t0, t1), counterclockwise from angle t0 to t1"""
    ta, tb, tv = (math.atan2(p[1] - c[1], p[0] - c[0]) for p in (a, b, avoid))
    sweep = (tb - ta) % (2 * math.pi)
    if (tv - ta) % (2 * math.pi) < sweep:
        return c, r, tb, tb + 2 * math.pi - sweep
    return c, r, ta, ta + sweep


def _ring_pts(c, r, t0, t1, n):
    """an n-segment polyline along a circular arc"""
    return [
        (
            c[0] + r * math.cos(t0 + (t1 - t0) * i / n),
            c[1] + r * math.sin(t0 + (t1 - t0) * i / n),
        )
        for i in range(n + 1)
    ]


def _seg_gap(a, b, c, d):
    """the distance between two segments that do not cross"""
    return min(
        _seg_dist(a, c, d), _seg_dist(b, c, d), _seg_dist(c, a, b), _seg_dist(d, a, b)
    )


def _side(n, v):
    """+1 if v has a component along n, -1 if against it; if neither, the side pointing up (or left)"""
    s = n[0] * v[0] + n[1] * v[1]
    if abs(s) > 1e-9:
        return 1 if s > 0 else -1
    return 1 if n[1] > 1e-9 or (abs(n[1]) <= 1e-9 and n[0] < 0) else -1


def _away(p, cen):
    """unit vector from the centroid cen to p (up when p is the centroid)"""
    v = (p[0] - cen[0], p[1] - cen[1])
    L = math.hypot(v[0], v[1])
    return (v[0] / L, v[1] / L) if L > 1e-9 else (0.0, 1.0)

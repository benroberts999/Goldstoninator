# Goldstoninator

Goldstone (time-ordered) diagrams, energy denominators and signs from the numerator of a many-body perturbation theory term.
One file, `goldstone_draw.py`, standard library only. `goldstone_draw.ipynb` is a worked tour.

## Setup

The module needs only Python 3. Jupyter is needed for the notebook:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab goldstone_draw.ipynb      # or open it in VS Code and pick the .venv kernel
```

To use the module elsewhere, copy `goldstone_draw.py` next to your script or notebook (or add this directory to `sys.path`).

## Example, in four steps

**1. Write the term and show the diagram.** A term is a product of integrals; `t_rm` is a one-body vertex, drawn here as a cross:

```python
import goldstone_draw as gd

example = gd.diagram("g_vbrs g_asnb t_rm g_mnva", styles={"t": "cross"})
example            # as the last line of a Jupyter cell: the picture, inline
```

![example](example.svg)

**2. Export the picture**, as a file or straight to TikZ:

```python
example.save("example.pdf")       # also .svg
example.save("example.tikz")      # a tikzpicture, to \input
example.save("example.tex")       # a standalone document for pdflatex
print(example.tikz())             # or just the TikZ source
```

Wavy lines use the `snake` decoration, so a document that `\input`s the TikZ needs `\usetikzlibrary{decorations.pathmorphing}`. One diagram unit is 1 cm (`unit=` scales it).

**3. The equation**, with sign, numerator and energy denominators, typeset or as source:

```python
example.equation()                # typeset, as the last line of a cell
print(example.tex())              # the LaTeX source
```

$$
+\frac{g_{vbrs}\,g_{asnb}\,t_{rm}\,g_{mnva}}{(\varepsilon_{va}-\varepsilon_{mn})(\varepsilon_{vb}-\varepsilon_{ms})(\varepsilon_{vb}-\varepsilon_{rs}+\omega_t)}
$$

The one-body vertex absorbs an energy ω<sub>t</sub>, which enters the denominators after it (see below).

**4. Several terms at once**, side by side, with the sum of their equations:

```python
g = gd.grid(["t_na g_wavn", "g_wnva t_an"], ncols=2)
g                                 # the pictures
g.equation()                      # the sum; g.tex(), g.tikz(), g.save(...) as above
```

## Reference

### Writing a term

Integrals are written as `g_vbms`, `g_{vbms}`, `g[v,b,m,s]`, or as a list of tuples `[('v','b','m','s'), ...]`.

* Two-body `g_pqrs` = ⟨pq|g|rs⟩: electron r → p and s → q, drawn as a vertical interaction line.
* One-body `h_pr` = ⟨p|h|r⟩: a single vertex (marker), e.g. an external field.
* Letters: core (holes) `a b c d e f`, excited (particles) `m n r s p q t u`, valence `v w`.
  Change them with `core=`, `exc=`, `val=`, or `types=dict(x='core')`.
* Antisymmetrised integrals are not accepted: expand g̃_pqrs = g_pqrs − g_pqsr first.
* Integrals written in the opposite convention (`g_rspq`) are detected: the reading with the fewest flipped integrals is used.

The diagram follows from the numerator alone. Every non-valence letter appears once as an out index (p, q) and once as an in index (r, s); particles are created before they are annihilated and holes are created before they are filled, which fixes the time order.
Time runs left to right, particle arrows point right and hole arrows left, and the incoming valence line enters at the top left.

### Energy denominators and sign

One denominator per gap between successive vertices: the initial energy ε_v, plus the energies ω absorbed at the vertices before the gap, plus the energies of the hole lines crossing the gap, minus those of the particle lines (external valence legs count as particles).
Each denominator is written with the valence energy first and positive, then ±ω. The sign is (−1)^(hole lines + closed loops), times the sign flips needed for that ordering.

A one-body vertex is taken as *absorption* of an energy named after it (`t_rm` absorbs ω_t). The final valence energy never appears: energy conservation ε_w = ε_v + Σω removes it. So the two time orderings of a field vertex give ε_a − ε_n ± ω_t:

```python
gd.grid(["t_na g_wavn", "g_wnva t_an"]).equation()
```

`omega=False` gives a static field (no ω), `omega=r'\omega'` a plain symbol, and `omega={'t': r'\omega', 'S': True}` a symbol per name (any vertex may carry one; `True` means ω with the name as subscript).

### Styles

`styles={'S': 'dotted', 't': 'cross'}` on `diagram` or `grid`, or `gd.STYLES['t'] = 'cross'` for the whole session.
Two-body lines: `wavy` (default for `g`), `dashed` (default for other names), `dotted`, `double`, `solid`.
One-body markers: `x` or `cross` (default), `dot`, `circle`, `square`.
`pad=` adds margin around the picture; the energy symbols are `gd.EPS` and `gd.OMEGA`.

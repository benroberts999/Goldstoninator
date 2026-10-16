# Angular reduction of Goldstone diagrams: plan

Goal: from a goldstoninator term (or a grid of terms, or a Feynman diagram) produce the angular-reduced
expression in terms of Q^k, W^k (= Q^k + P^k), reduced one-body matrix elements, 6j/9j symbols and
phases, with the conventions of ~/Methods/main.tex (Sec. "Coulomb integrals", eqs. Qk, Wk, Pk, Ckab,
WigEck).  Outputs: LaTeX, ampsci C++ for the reduced form, and ampsci C++ for the full m-sum (the
independent test).

## Decisions

- Diagrams with no external field: the m-independent value, **averaged** over the external m,
  (1/[j_v]) sum_m <w m|X|v m> (times delta_{kappa_v kappa_w}); the output states this explicitly.
- One external field of rank k: the reduced matrix element <w||X^k||v>, Wigner-Eckart as in WigEck
  ((-1)^{j_w-m_w} 3j), which is also ampsci's `rme3js`.
- Several external fields: **uncoupled** by default, i.e. the external m-dependence is left as explicit
  3j's (see "External legs").  Coupled basis {T^{k1} x S^{k2}}^K optional.
- Two valence lines: S^k_{xyvw} with sum_k A^k_{xyvw} S^k_{xyvw} (as CI Sigma^2).
- One-body reduced matrix elements in **literal** order <p||t||r>, never swapped.
- Direct/exchange pairs become W; when either vertex could carry it, prefer **QW** over WQ (W on the
  later integral).
- No Python `.evaluate()`.  Physics-level testing is done in ampsci via `.ampsci_fullmsum()`.
- goldstoninator gets g~ input (below; done).

## Facts the design rests on

1. Every scalar two-body integral has the same m-structure: g_abcd = sum_k A^k_abcd Q^k_abcd and
   g~_abcd = g_abcd - g_abdc = sum_k A^k_abcd W^k_abcd, with the same A^k.  So W comes from pairing a
   diagram with its exchange partner (the two in-indices of one vertex swapped): same denominators
   (both lines end at the same vertex) and relative sign exactly -1 (a transposition changes the cycle
   count of the line permutation by one).  The angular algebra never touches the 6j inside P.
2. With these conventions the diagram already is a standard angular graph:
   A^k_abcd = (-1)^{1-j_a-j_b} (-1)^{j_a-m_a} (-1)^{j_b-m_b} sum_q (-1)^{k-q} (j_a k j_c; -m_a -q m_c)(j_b k j_d; -m_b q m_d)
   (checked by hand against the Laplace expansion + eq. Ckab; agrees with the A^k of main.tex), and
   t_pr = (-1)^{j_p-m_p} (j_p k j_r; -m_p q m_r) <p||t||r>.  Each orbital is once a bra (-m, with
   (-1)^{j-m}) and once a ket (+m): every fermion line is a metric contraction
   sum_m (-1)^{j-m} (..-m..)(..+m..), arrow along the line; every Coulomb line the same with j = k.
3. The angular factor does not depend on time ordering (hole/particle changes only ranges, signs,
   denominators): reduce once per Feynman diagram / topology.  Flipped integrals are harmless:
   A^k_pqrs Q^k_pqrs = A^k_rspq Q^k_rspq for real orbitals.
4. Phases: substitute j = J + 1/2 (and m likewise) for orbitals; k, q integer.  A real phase (-1)^x
   then has x integer-linear in J, k, m; coefficients mod 2 give a unique canonical form.  (-1)^{2j_a} = -1,
   (-1)^{j+m} = -(-1)^{j-m} etc. are automatic; a non-integer exponent raises.  Every 3j in these
   graphs has 0 or 2 half-integer entries, so triangle parities are automatic.

## goldstoninator: g~ input

`\tilde g_{abcd}`, `\widetilde{g}_{abcd}` and `g~_{abcd}` expand to g_abcd - g_abdc: a term with n
tildes is 2^n diagrams, `gd.diagram` returns their Grid (its .tex()/.equation() the sum), each diagram
with its own Goldstone sign; assert that the two halves of each pair have opposite signs.  The angular
module maps g~ straight to W (no pairing search needed for that vertex).

## Module `angularinator.py` (stdlib only)

1. **wigner**: numeric 3j/6j/9j (Racah formula) and C^k_ab, for the internal tests only.
2. **phase**: linear-form exponents, canonical mod-2 form, pretty printing (e.g. (-1)^{j_a-j_b+k}).
3. **network**: nodes/lines from a goldstoninator Diagram (`verts`, `lines`, `vins`, `vouts`).
   One-body names get a rank: `ranks={'t': 'k', 'T': 'k_1', 'S': 'k_2', 'V': 0}`; four-index names
   (g, X, ...) an A^k with a fresh multipole.
4. **external legs**: a basis for the open legs; the coefficient is the closed graph
   (diagram x conjugate basis) / norm, so everything reduces to closed graphs.
   - `average` (no field): close v with w, divide by [j_v].
   - `reduced` (one field): attach (-1)^{j_w-m_w} (j_w k j_v; -m_w q m_v).
   - two valence lines: project on A^k_{xyvw} (normalisation derived and tested).
   - `uncoupled` (several fields, default): an invariant with 4+ open legs is not a single product of
     3j's, so one intermediate per extra field is unavoidable.  Proposed: chain along the valence line in
     written order, sum_x (-1)^{j_w-m_w}(j_w k_1 j_x; -m_w q_1 m_x)(-1)^{j_x-m_x}(j_x k_2 j_v; -m_x q_2 m_v) X_x
     (x half-integer), like <w||T||x><x||S||v>.  (Agreed.)
   - `coupled` (optional): {T^{k1} x S^{k2}}^K_Q, sum over K.
5. **yutsis**: symbolic reduction of closed graphs.  Canonicalise nodes with the two elementary moves
   (reverse an arrow: (-1)^{2j}; reverse a node: (-1)^{j1+j2+j3}); rules on canonical templates only:
   zero line, self-loop (tadpole: delta_{k0}), disconnected -> product, 2-line cut -> delta/[j],
   3-line cut -> factorisation, triangle -> 6j, square -> sum_x [x] 6j 6j, longer cycles -> interchange
   (one 6j and sum_x) to shorten; shortest cycle first; recognise 9j at the end.  Forms are not unique
   (Biedenharn-Elliott): compare numerically, never as strings.
6. **exchange**: pair direct/exchange terms -> W (QW preferred), canonical relabelling of dummy letters
   (brute force within each class), merge identical terms.
7. **output**: Q (8-fold) and W/P (4-fold) symmetries and 6j tetrahedral symmetry to tidy indices;
   `.tex()` with the Methods macros (\sixj, \threej, \redmatel, [k]); `.ampsci()`; `.ampsci_fullmsum()`.

Internal tests (Python, development only, not user-facing): every yutsis rule and every reduced graph is
checked numerically against the direct m-contraction of its 3j network for random j's (half-integer
orbitals, integer multipoles).  This is how the engine is developed without a compile cycle; the
physics-level test is ampsci.

## ampsci output

Both functions get the same signature and return the same quantity, so they compare one to one:

    double <name>(const DiracSpinor &v, const DiracSpinor &w,
                  const std::vector<DiracSpinor> &core, const std::vector<DiracSpinor> &excited,
                  const Coulomb::QkTable &qk, const DiracOperator::TensorOperator &t, double omega_t,
                  [int two_mw, int two_mv, int two_q1, ... for the uncoupled multi-field case])

- `.ampsci()`: the reduced form.  Loops over core/excited per dummy letter; multipoles from
  `Coulomb::k_minmax_Q` / `k_minmax_W`; `qk.Q(k,a,b,c,d)` and `qk.W(k,a,b,c,d)`; `Angular::sixj_2`
  (or `Angular::SixJTable`), `Angular::neg1pow_2` with 2j integers; `t.reducedME(Fa, Fb)`;
  energies `Fa.en()`; denominators and sign from goldstoninator; selection-rule early exits.
- `.ampsci_fullmsum()`: the same quantity by brute force over every m of every letter, using only
  `Coulomb::g_abcd(a,b,c,d,tma,tmb,tmc,tmd)` (built from `Ck_kk_mmq` and `Rk_abcd`, independent of
  A^k/Q/W), with g~ expanded back into g_abcd - g_abdc, and `t.fullME(Fa, Fb, two_ma, two_mb, two_q)`;
  then the same external treatment (average / projection with `rme3js` / A^k projection / fixed m's).
  Slow (g_abcd recomputes radial integrals): tiny basis only.
- Optional: an ampsci Module (after `Modules/exampleModule.cpp`) that runs both on a small basis for a
  list of valence states and prints the relative differences.

Not covered by this comparison: the signs and denominators, which both versions take from
goldstoninator (check those against known results, e.g. ampsci's Sigma2 and StructureRad).

## Milestones

| | Contents | Checks | Status |
|---|---|---|---|
| M0 | g~ input in goldstoninator | expansion, relative signs, grid sums | done |
| M1 | external legs, `.ampsci_fullmsum()` (angularinator.py; methods on gd.Diagram, gd.Grid, fd.Diagram) | `ampsci_tests/run.sh`, below | done |
| M2 | wigner, phase, network builder, yutsis engine | internal numeric tests (rules, random graphs, networks) | done |
| M3 | exchange -> W, tidying, `.tex()`, `.ampsci()` | Sigma^(2) (4 diagrams -> eq. dE-SO-red), first-order RPA, CI Sigma^2_{vwxy}; `.ampsci()` vs `.ampsci_fullmsum()` in `ampsci_tests` | done (structure radiation still to do) |
| M4 | Feynman level (one angular factor per Feynman diagram), several fields (v T S w Q), coupled basis | as M3; bonus: against ampsci Sigma2 / StructureRad | |

Where a main.tex formula disagrees with the ampsci full m-sum, report it rather than adjust the code.

### M1 results (`ampsci_tests/run.sh`: H-like basis, core n <= 2; all agree to ~1e-15)

- Sigma^(2) energy, `g_vamn g~_nmav + g_vnab g~_banv` (average) = `MBPT::Sigma_vw` (ampsci's QW
  formula): confirms eq. dE-SO-red, 1/([k][j_v]) sum Q^k W^k, against the brute-force m-sum.
- Lone `g_xyvw` in Ak mode = Q^k_xyvw for every k: the A^k projection (phase, [k] normalisation).
- Lone `t_wv`: reduced = `reducedME`, m-resolved = `fullME`.
- First-order RPA, `t_na g~_wavn + t_an g~_wnva` (reduced) = the Methods formula
  sum (-1)^{k+w-n}/[k] (T_na W^k_wavn/(e_a-e_n+w) + (-1)^{a-n} T_an W^k_wnva/(e_a-e_n-w)).
- Feynman polarisation self-energy `v(1) G(1,2) w(2) Q(1,3) PI(3,4) Q(4,2)` = direct part of
  `MBPT::Sigma_vw`.
- RPA in m mode, projected by hand with rme3js = reduced mode.
- The v T S w Q example (m mode, 6 time orderings) compiles and runs.

`run.sh` compiles a private copy of the ampsci library into `ampsci_tests/build` (incrementally,
with the commands of ampsci's Makefile, `-Werror` and all its warnings), then `gen.py` -> `gen.hpp`
and `main.cpp`.  M3 adds the `.ampsci()` versions next to the full-m ones.

Implementation notes: a one-body integral that goldstoninator would have to flip (t_{in,out}) is an
error; an omega that appears in no denominator is not an argument; unused arguments are commented out
(the signature stays the same as the reduced form's).

### M2/M3 results

- Template phases fitted (j-linear, coefficients mod 4, enumerated): triangle A=(a-,l3-,l2+),
  B=(b-,l1-,l3+), C=(c-,l2-,l1+) -> (-1)^{a+b+c} {a b c; l1 l2 l3} (a-,b-,c-); interchange
  A=(a1-,a2-,e-), B=(b1-,b2-,e+) -> sum_x [x] (-1)^{a2+b1-e-x} {a1 a2 e; b2 b1 x} (a1-,b1-,x-)(a2-,b2-,x+).
  Bubble, theta, zero line, tadpole derived by hand.  Theta and bubble record triangle conditions.
- `ampsci_tests/test_reduce.py`: ~2000 random closed networks (loops, double lines, zero lines, K33,
  cube) reduce to the m-contraction value (~1e-15).  `test_network.py`: networks of real terms
  (all three closures) = direct m-sums from the A^k definition; reductions agree.
- Reduced forms reproduce the notes: dE-SO-red; RPA (the second term as (-1)^{k+w-a}/[k], equal
  to the notes' (-1)^{k+w-n}(-1)^{a-n}); CI ladder [k]{mu lambda k; x v a}{mu lambda k; y w b} with
  the notes' phase; CI g~g~ (-1)^k/[k].
- `ampsci_tests/run.sh`: `.ampsci()` = `.ampsci_fullmsum()` for 15 cases (Sigma2 three ways, RPA,
  lone g, CI ladders and exchange terms, third order, g~ third order, Hartree/exchange tadpoles,
  tadpole with field, scalar operator, two Feynman diagrams, third order with field): ~1e-15
  (1e-8 only on values that cancel to 1e-11).  Test energies spread to avoid H-like degeneracies;
  valence states excluded from the internal excited set.

## M4 (done; two reviews, all confirmed findings fixed)

- 'm' reduced form: chain basis along the valence line (closure (j_{i-1}+, K_i-, j_i-), (-1)^n,
  prod [j_i]); C++ loops over the chain and sums B(m) X_j; LaTeX shows the basis.
- 'coupled' (two fields): X_K, closure (k1-, k2-, K+)(w+, K-, v-), -sqrt[K](-1)^{k1-k2+K}; also in
  the full-m form.
- 9j recognition (sum_x [x] 6j6j6j with x-independent phase); notes-style 6j orientation.
- Structure radiation: the 36 SR diagrams (six Feynman diagrams) reduce to 18 QW terms; reduced =
  full-m (1e-14) and = ampsci StructureRad::SR (1e-13, omega = e_w - e_v).
- Demo notebook angular_reduction.ipynb (tracked, the user asked for it).

### First review (28 agents): 22 confirmed findings, 10 distinct issues, all fixed with regressions
phase accumulation for repeated symbols (high); deltas k=0 / rank=0 dropped (high); pairing across
time orderings (high); C++ integer division in [j] factors (high); \widetilde{g_{abcd}} dropped
(high); name collisions (letters k/J, user ranks, numeric/TeX ranks); omega' C++ names; repeated
valence letters; Grid in lists; g~ with equal last labels.

### Second review (21 agents): 16 confirmed, 7 distinct issues, all fixed with regressions
one-field m mode redeclared the rank (-Wshadow); user rank names equal to internal symbols merged
(internal symbols now contain '#', two-step renaming, chain j's get display names); rendered-name
collisions (\mu vs mu, k_T vs kT, equal rank names) now checked in raw/TeX/C++ forms; ranks={'t':'k'}
allowed again; chain_tex names; exchange partners with other letter names now pair (placeholder
relabelling); W orientation in the pairing key (order independence).  The SR diagrams now merge to
12 terms (6 QW, 6 WW), still = StructureRad::SR.  ampsci_tests: 34 comparisons pass.

## Next

- Condensed output (one angular factor per Feynman diagram / group of terms): ask the user about the
  format (generic line labels with per-ordering letter maps?).
- n >= 3 fields coupled basis (coupling tree), if wanted.

## Open

- Tilde on other two-body names (X~): later, if wanted.

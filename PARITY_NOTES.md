# MEQ-JAX / Octave parity — findings and current state

Work towards parity between the JAX FGE implementation and the reference
Octave MEQ implementation, on branch `octave-parity`.

**Summary.** Two numerical divergences from the Octave reference were found and
fixed, both verified against a live MEQ. The test suite is green for the first
time (817 passed / 10 skipped / 1 failed → 819 / 7 / 0). Three previously
disabled plasma shapes were shown to pass unmodified. The repository could not
be built from its own instructions; it now can. The repository had no
benchmark; it now has one, on MEQ's own perf axes.

---

## 1. Environment

Reproducing any of the numbers below requires the Octave reference:

| Component | Version |
|---|---|
| MEQ | `crpptbx-release-v6.9.0` |
| Octave | 8.4.0 |
| JAX | 0.10.2, CPU backend, float64 |

```sh
git clone --branch crpptbx-release-v6.9.0 https://gitlab.epfl.ch/spc/public/meq/meq.git
# build per the MEQ README, then:
export MAT_ROOT=<parent of the meq checkout>
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   # required; MEQ-JAX hangs otherwise
python -m pytest meq_jax/tests -o addopts="-rf"
```

`addopts = "-ra -q"` in `pyproject.toml` suppresses pytest's summary line, hence
the override.

---

## 2. Finding: contour branch selection in `rtci`

**Commit `c384f0c`. Severity: wrong results, silently.**

`rtcimex` is Newton on the flux along a line of sight, plus a correction: when
the local gradient points at the wrong contour branch, the step is overridden
to ±`dap` to escape. `libmeq/rtci.c:66` gates that correction on

```c
else if (dF*(Fo[io*(no-1-i)] - *Fp) > FLTC(0.0))   /* strictly greater than zero */
```

The port gated it on a tolerance instead:

```python
grad_check_mask = (df * (Fo - fq)) > 1e-13         # meq_jax/_src/rtci.py:128
```

**Why it matters.** Where a contour *touches* an origin point, `Fo - fq`
collapses to one ULP while its sign still carries the branch information. For
the outboard-midplane wall gap of an elongated limiter plasma:

```
df         = -1.239098e+00
Fo - fq    = -5.551115e-17     one ULP of 0.371: the plasma touches the wall
df*(Fo-fq) =  6.878374e-17     positive, but far below 1e-13

C   (> 0)      -> True   -> escape branch  -> walks to the near root
JAX (> 1e-13)  -> False  -> plain Newton   -> walks to the far root
```

Scanning the flux along that line of sight confirms two crossings, at
`a ≈ 0.0063` (the plasma boundary, the wall gap) and `a ≈ 0.7062` (where the
ray exits the plasma on the far side). MEQ returned ~0; MEQ-JAX returned 0.704.

Only contours touching the wall are affected, which is why exactly one wall gap
of ten diverged, and only for the elongated limiter case.

**Verification.** `rtciwall_test` seeds the solver with an unseeded `rand()`
initial guess, so the failure was intermittent. Driving the real test over 15
draws, comparing `aW[0]` against Octave:

| | mismatches |
|---|---|
| before | 11 / 15 |
| after | **0 / 15** (14 bit-identical, 1 within one ULP) |

After the fix MEQ-JAX reproduces MEQ's behaviour *including where MEQ is itself
unstable* — both saturate at the `L.G.aW` clamp for large initial guesses.

The epsilon dated from the initial import and had no recorded rationale.

## 3. Finding: no-Jacobian path of the CDE residual in `fgeF`

**Commit `8c43665`. Severity: crash in an untested run mode.**

`fgeF.m` computes the CDE constraint residual two ways — with Jacobians it
unpacks all fifteen outputs of `meqcde` and rescales by `resscalp`; without
them it calls `meqcde` for a single output, which in Matlab is the residual
alone, and does not rescale:

```matlab
else
  residualcde = meqcde(L,LX,LYp,Fx,ag,Iy,F0,F1,Ie,Opy);   % first output only
end
```

The port rendered that as `jnp.array(meqcde.meqcde(...))`, but Python returns
the whole fifteen-tuple, so the array constructor tried to stack the residual
with fourteen empty Jacobian blocks of differing widths. Every evaluation with
`dojacx=false` failed:

```
TypeError: Cannot concatenate arrays with shapes that differ in dimensions
other than the one being concatenated: ... (1, 0, 1), (1, 0, 1122), (1, 0, 3),
(1, 0, 992), ...
```

— exactly the fifteen return values. Fixed by taking `[0]`, and deliberately
not rescaling, matching `fgeF.m` on that path.

**Verification.** Residual against MEQ, shot 1: `max|diff| = 4.34e-15` with
`dojacx` both true and false.

**How it surfaced** is the point worth noting: not from the tests, which pin
`dojacx=true`, but from building a benchmark. MEQ's own `perf/fgeF_perf.m`
treats residual-only evaluation as a first-class case, so the path had simply
never been executed here.

---

## 4. Test coverage

**Parity gaps are now reported rather than commented out** (`8f001eb`). Each
axis of the `fgeF` matrix lists every value MEQ supports, mapped either to
`None` (at parity, the case runs) or a reason string (the case is generated but
skipped). `pytest -rs` prints the remaining deficit instead of it being visible
only by reading source. Cases are the cross product over at-parity values plus
one case per gap holding the other axes at baseline, so coverage grows back
automatically as gaps close and un-skipping a case isolates one feature.

**Three shapes were not gaps at all** (`44be8ac`). `diverted` (shot 2),
`diverted2` (3) and `squashed` (5) pass against Octave unmodified — they had
simply never been switched on. `fgeF_test.py` now runs 4 of 11 cases, up from
1.

**Suite:**

| | passed | skipped | failed |
|---|---|---|---|
| baseline | 817 | 10 | 1 |
| now | 819 | 7 | **0** |

---

## 5. Benchmark

The repository had no benchmark, so the README's claim that MEQ-JAX is "several
times faster than Octave on CPU" was not reproducible from it.
`meq_jax/examples/fgeF_bench.py` (`79ce4ba`) sweeps a single `fgeF` evaluation
over the axes MEQ's own `perf/fgeF_perf.m` uses — grid, shot, `icsint`,
`dojacx` — running both implementations from the same Octave-initialized
equilibrium. Octave is timed inside Octave so `oct2py` transport is excluded;
JAX excludes compile time and reports it separately.

Shared 4-vCPU container, single-threaded OpenBLAS. Ratio > 1 means MEQ-JAX is
faster:

| grid | shot | jacx | jax ms | meq ms | ratio |
|---|---|---|---|---|---|
| 32x16 | 1 | false | 4.868 | 19.554 | **4.02x** |
| 32x16 | 1 | true | 29.043 | 48.515 | 1.67x |
| 32x16 | 3 | false | 5.843 | 15.498 | 2.65x |
| 32x16 | 3 | true | 27.275 | 42.948 | 1.57x |
| 64x32 | 1 | false | 14.647 | 16.178 | 1.10x |
| 64x32 | 1 | true | 444.645 | 588.239 | 1.32x |
| 64x32 | 3 | false | 15.406 | 17.815 | 1.16x |
| 64x32 | 3 | true | 395.445 | 482.834 | 1.22x |

MEQ-JAX is faster in every configuration, so the README's claim holds — with
two qualifications it does not make:

* **The advantage collapses with grid size**, 4.02x at 32x16 down to 1.10x at
  64x32. "Several times faster" is true on the small grid only.
* **Jacobian evaluation scales poorly**: at 64x32, `dojacx=true` costs 445 ms
  against 14.6 ms without, a 30x jump for a 4x grid increase. XLA separately
  warned about constant-folding an `f64[3969,3969,1]` gather (3969 = 63²),
  suggesting a dense materialization worth investigating.

The two 64x32 Jacobian rows carry an interquartile range near 80% of the median
on this machine and should be treated as indicative only. The 32x16 rows are
stable (IQR under 20%).

### Batch scaling, GPU against CPU

`meq_jax/examples/profile_fge.py` (`046a0a5`) is the accelerator-oriented
counterpart: it measures compile time, per-step time and `vmap` batch scaling,
and replays a saved state with `--load_init`, so it runs on a machine with no
MEQ installed.

The README claims that a single simulation cannot saturate a GPU and that
`jax.vmap` over a batch amortizes the per-step kernel-launch overhead, citing
better than an order of magnitude at batch 256 on an H100. No measurement of
that existed. Below is one, on an NVIDIA Quadro P4000 (8 GB, Pascal GP104)
against the same machine's CPU (Intel Core i7-8700K, 6 cores), float64,
from an identical initial state:

| batch | CPU ms/sim | GPU ms/sim | GPU advantage |
|---|---|---|---|
| 1 | 47.43 | 51.11 | 0.93x — GPU slower |
| 2 | 49.56 | 47.73 | 1.04x |
| 4 | 45.31 | 37.89 | 1.20x |
| 8 | 41.09 | 32.70 | 1.26x |
| 16 | 39.58 | 32.95 | 1.20x |
| 32 | 38.04 | 27.77 | 1.37x |
| 64 | 37.73 | 24.35 | **1.55x** |
| 128 | 37.76 | out of memory | — |

The mechanism holds; the magnitude is strongly hardware-dependent, which the
README does not say:

* **At batch 1 the GPU loses**, 51.1 ms against 47.4 ms. GP104 runs float64 at
  1/32 of its float32 rate, and parity requires float64. Crossover is at about
  batch 2.
* **Batching helps the CPU too** — 47.4 to 37.7 ms, 1.25x — through
  vectorization rather than launch-overhead amortization. Only the excess over
  that is attributable to the accelerator.
* **The CPU saturates, the GPU does not.** CPU per-simulation time is flat from
  batch 32 (38.04, 37.73, 37.76), while the GPU is still improving at 64 when
  8 GB runs out, wanting a further 2.76 GiB for batch 128. The curve was still
  descending where the card stopped it, so a larger-memory device should go
  further — consistent with the README's claim, which was measured on an H100
  at batch 256.
* **Compile time is flat in batch size** on both, 11.7 to 15.6 s on GPU and 7.4
  to 15.4 s on CPU across a 128x range, so the `lax.scan` structure does what
  it is meant to.

Interquartile ranges are 0.2–6 ms against medians of 24–51 ms, so unlike the
`fgeF` benchmark above these figures are reliable.

What this does not settle is the order-of-magnitude claim itself: that needs a
device with unthrottled float64 and enough memory for batch 256. Running this
harness on one is a single command.

---

## 6. Reproducibility

**The repository could not be built from its own instructions** (`3d27028`).
The Dockerfile cloned MEQ's default branch, unpinned since the initial import.
MEQ v6.10.0 (upstream `32095bef`, 2026-05-18, "Implement Iecon parameter
similarly to agcon and replacing meqcircuit") removed `L.ind.ira`, which
`fgeF.py:355`, `types.py:147` and `fgeF_test.py` still read, so from that date
any fresh build failed with `error: structure has no member 'ira'` before a
single numerical comparison ran.

Nothing recorded the constraint: not this README, not the Dockerfile, and
MEQPy's install instructions say to take the *latest* MEQ. Now pinned through
`ARG MEQ_VERSION=crpptbx-release-v6.9.0` — an argument rather than a literal,
so it stays overridable — and stated in the README.

Also: `.gitignore` for build and cache artifacts (`a9cb7f2`) and for the
`octave-workspace` dumps `oct2py` leaves in the repository root (`a9cb7f2`).

---

## 7. Refactor

`fgeF_test.py` carried 400 lines of Octave→JAX struct conversion inline, so
anything else wanting to call `fgeF` against an Octave-initialized equilibrium
had to copy it. Extracted unchanged into `meq_jax/_src/fgeF_octave.py` as
`fgeF_inputs_from_octave` (`dbebc42`), and the Octave setup commands moved
alongside it as shared constants (`dbebc42`). `fgeF_test.py`: 826 → 351 lines.
This is what made the benchmark possible, and makes adding parity
configurations cheap rather than copy-paste.

---

## 8. What parity still requires

Restating the goal — parity "across all run modes and various possible input
settings" — against what MEQ actually exposes. **MEQ ships 17 FGE-related test
files; `fgeF_test.py` is based on one** (`liuNLmeas_jacobian_test.m`, a
Jacobian test).

### Run modes

| Mode | Values | Status |
|---|---|---|
| `algoNL` | `all-nl`, `all-nl-Fx`, `Newton-GS`, `Picard` | only `all-nl`; `Newton-GS` broken; `picard` → `NotImplementedError` |
| `optsF` flags | `dojacx`, `dojacu`, `dojacxdot`, `dojacF`, `dopost`, `dodisp`, `doplot` | 4 of 7 exercised; `dojacF` implemented but never set |
| evolutive | `fbt` vs `fge` | `fbt` only |
| `initguess` | `LIH`, `previous` | → `NotImplementedError` |
| vacuum | no-plasma case | untested (`fge_vacuum_tests.m`) |
| linearised | `fgelin`, multi-time | no JAX port |
| solver variants | Broyden, sparse Jacobian, Block-GMRES, SQP | untested / unported |

### Input settings

| Parameter | Values | Status |
|---|---|---|
| `cde` | `OhmTor`, `OhmTor_0D`, `OhmTor_rigid`, `OhmTor_rigid_0D`, off | `OhmTor_0D` only; multi-domain asserted out |
| `bfct` | `bfabmex`, `bf3imex`, `bf3pmex`, `bfprmex` | `bfabmex` only |
| `ipm` | 0/1/2/4 (MEQ has five test files) | pinned true |
| `idml`, `stabz`, `wreg`, `Ipmin`, `isaddl`, `xscal`, `ilackner` | each branched on in `fgeF.m` | pinned to one value; several not read by `fgeF.py` at all |
| `icsint` / `ilim` | 0/1, 1/3 | `icsint=0` only |
| domains | `nD=1`, doublets, inactive | `nD=1` only |
| constraints | `agcon`, `Iacon`, `Iecon` | partial |
| machine | `ana`, TCV | `ana` only |

Full parity across that matrix is a matter of weeks, not days. `fgeF.m`
branches on `idml`, `ipm`, `stabz` and `wreg`, none of which `fgeF.py` reads.

### Known, not fixed

* `rtciwall_test` seeds the solver with unseeded `rand()`. For large initial
  guesses both implementations saturate at the clamp; they now agree, so it
  passes, but it agrees on a non-answer. Seeding would make it reproducible.
* Each test class creates an Octave session in `setUpClass` and none tears it
  down; ~17 sessions at ~190 MB accumulate over a full run.
* The Newton-Raphson convergence assertion at the end of `test_fgeF` is
  commented out, so convergence is unverified for every configuration.
* `fgeF_octave.py` prints progress diagnostics, inherited verbatim from the
  test so the extraction stayed a pure move.
* **MEQ-side gap:** MEQ's `fgeF` accepts an `fgs` steady-state `LX`, which
  carries neither `dt` nor a CDE; MEQ-JAX's requires both. This is why the
  benchmark cannot mirror `fgeF_perf.m`'s setup exactly.
* **MEQ v6.10 drift.** Porting past `32095bef` means following `meqcircuit` →
  `Iecon`/`Iacon`: 22 files and +593/−372 upstream, landing on `fgeF.py` and
  `meqcdefun.py` here. `meqagcon.py`/`meqagconfun.py` are a template, since the
  upstream change is modelled on `agcon`. Judged too large to port *with
  verification* in the time available.

---

## 9. Commits

    git log --oneline mex-tests-and-jax-performance..octave-parity

The findings above cite the commit behind each one; this is the full list.

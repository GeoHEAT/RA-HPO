# Analytic reference for the infinite planar fracture

The exact solution for the infinite planar case: the field reflected by an unbounded planar
fluid-filled layer, evaluated from Sommerfeld integrals. It is independent of RA-HPO and of
gprMax.

| File | Role |
|---|---|
| `exxgprt.m` | Main script. Evaluates the wavenumber integral per frequency, applies the Ricker spectrum, transforms to time and writes the CSV |
| `hfhorxrall.m` | Integrand: the thin-layer TE/TM generalized reflection coefficients and the J0 (or J2) Bessel kernel |
| `vertical_plannar_2cm_analy_trace.csv` | The reference trace. Column 1 time (ns), column 2 electric field (V/m), 2048 samples at 0.2083 ns |
| `exxgprt_check.py` | Python transcription of the two MATLAB files. Recomputes the trace and compares it with the CSV, so the reference can be checked without MATLAB |
| `j2_term_check.py` | Evaluates the J2 term that `exxgprt.m` leaves out and reports its size relative to the J0 term |

## Configuration

| Quantity | Value |
|---|---|
| Host (both sides) | eps_r = 9, sigma = 1e-4 S/m |
| Layer | eps_r = 80, sigma = 0.5 S/m, thickness 0.02 m |
| Antenna to front face of the layer | 5.99 m for both dipoles (layer centered 6.0 m from the borehole axis) |
| Offset | 1.0 m, inline, dipoles parallel to the layer |
| Source | 100 MHz Ricker; 256 frequencies up to 600 MHz, 8x oversampled in time |

These match the full-domain FDTD model
`examples/gprMax_models/For_full-domain_verifications/3D8x8x8-shifted_nobh_frac-dist6-2h_h0pt010.in`
(source (0.5, 4.0, 4.0), receiver (0.5, 4.0, 3.0), 0.02 m layer centered at x = 6.5 m).

## Reproducing

MATLAB (the script uses only `integral`, `besselj` and `writematrix`):

```matlab
cd examples/analytic
exxgprt          % writes vertical_plannar_2cm_analy_trace.csv in the current folder
```

Without MATLAB, from the repository root:

```bash
python examples/analytic/exxgprt_check.py
# d=0.02 ncells=1: time axis equal True; max|diff|/peak = 1.51e-12
python examples/analytic/j2_term_check.py
```

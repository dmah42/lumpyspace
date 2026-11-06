# Run notes: WEC outside the spatial weighting

Source material for blog post 7. These notes cover one run: the WEC penalty
moved outside the spatial weighting (Task 4.14), on top of stage one (bounded
CMB fractions, per-constraint starting weights and floors, absolute
$\kappa\rho_0$). The run went 12,800 steps. The best checkpoint is step 9719
($S = 2.33 \times 10^7$).

## Figures

All `lumpyspace7_wec_*` images are from this run. The loss chart covers the
whole run. The H/a/shear plot and the maps are from the best checkpoint,
step 9719.

- `lumpyspace7_wec_loss.png`: loss, residual breakdown, Ω_m.
- `lumpyspace7_wec_hubble_shear.png`: H(t), a(t), σ²(t) at the observer's
  spatial position.
- `lumpyspace7_wec_maps_t{-4..1}.png`: shear, Ricci scalar, contraction and
  spatial weight W on three planes, at each integer t.

## What changed and the prediction

- The WEC penalty $\lambda c + \tfrac12 w c^2$ is averaged over the EFE points
  and added to the loss next to the CMB penalties. The spatial weighting
  $\tfrac12 e^{-2W} L + W$ applies to the EFE residual alone.
- The mechanism: inside the weighting, the optimum $W = \tfrac12 \ln L$ turns
  any constant multiple of $L$ into a constant. So WEC's weight and
  multiplier had no effect on the metric. In run 3, W sat at 3–6 and WEC
  stayed near 1 through every bump.
- The prediction was that W would fall below zero, WEC would respond to its
  weight, and the field equations would be weighted up.

## What happened (up to the best step)

- **W:** negative everywhere by step 30. At the best step it was −7.0 to
  −6.7 at t = −4 and −6.1 to −5.5 at t = 1. W tracks the EFE residual
  alone: $e^{2W}$ matches the logged EFE term.
- **WEC:** fell from 0.6 (step 100) to 0.026 (step 5240) at its starting
  weight of 10, with no bump. After the first bump (w 10 → 50 at step
  8000) it fell to $2 \times 10^{-4}$ by step 8750.
- **Logged EFE:** fell steadily, from 0.2 to $3 \times 10^{-6}$ by step
  9300, with no loss spikes. The step in the total loss at step 1000 is
  the first multiplier update. Most of the late fall in the logged EFE is
  the rescaling below.
- **H at the observer** (proper-time $H_{mean}$ at (1, 0, 0, 0); $H_0$ =
  0.2335 code units). The ratio is $H_{obs}$/$H_0$:

  | step | $H_{obs}$ | $H_{obs}/H_0$ |
  | --- | --- | --- |
  | 5129 | 0.0105 | 4.4% |
  | 7770 | 0.0207 | 8.9% |
  | 9390 | 0.0305 | 13% |
  | 9719 | 0.0325 | 14% |

  After the best step it briefly reached about 17% (Ω_m ≈ 9). For
  comparison, run 3's best was 39%.
- **H(t) at the best step:** 0.070 at t = −4, then falling steadily to
  0.0325 at t = 1.
- **κρ₀:** at the ceiling (0.0467) from about step 200 onward.
- **SN:** χ²/N between 0.47 and 0.51 from step 3500 to step 9700, with
  `l_sn_cover` at about $10^{-5}$.
- **CMB at t = −4:**
  - Contraction is essentially zero.
  - The shear fraction has held at $5 \times 10^{-3}$ to $10^{-2}$ since
    step 3500.
  - The homogeneity term has held at about 0.016 since step 7000.

## Penalty events

| step | event |
| --- | --- |
| 2000 | expansion, shear and homogeneity weights decayed |
| 5000 | expansion weight bumped to 500, later decayed back to 100 |
| 8000 | WEC weight 10 → 50, shear weight 100 → 500 |
| 9000 | shear → 2500, homogeneity → 500, WEC decayed to 25 |
| 10000 | shear → $10^4$ (cap), homogeneity → 2500 |
| 11000 | homogeneity → $10^4$ (cap), WEC → 125, expansion → 2500 |

## Breakdown after step 10,000

The shear and homogeneity constraints reached the cap without their
violations falling. At the cap, each check raises λ by about $w c \approx 300$
for each of them. Their penalties took over the loss:

- the total loss went from about 1.3 to 60–130;
- `l_sn` went from 0.49 to 2–4;
- `l_sn_cover` went from $2 \times 10^{-5}$ to about $10^{-2}$;
- there was no new best after step 9719.

Both jumps in the loss land on the check steps where those weights went up.
The cause of the shear plateau is not diagnosed (Task 4.14, open).

## The scale factor and the curvature at the observer

- **The shrink.** a at the observer is $(\det h)^{1/6}$ at (1, 0, 0, 0).
  It fell from 0.47 (step 5129) to 0.30 (7770), 0.268 (9390) and 0.262
  (9719). The part-5 best had 0.73 and the part-6 best had 0.41.
- **The curvature peak.** The Ricci scalar has a peak at the observer: about
  70, 165 and 240 at steps 5129, 7770 and 9719, and about 100 at the
  part-6 best. $R\,a^2$ stays at 15–17 throughout, so the peak has the same
  proper geometry each time, drawn at a smaller scale.
- **Why it doesn't violate the dust trace equation.** That equation needs
  $R = \kappa\rho$. The observer's density is $\kappa\rho_0 / a^3$: 2.43 at
  step 9390, which is 52× the ceiling, and about 14× at the part-6 best.
  R at the observer exceeds even that local density by about 100×.

### Rescaling check (step 9390, CPU)

Every proper spatial length was multiplied by c, using
$g \to SgS$ with $S = \mathrm{diag}(1, c, c, c)$. All terms were
evaluated at the checkpoint's multipliers and weights. The sample was 512
EFE points, 512 CMB points and 96 supernovae.

| c | logged EFE | unweighted EFE | EFE + WEC term | CMB penalties | SN | total | a_obs | H_obs | R_obs | κρ_obs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8 | 4.5e-7 | 75.6 | −6.01 | 2.31 | 0.81 | −2.89 | 0.214 | 0.0305 | 361 | 4.75 |
| 0.9 | 1.1e-6 | 46.1 | −5.97 | 2.44 | 1.22 | −2.31 | 0.241 | 0.0305 | 285 | 3.34 |
| 1.0 | 2.6e-6 | 30.0 | −5.84 | 2.58 | 0.71 | −2.55 | 0.268 | 0.0305 | 231 | 2.43 |
| 1.1 | 5.5e-6 | 20.5 | −5.57 | 2.74 | 1.10 | −1.73 | 0.295 | 0.0305 | 191 | 1.83 |
| 1.25 | 1.6e-5 | 12.5 | −4.63 | 3.00 | 1.93 | −0.30 | 0.335 | 0.0305 | 148 | 1.25 |

- **Invariant under the rescaling:** H, the shear fraction and the
  contraction fraction. R scales as $c^{-2}$ and the local density as
  $c^{-3}$.
- **The two EFE measures move in opposite directions.** The logged EFE,
  multiplied by $\det g_{ij}$, scales as $c^8$. The unweighted residual
  scales as $c^{-4}$. Shrinking improves the trained number and worsens
  the field-equation violation.
- **The homogeneity term scales roughly as c.**
- **The supernovae resist but don't win.** The SN term is lowest near the
  current scale. The total is still lower at c = 0.8.

Consequences:

- **The density bound doesn't hold at the observer.** It bounds the density
  only where $\det h = 1$.
- **The reported Ω_m is too low there.** It uses κρ₀, so it understates the
  observer's local value by $1/a^3$.
- **The SN distance assumes $a_0 = 1$.** `get_luminosity_distance` computes
  $(1+z) \times$ coordinate radius × 1 Gpc, which is the flat FLRW
  $(1+z)\,a_0 r$ at $a_0 = 1$.

Task 4.15 pins $a_0 = 1$ in the parameterisation.

## Interpretation (for the post)

- **The observer's pocket.** The observer sits in a small, slowly expanding,
  highly curved pocket. The volume the light crosses expands at the rate the
  data need: the SN fit was χ²/N ≈ 0.5, with H at the observer 14% of $H_0$
  and a sharp curvature peak at the origin.
  - This is an inference: H away from the origin has not been measured.
    Measure it before the post states it as fact.
  - The distances are coordinate distances (above). So the pocket's
    relation to the data rests on the FLRW distance formula.
- **The WEC result.** The field equations were being switched off wherever
  WEC was violated. With the penalty outside the weighting, the energy
  condition responds to its own pressure and the field equations get
  weighted up everywhere.

## FLRW assumptions found in the ray tracer

These have been in the ray tracer since it was written (2f9d33a, 2026-05-16).

- **Luminosity distance:** $(1+z)$ × coordinate radius. In a lumpy metric,
  the area distance comes from how the ray bundle focuses and shears (the
  Sachs optical equations). The coordinate radius contains none of that.
- **Redshift:** uses $u_\mu = (-\sqrt{-g_{00}}, 0, 0, 0)$, which is the
  comoving observer only when $g_{0i} = 0$.
- **Sky directions:** RA/Dec are used as coordinate directions at the
  observer. That is exact only if h there is isotropic.
- **BAO:** $D_M$ is the coordinate radius, and $D_H$ is 1/H at the spatial
  origin at the source's time. `l_bao` is 0 in this run.
- **The fix (Task 4.16):** two deviation vectors integrated alongside each
  ray give the area distance $d_A$, and $d_L = (1+z)^2 d_A$.

### Cost of the bundle ray tracer

Training with the Task 4.16 ray tracer runs at far fewer steps per hour.
The extra work is in tracing the supernovae:

- **Second derivatives at every solver step.** The deviation vectors evolve
  under the geodesic flow linearised about the ray. That needs the
  derivatives of the Christoffel symbols, so second derivatives of the
  network, at every evaluation of the right-hand side. Each ray carries 24
  components (the ray and two deviation vectors), where the ray alone is 8.
- **The backward pass repeats it.** Training differentiates through the
  integration, so the reverse pass runs through the second-derivative work
  and the larger state.
- **Smaller steps.** The adaptive step controller bounds the error on the
  deviation vectors as well as on the ray. They change fastest where the
  curvature is strong, such as the peak at the observer, so the solver
  takes smaller steps there.
- **The slowest ray sets the pace.** The supernovae are traced in parallel,
  and the batch waits for the ray that needs the most steps.
- **The observer normalisation (Task 4.15)** evaluates the network at the
  observer inside each metric call. Within the nested derivatives of a
  solver step this adds a few small forward passes, which is minor next to
  the items above.

### The best checkpoint under the model's own luminosity distance

This is the Task 4.16 diagnostic. The step-9719 checkpoint was scored on
GPU against all 1701 Pantheon+ supernovae. The metric is the network's
metric as trained, before scale normalisation.
The distance is $d_L = (1+z)^2 d_A$, and the redshift is the comoving one,
with the shift included.

- **Reached:** 1673 of 1701 supernovae. The run had almost all reached.
- **Fit:** χ²/N = 204, against 0.49 logged at that step.
- **Offset:** $\mu_{pred} - \mu$ has median −3.0 mag and spread 0.52 mag
  (Pantheon+ errors are about 0.15). The offset is close to constant in
  redshift:

  | z | n | median Δμ |
  | --- | --- | --- |
  | < 0.03 | 472 | −2.96 |
  | 0.03–0.1 | 269 | −3.09 |
  | 0.1–0.5 | 750 | −3.03 |
  | 0.5–1.0 | 177 | −2.43 |

  $5 \log_{10} a_0 = 5 \log_{10} 0.262 = -2.9$: almost all of the offset
  is the shrunken spatial scale, which the coordinate-radius distance could
  not see.

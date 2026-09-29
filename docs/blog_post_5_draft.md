---
title: "lumpyspace part 5: symmetry breaking, and the data that never arrived"
date: TBD
draft: true
tags:
  [
    "machine learning",
    "neural networks",
    "physics",
    "cosmology",
    "general relativity",
    "jax",
    "siren"
  ]
---

## Adding CMB constraints, at last

At the end of my [last post](/blog/lumpyspace4/), the neural network had found a
mathematically valid but physically unlikely contracting universe. The plan was
to immediately implement Cosmic Microwave Background (CMB) constraints to
provide a physical basis for the very early universe.

Before deploying those constraints, I made one small adjustment. To stitch them
together with the existing EFE-sampled spacetime, I needed to expand the time
sampling from $t=-3.9$ to $t=-3.99$.

I didn't expect much from that. I expected to spend this post writing about
anisotropic shear channels. Instead I spent it discovering that since day 1[^0]
the supernova data had never reached the model at all.

Here is the timeline.

### Pushing to $t = -3.99$

Expanding the coordinate sampling back to $t = -3.99$ pushed the network into
the extremely dense early universe. Physical density scales as
$\rho \propto a^{-3}$ so the Einstein Field Equation (EFE) residuals in this
region exploded.

To minimize that loss, the network took the path of least mathematical
resistance: it dropped the matter density parameter ($\Omega_m$) to the
baryonic floor ($\approx 0.05$). By turning the universe into a near-vacuum, it
zeroed out the density constraints entirely.

But without Dark Matter or Dark Energy, how was it still fitting the Supernova
acceleration data?

I assumed, and the data showed, that it cheated. It generated a metric with
**massive early shear**, using violent early-universe anisotropy as a physical
"engine" to drive late-time acceleration.

### Adding the CMB priors

That early shear is physically illegal. The Cosmic Microwave Background proves
the early universe was highly isotropic and homogeneous.

To shut the loophole I deployed Augmented Lagrangian boundary constraints in
the deep past ($t \in [-4.0, -3.99]$):

1. **Expansion Rate:** $H_i \ge 0$ (no contracting axes)
2. **Isotropy (Near-Zero Shear):** $\sigma^2 \le 10^{-5}$
3. **Homogeneity (Near-Zero Spatial Gradients):**
   $\sum (\partial g)^2 \le 10^{-5}$

Enforcing CMB isotropy took away the network's engine without undermining the
goal of the project. Forced to be isotropic in the early universe, the network
_had_ to rely on Dark Matter ($\Omega_m = 0.3$) to balance the EFE. But in
standard General Relativity, a universe filled only with matter **must
decelerate**.

Faced with forced deceleration, the optimizer fell right back into the local
minimum of a collapsing universe to brute-force the Supernova chi-squared fit.

### The symmetry breaking problem

If late-time lumpiness (backreaction) is a possible physical alternative to
Dark Energy, why wasn't the network finding it?

The answer lies in how neural networks initialize. A standard MLP initializes
with tiny, near-zero weights. Because our metric components are defined as
$g_{ii} = \exp(\text{NN}_i)$, the initial state maps almost exactly to flat,
empty Minkowski space, even with $\sim 10^{-4}$ noise.

In Minkowski space the expansion rate is exactly zero, the spatial gradients
are exactly zero, and it is perfectly homogeneous. Because the Einstein Field
Equations are highly non-linear, that perfectly homogeneous state acts as a
massive mathematical saddle point. The optimizer looks at the local gradients,
sees no spatial variance to exploit, and scales the entire metric uniformly
down the contracting branch.

The network cannot build a lumpy universe because it starts too perfectly
smooth to ever find the gradients required to build lumps.

The fix came from the [SIREN](https://arxiv.org/abs/2006.09661) architecture we
had already adopted. A SIREN multiplies the first layer's coordinates by a
high-frequency scalar $\omega_0$, so the network initializes as a superposition
of high-frequency ripples. That is the mathematical equivalent of seeding the
universe with a primordial power spectrum of density perturbations.

I chose a conservative $\omega_0 = 10.0$ rather than image-processing defaults of
30, to avoid an initial curvature explosion that would blow up the
EFE residuals. It seeds supercluster-scale gradients while keeping the initial
curvature inside the optimizer's stable bounds.

## Anchoring the fluid

While hunting the symmetry breaking problem I found a quieter bug in the
physics loss.

Falling back on FLRW and diagonal metric intuition, I had simplified the
pressureless matter Stress-Energy tensor. I set the energy density component as
$T_{00} = \rho(-g_{00})$, left every off-diagonal entry zero, and converted to
mixed form by contracting with the inverse metric.

For perfectly diagonal metrics where $g_{0i} = 0$, that is harmless. When the
PINN explores arbitrary 4D spacetimes with non-zero shift components
($g_{0i} \neq 0$), it becomes a bug. The inverse metric $g^{\mu 0}$ carries
off-diagonal elements, so multiplying by the naive $T_{00}$ injects phantom
momentum fluxes:

$$T^i_0 = -g^{i0} g_{00} \rho \neq 0$$

and simultaneously corrupts the rest-frame energy density:

$$T^0_0 = -g^{00} g_{00} \rho \neq -\rho$$

The entire physical premise of this project rests on matter being strictly
**comoving** ($u^i = 0$). Galaxies ride along with the expansion grid; they do
not move relativistically through coordinate space. That assumption is what
justifies the geometric density dilution law
($\rho \propto 1/\sqrt{\det(g_{ij})}$) and guarantees that the redshifts tracked
by the ray-tracer represent true metric expansion.

Leaving the bug in place handed the network a convenient cheat: by generating
non-zero coordinate shifts, the optimizer could fabricate fictitious matter
currents to bypass the EFE residual.

The fix is the first-principles definition. With $u^i = 0$, the exact
4-velocity is

$$u^\mu = \left(\frac{1}{\sqrt{-g_{00}}}, 0, 0, 0\right)$$

and lowering the index gives the exact mixed tensor

$$T^0_0 = -\rho, \qquad T^0_i = \rho \frac{g_{0i}}{-g_{00}}, \qquad T^i_0 = 0, \qquad T^i_j = 0$$

I implemented that directly in the EFE loss, which eliminated the phantom
momentum fluxes and pinned $T^0_0$ to $-\rho$.

## The result that didn't add up

With symmetry broken and the fluid anchored, I let it run. And the numbers were
wrong in a way I couldn't explain.

$\Omega_m$ sat pinned at the $0.05$ floor. $H_{\text{mean}}$ came out around
$-0.003$ in code units, where $H_0 = 73$[^1] should be $\approx 0.2435$. Two
orders of magnitude too small, and negative.

Then I looked properly at the supernova loss. `l_sn` is a **mean** of squared
standardized residuals, which is to say a reduced chi-squared. A good fit sits
at $\approx 1$. Mine sat at $\sim 86$, and its best value across the entire run
was $\sim 29$. Every supernova in Pantheon+ was sitting five to nine $\sigma$
from the model's prediction.

That is not a fit. And I had been reading those numbers for months as though
they meant something.

### Where the light never got to

The ray-tracer integrates null geodesics backwards from the observer until it
accumulates the target redshift for each supernova. So I instrumented it and
asked how far a ray actually gets.

```text
 pctile  z_target  max_z_found  short?    sol.t1
      0    0.0012      0.00002    True  -10.0000
     50    0.1636      0.00002    True  -10.0000
     99    1.2359      0.00002    True  -10.0000
    100    2.2613      0.00002    True  -10.0000
```

Every ray, across the entire affine path, accumulated a total redshift of
$2 \times 10^{-5}$. The nearest supernova in the catalogue sits at $z=0.0012$,
sixty times beyond reach. **One hundred percent of Pantheon+ was unreachable.**

So what had `get_luminosity_distance` been returning for four months?

When the integration cannot reach the target, `jnp.interp` clamps to the end of
the trajectory, and a penalty I had written as an "optimizer compass" gets
added on top to encourage expansion:

```python
penalty = jnp.where(max_z_found < z_target, 10 * (z_target - max_z_found), 0.0)
return (result + penalty) * L_UNIT
```

With `max_z_found` pinned near zero, that reduces to

$$d_L \approx 10^4 z - 105 \ \text{Mpc}$$

a straight line in the **target** redshift, offset by a near-constant term from
the geometry. Checking it against the real returns: at $z=0.0030$ the function
gave $-75.3$ Mpc against the formula's $-75$; at $z=0.0199$, $94.1$ against
$94$; at $z=0.0283$, $177.7$ against $178$.

The luminosity distance contained no distance-redshift physics whatsoever. It
was a linear function of the number I passed in.

### Why it survived for months

Three things conspired to keep it quiet.

The penalty was scaled to look plausible. At $z=1$ it produces
$\mu \approx 44.8$ against a real value of $44.1$ — wrong by an amount that
reads as a mediocre fit rather than a catastrophic failure.

Negative distances were silently absolved. The interpolation ran over a
non-monotonic redshift array, which `jnp.interp` does not support and does not
warn about, and it duly returned negative "distances". Those were then
rectified by an `jnp.abs()` in the distance modulus calculation.

And the penalty term looked like diligence. A penalty term whose comment says
it exists to drive expansion reads, at a glance, like me being careful.

## Minkowski is the problem

Once I understood what `l_sn` had been measuring, a more uncomfortable question
followed: what had been holding the geometry up at all?

So I checked what empty flat space scores on each term in the loss.

| term | at Minkowski | why |
| --- | --- | --- |
| $\mathcal{L}_{\text{phys}}$ | **0** | $G_{\mu\nu}=0$, and `kappa_rho_0 = omega_m * 3 * h_mean**2` zeroes the stress-energy at $H \to 0$ whatever $\Omega_m$ says |
| $l_{\text{wec}}$ | **0** | $R = 0$ |
| $l_{\text{expand}}$ | **0** | $H_i = 0$, so $\max(0, -H_i)^2 = 0$ |
| $l_{\text{shear}}$ | **0** | $\sigma^2 = 0 \le 10^{-5}$ |
| $l_{\text{spatial}}$ | **0** | no spatial gradients |

Empty static space is a **global minimum of every physics term in the
objective**. Every constraint I spent two posts building (Augmented
Lagrangian, the adaptive penalty scheduler, CMB priors) is perfectly
satisfied by a universe containing nothing and doing nothing.

The supernova data was the only term in the entire loss capable of pushing the
metric off Minkowski. And that was the broken term.

That single fact retroactively explains everything else. $\Omega_m$ dropping
to the floor: with $H \approx 0$ there is no matter regardless of $\Omega_m$,
so nothing pushed back. The collapsing universe: nothing was holding the
expansion up. The constraints that refused to improve under escalating
penalties: they were being dragged off their comfortable Minkowski values by
the lumpiness pressure, and fighting to return.

Three separate mysteries across three posts, all the same one.

## One ray for seventeen hundred supernovae

While repairing the channel I found a second problem, less a bug than a
simplification I had never revisited.

```python
initial_k = jnp.array([kt, kx, 0.0, 0.0])
```

Every supernova was traced along $+x$. All 1701 of them, wherever they actually
sit on the sky, differing only in how far along that one ray the integration
ran.

The model's geometry is genuinely four-dimensional and anisotropic, which is
the entire thesis of this project, and the data was only ever probing a single
line through it. The network was free to put whatever it liked everywhere else,
because nothing observed it.

Pantheon+ ships RA and Dec for every supernova and I had simply never loaded
the columns. Its sky distribution carries a dipole of magnitude $0.28$, so
those directions hold real angular structure.

Firing each ray along its own direction also required fixing the null
normalization. The old code solved $g_{00}k_t^2 + g_{11}k_x^2 = 0$, ignoring
the shift terms $g_{0i}$, so the initial ray stopped being null the moment the
metric developed off-diagonal time-space components. The general case is a
quadratic in the spatial magnitude $s$ along a direction $\hat{n}$:

$$\left(g_{ij}n^i n^j\right)s^2 - 2k_t\left(g_{0i}n^i\right)s + g_{00}k_t^2 = 0$$

## Separating the two questions

The repair is to stop letting one number answer two questions.

`get_luminosity_distance` now returns `(dL, z_reached)` and doesn't invent a
value for an unreachable target. The loss splits to match:

- `l_sn` is the chi-squared over reachable supernovae, with a **fixed**
  denominator of the full catalogue. Unreachable sources contribute zero to the
  numerator while still counting in the denominator, so a badly-fitting
  supernova cannot be shed by making it unreachable.
- `l_sn_cover` is the mean redshift shortfall, a hinge that reads exactly zero
  once every ray reaches its source.

`z_reached` is the redshift at the last finite state of the trajectory rather
than the maximum anywhere along it, which would reward a ray that redshifts
briefly and then blueshifts back.

## Results

<!-- PLACEHOLDER: regenerate every figure from the repaired run.
     Target ~10k steps unless patience triggers first. -->

The channel reconnected immediately. The mean redshift shortfall collapsed from
$0.25$ — effectively the mean redshift of the catalogue, since nothing was
reachable — through $9\times10^{-2}$, $1.7\times10^{-2}$, $8.4\times10^{-3}$,
to around $3\times10^{-3}$ inside a few hundred steps. The supernova
chi-squared fell alongside it, from the fabricated $46$ to $\approx 1.6$ per
supernova.

![Loss over time](PLACEHOLDER_lumpyspace5_loss.png)

<!-- PLACEHOLDER: total loss, residual breakdown including l_sn_cover,
     and Omega_m. -->

### The abandoned tail

`l_sn_cover` then stopped falling and began oscillating around
$3\times10^{-3}$. Because it is a mean over the full catalogue, that number
inverts to a horizon: find $h$ such that
$\sum_{z_i > h}(z_i - h) = 1701 \times 3\times10^{-3}$.

The model reaches $z \approx 1.22$. Nineteen supernovae — the entire $z > 1.23$
tail, $1.1\%$ of the catalogue — sit permanently beyond it, and every unit of
shortfall lives there.

Abandoning them is rational. The coverage penalty is linear in shortfall, so
the most distant supernova is always the cheapest thing to give up. Reaching
$z = 2.26$ means extending the horizon by a further $1.04$ in redshift, an
enormous change to the geometry, while leaving it out costs a fraction of the
loss. So the optimizer parks the tail and polishes the 1682 supernovae it can
already see, which is exactly what a falling `l_sn` beside a flat `l_sn_cover`
describes.

The horizon settles at $z \approx 1.22$. The 99th percentile of Pantheon+ is
$z = 1.236$. The model reaches precisely as far as the data is dense, and
stops.

![Directional expansion rates and shear](PLACEHOLDER_lumpyspace5_hubble_shear.png)

<!-- PLACEHOLDER: H(t) and shear. The question that matters is whether
     H_mean(t=1) approaches 0.2435 (H_0 = 73 in code units) or stays orders
     of magnitude below it. -->

![Curvature and shear](PLACEHOLDER_lumpyspace5_ricci_shear_2d.png)

<!-- PLACEHOLDER: 2D curvature and shear across the principal planes, plus
     the spatial weight maps. No pre-repair figures exist to compare
     against, so describe the structure on its own terms rather than as a
     before/after. -->

## What this cost, and what it bought

<!-- PLACEHOLDER: conclusion, pending the full run at ~10k steps.

     The shape depends on what H_mean does:

     - If H_mean(t=1) climbs toward 0.2435 with a sane expansion history,
       this is the first run where the model has genuinely fitted Pantheon+
       under General Relativity, and the lumpiness question can finally be
       asked honestly.
     - If H stays orders of magnitude low while chi-squared looks healthy,
       something other than expansion is sourcing the fit, and that needs
       chasing before any cosmological claim gets made.

     Either way: state plainly that the figures and numbers in parts 3 and 4
     were produced under the broken channel and should be read accordingly.
-->

## Next: matter that can push back

One thread from the Minkowski table is worth pulling.

`kappa_rho_0 = omega_m * 3.0 * h_mean_today**2`. The matter density is
_derived_ from the expansion rate, so $H \to 0$ forces $\rho \to 0$ whatever
$\Omega_m$ holds. Matter can never be a reason for the universe to expand,
because there is no matter unless it is already expanding.

Make the free parameter an absolute density $\rho_0$ and flat static space with
$\rho_0 > 0$ produces a non-zero EFE residual, where the only way down is to
curve. "There is matter here" becomes incompatible with "there is nothing
happening", which is the physically true statement.

$\Omega_m$ then becomes a quantity the fit _reports_ rather than a tuning knob
clipped to $[0.05, 0.3]$; considerably more tabula rasa than where this project
currently stands.

That is the next post.

[^0]: or $t=0$ i suppose.
[^1]: [_A Comprehensive Measurement of the Local Value of the Hubble Constant with 1 km $text{s}^-1$ $text{Mpc}^-1$ Uncertainty from the Hubble Space Telescope and the SH0ES Team_](https://arxiv.org/abs/2112.04510)
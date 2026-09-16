---
title: "lumpyspace part 5: breaking symmetries and discovering lumpyspace"
date: 2026-06-08
draft: false
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

## Adding CMB constraints, at last.

At the end of my [last post](/blog/lumpyspace4/), the neural network had found a
mathematically valid but physically unlikely contracting universe. Our plan was
to immediately implement Cosmic Microwave Background (CMB) constraints to
provide a physical basis for the very early universe.

But before we deployed those constraints, we made one small adjustment. To be
able to apply these constraints to the network and stitch it together with the
existing EFE-sampled space-time, I needed to expand the time sampling from
$t=-3.9$ to $t=-3.99$.

I didn't expect this to change much in the network, but needed to run a test to
make sure we could still navigate the steeper gradients in this time period.
However, that single change unlocked a revelation about how the network was
trying to "cheat" the physics, and ultimately led me to a fundamental
understanding of how neural networks are initialized.

Here is the exact timeline of how the model collapsed, and how we broke the
symmetry to fix it.

### Pushing to $t = -3.99$

When we expanded our coordinate sampling back to $t = -3.99$, we pushed the
network into the extremely dense early universe. Because physical density scales
as $\rho \propto a^{-3}$, the Einstein Field Equation (EFE) residuals in this
region exploded.

To minimize this massive loss, the network took the path of least mathematical
resistance: it dropped the matter density parameter ($\Omega_m$) to the baryonic
floor ($\approx 0.05$). By turning the universe into a near-vacuum, it
effectively zeroed out the massive density constraints.

But without Dark Matter or Dark Energy, how was it still fitting the Supernova
acceleration data?

It cheated. It generated a metric with **massive early shear**. By exploiting
the lack of constraints in the early universe, the network used violent,
early-universe anisotropy as a physical "engine" to drive late-time
acceleration.

### Adding the CMB Priors

I knew this massive early shear was physically illegal: the Cosmic Microwave
Background proves that the early universe was highly isotropic and homogeneous.

To shut down this loophole, I deployed Augmented Lagrangian boundary constraints
at the deep past ($t \in [-4.0, -3.99]$):

1. **Expansion Rate:** $H_i \ge 0$ (no contracting axes)
2. **Isotropy (Near-Zero Shear):** $\sigma^2 \le 10^{-5}$
3. **Homogeneity (Near-Zero Spatial Gradients):**
   $\sum (\partial g)^2 \le 10^{-5}$

By enforcing the CMB isotropy constraint, I took away the network's engine.
Forced to be perfectly isotropic in the early universe, the network _had_ to
rely on Dark Matter ($\Omega_m = 0.3$) to balance the EFE. But in standard
General Relativity, a universe filled only with matter **must decelerate**.

Faced with forced deceleration, the optimizer fell right back into the local
minimum of a collapsing universe to brute-force the Supernova chi-squared fit.

### The Symmetry Breaking Problem

If late-time lumpiness (backreaction) is a possible physical alternative to Dark
Energy, why wasn't the network finding it this time?

The answer may lie in how neural networks initialize. A standard Multi-Layer
Perceptron (MLP) initializes with tiny, near-zero weights. Because our metric
components are defined as $g_{ii} = \exp(\text{NN}_i)$, the initial state of the
network maps almost exactly to flat, empty Minkowski space, even with our slight
noise $\sim 1e-4$.

In Minkowski space, the expansion rate is exactly zero ($H=0$), and the spatial
gradients are exactly zero. It is perfectly homogeneous.

Because the Einstein Field Equations are highly non-linear, a perfectly
homogeneous state acts as a massive mathematical saddle point. The optimizer
looks at the local gradients, sees absolutely no spatial variance to exploit,
and simply scales the entire metric uniformly down the contracting branch. The
network suffers from a symmetry breaking problem: it cannot build a lumpy
universe because it starts too perfectly smooth to ever find the gradients
required to build lumps.

To find a solution, I looked again to the
[SIREN (Sinusoidal Representation Networks)](https://arxiv.org/abs/2006.09661)
architecture, which I had already adopted. A SIREN multiplies the first layer's
coordinates by a high-frequency scalar, $\omega_0$. By scaling the spatial
coordinates by $\omega_0$ in the first layer, the network initializes as a
superposition of high-frequency macroscopic ripples. This is the mathematical
equivalent of seeding the universe with a primordial power spectrum of density
perturbations, or quantum fluctuations.

I chose a conservative frequency of $\omega_0 = 10.0$ (rather than the
image-processing default of 30) to avoid an initial curvature explosion that
could blow up the EFE residuals.

This seeds the initial universe with the supercluster-scale gradients needed.
It breaks the homogeneous saddle point, giving the EFE the fluctuations required
to build late-time backreaction, while keeping the initial curvature well within
the stable bounds of the optimizer.

### Anchoring the Fluid

While hunting down the symmetry breaking issue, I also discovered a subtle bug
in the physics loss calculation that had been quietly lingering in the codebase.

Falling back on FLRW and diagonal metric implementations, I had made a
simplification for the pressureless matter Stress-Energy tensor. I set the
energy density component as $T_{00} = \rho (-g_{00})$, left every off-diagonal
entry in the $4 \times 4$ matrix as zero, and then converted it to mixed form by
contracting with the inverse metric: $T^\mu_\nu = g^{\mu\alpha} T_{\alpha\nu}$.

For perfectly diagonal metrics where $g_{0i} = 0$, that simplification is
harmless. But when the PINN architecture explores arbitrary 4D spacetimes with
non-zero shift components ($g_{0i} \neq 0$), that shortcut turns into a bug.

Because the inverse metric $g^{\mu 0}$ contains off-diagonal elements ($g^{i0}$),
multiplying $g^{i0}$ by our naive $T_{00}$ silently injected phantom momentum
fluxes into the equations:

$$T^i_0 = -g^{i0} g_{00} \rho \neq 0$$

At the same time, it corrupted the rest-frame energy density:

$$T^0_0 = -g^{00} g_{00} \rho \neq -\rho$$

This is a serious issue. The entire physical premise of this project rests on
matter being strictly **comoving** ($u^i = 0$). In our cosmological model,
matter rides passively along with the cosmic expansion grid: galaxies do not blast
relativistically through coordinate space. That comoving assumption is what
justifies our geometric density dilution law ($\rho \propto 1/\sqrt{\det(g_{ij})}$)
and guarantees that the cosmological redshifts tracked by our geodesic ray-tracer
represent true metric expansion rather than arbitrary bulk Doppler shifts.

By leaving that bug in place, I had accidentally handed the neural network a
convenient mathematical cheat: By generating non-zero coordinate shifts ($g_{0i}$),
the optimizer could fabricate fictitious relativistic matter currents and tamper
with the local energy density to bypass the Einstein Field Equation residuals.

To fix this, I went back to the first-principles definition of comoving dust.
With spatial 4-velocity $u^i = 0$, the exact 4-velocity in an arbitrary metric is:

$$u^\mu = \left(\frac{1}{\sqrt{-g_{00}}}, 0, 0, 0\right)$$

Lowering the index via $u_\nu = g_{\nu\mu} u^\mu$ gives $u_0 = -\sqrt{-g_{00}}$
and $u_i = g_{0i} / \sqrt{-g_{00}}$. The exact mixed Stress-Energy tensor
$T^\mu_\nu = \rho u^\mu u_\nu$ is therefore:

$$T^0_0 = -\rho, \qquad T^0_i = \rho \frac{g_{0i}}{-g_{00}}, \qquad T^i_0 = 0, \qquad T^i_j = 0$$

I scrapped the naive $T_{\mu\nu}$ matrix calculation and implemented this exact
mixed tensor directly in the EFE loss function. This strictly eliminated all
phantom momentum fluxes ($T^i_0 = 0$), pinned $T^0_0$ firmly to $-\rho$, and
closed the loophole completely.

### The Discovery of Lumpyspace

With the symmetry broken, the dust comoving, and the strict CMB constraints locking
the Big Bang boundary, I let the network run. And it found something absolutely wild.

At first glance, it looked like the network was still failing: the global mean
expansion rate ($H_{\text{mean}}$) was crossing into negative territory. The
universe as a whole was collapsing under the massive gravitational weight of
$\Omega_m = 0.3$. Because the universe was collapsing, standard cosmology
dictates that the Supernova light should be severely blueshifted, completely
failing the Pantheon+ distance constraints.

And yet, the Supernova data loss ($\mathcal{L}_{sn}$) was miraculously low. The
network was producing positive redshifts in a collapsing universe.

When we plotted the 3D geometry of the shear ($\sigma^2$) and the Ricci
curvature ($R$), the solution became perfectly clear. The network hadn't found a
code bug; it had found a valid, non-standard geometric solution.

It constructed **Lumpyspace**.

By shattering Bianchi Type I symmetry, the network built massive diagonal bands
of high shear separated by deep valleys of zero shear, along with distinct
"lumps" of positive and negative Ricci curvature. While the universe was
_globally_ collapsing, the network routed the specific geodesic light paths from
the Supernovae through these localized, high-shear anisotropic channels.

The local shear and spatial curvature gradients physically stretched the photons
just enough to perfectly mimic the redshift normally attributed to Dark Energy.
It proved the core thesis of this project: by relaxing standard homogeneity and
isotropy assumptions, localized cosmic structures can produce the exact optical
illusions required to fit the expansion data without ever needing a Cosmological
Constant ($\Lambda$).


---
title: "lumpyspace part 3: calling the physics cops"
date: 2026-05-31
draft: false
tags: ["machine learning", "neural networks", "physics", "cosmology", "general relativity"]
---

## Transitioning from Soft Penalties to Hard Constraints

In my [previous post](/blog/lumpyspace2/), I introduced a trainable matter
density parameter $\Omega_m$ and a soft penalty to enforce the Weak Energy
Condition (WEC), which requires the local curvature, the Ricci scalar $R$, to be
non-negative ($R \ge 0$) everywhere. The model responded with a fascinating
result: it rejected Dark Matter entirely, driving $\Omega_m$ straight to the
baryonic floor of $\approx 0.05$, and fit the Pantheon+ supernovae using a
spatial curvature dipole and anisotropic shear.

But I also noted a critical caveat: The WEC was implemented as a soft, 
quadratic penalty in the loss function so the optimizer struck a mathematical
compromise. It allowed a slight residual negative curvature ($R < 0$) and
non-zero early-universe shear to help fit the raw supernova distances.

To take this inhomogeneous model seriously, we cannot allow the network to
compromise on general relativity. The physics must be a hard boundary. So, it's
time to call the physics cops.

### The Vanishing Gradient of Soft Constraints

Our initial WEC penalty was defined using a standard squared minimum:

$$\mathcal{L}_{\text{WEC\_soft}} = \text{mean}(\min(R, 0.0)^2)$$

Mathematically, this seems reasonable. If $R \ge 0$, the loss is zero. If
$R < 0$, the penalty scales quadratically. 

However, this formulation has a fatal flaw at the boundary: the gradient of this
penalty is proportional to $2 \cdot \min(R, 0.0)$. As the curvature $R$
approaches the boundary from the negative side ($R \to 0^-$), the gradient
vanishes. The "restorative force" pulling the model back into the physical
regime becomes weaker and weaker the closer it gets to satisfying the law,
allowing the network to comfortably park itself in a state of mild violation.

To fix this, we made two key changes:

#### Linear Violation Penalty
We changed the penalty term from quadratic to linear:

$$\mathcal{L}_{\text{WEC}} = \text{mean}(\max(0.0, -R))$$

Because this function is linear for negative values, its gradient is a
constant non-zero step function. The network receives the exact same, strong
restorative gradient regardless of whether it is slightly violating the WEC
or massively violating it.

#### The Augmented Lagrangian Method
Instead of relying on a static, soft weight that the network can easily scale
away, we treat the energy condition as a true hard inequality constraint. The
total loss now incorporates a dynamic Lagrange multiplier ($\lambda_{\text{WEC}}$)
alongside a quadratic penalty term:

$$\mathcal{L}_{\text{Total}} = w_P \mathcal{L}_{\text{EFE}} + w_D\mathcal{L}_{\text{Data}} + \lambda_{\text{WEC}} \mathcal{L}_{\text{WEC}} + \frac{w_{\text{WEC}}}{2} \mathcal{L}_{\text{WEC}}^2$$

Inside our JIT-compiled training loop, the Lagrange multiplier dynamically
accumulates at every single optimization step based on the remaining WEC
violation:

$$\lambda_{\text{WEC}} \leftarrow \lambda_{\text{WEC}} + w_{\text{WEC}} \mathcal{L}_{\text{WEC}}$$

If the network attempts to violate the WEC, the multiplier
$\lambda_{\text{WEC}}$ climbs higher and higher, acting as a feedback loop that
increases the cost of violation until the network is forced back into
compliance.

### Results

We ran the model with the Augmented Lagrangian active, initializing
$\lambda_{\text{WEC}} = 0$ and setting the penalty scale $w_{\text{WEC}} = 10.0$.

#### Complete WEC Compliance

During training, the WEC loss ($\mathcal{L}\_{\text{WEC}}$) plummeted rapidly.
Between steps 650 and 690, the telemetry logged a value of exactly
**`0.000000e+00`**. 

Because the WEC loss is the average of the violation over the entire coordinate
grid, hitting exactly zero means the Ricci scalar $R \ge 0$ at **every single
coordinate point** evaluated. The feedback loop worked flawlessly. When the
model drifted slightly to a tiny violation of $\sim 3 \times 10^{-5}$ at step
700, the accumulated Lagrange multiplier immediately kicked in, pulling it back
down to $\approx 8 \times 10^{-6}$ by step 740.

![Loss over time](monitor_training_1_0.png)

#### Visualizing the Spatial Curvature Today

We generated fresh 2D spatial maps of the Ricci scalar curvature today ($t =
1.0$):

![2D Curvature and Shear Maps](monitor_training_4_0.png)

Looking at the bottom row (Ricci Curvature $R$), the colorbars tell a story of
absolute success. In the $x$-$y$ and $y$-$z$ planes, the minimum value is
exactly `0.00`. In the $x$-$z$ plane, the minimum is `0.008`. The negative-mass
shortcuts have been completely erased.

And yet, $\Omega\_m$ was still driven straight to the baryonic floor of `0.05`.
Even under strict general relativity, the universe rejects the homogeneous Dark
Matter assumption.

### The Next Frontier: The past drifts

Enforcing the laws of physics as a hard boundary has solved our negative mass
problem, but it has revealed a new, fascinating artifact in the early universe
($t = -4.0$):

![Hubble and Shear Plots over Time](monitor_training_2_1.png)

In the left panel, the directional Hubble parameters ($H(t)$) start near zero
or slightly negative in the past ($t = -4.0$). In the right panel, the shear
scalar ($\sigma^2(t)$) bends back up to $\approx 8 \times 10^{-5}$ at $t =
-4.0$. 

As the matter density is so low ($\Omega\_m = 0.05$), the universe in the past
($t \in [-4.0, -1.5]$) behaves like a vacuum. Without a heavy background of
matter to drive cosmic expansion and deceleration, and because we have no
observational supernova data in the far past (Supernovae only go up to
$z \approx 2.3$, or $t \approx -1.5$), the neural network's extrapolation is
free to drift. Under pure vacuum field equations, it drifts into a
static/contracting past with positive shear.

But the Cosmic Microwave Background (CMB) tells us that the early universe ($z
\approx 1100$, or $t \ll -4.0$) was highly isotropic and expanding.

### What's Next: Enforcing Isotropy in the Early Universe

To address this, we must introduce the final piece of the physical puzzle: an
**Early-Universe Isotropy and Expansion Prior** at $t \le -3.0$. By penalizing
any shear ($\sigma^2 > 0$) and any negative expansion ($H_{\text{mean}} < 0$) in
the early universe, we will force the model to merge smoothly with standard FLRW
cosmology at high redshifts, matching CMB observations.

Stay tuned!

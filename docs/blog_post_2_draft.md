## Weighing the Universe: Adding BAO and the "Tabula Rasa" Matter Density

In my [previous post](https://hamon.dev/blog/lumpyspace/), I explored how a
Physics-Informed Neural Network (PINN) could learn the 4D metric tensor of the
universe directly from Pantheon+ Supernova data, constrained by the Einstein
Field Equations (EFE). The model discovered that extreme spatial anisotropy (a
"Cosmological Dipole") could act as a physical mechanism to mimic Dark Energy
in a vacuum universe.

But a vacuum universe is a spherical cow[^0]. And Supernovae, while excellent
standard candles, only give us part of the story. They provide luminosity
distances, which act as a proxy for the expansion history of the universe. To
really constrain the geometry, we needed a 3D ruler.

Enter **Baryon Acoustic Oscillations (BAO)**.

### Bypassing Cosmic Chronometers

We chose not to use cosmic chronometers for a very specific reason. Cosmic
chronometers measure the Hubble parameter $H(z)$ directly by looking at the
relative ages of passively evolving galaxies. While incredibly useful, the
[standard 32-point Cosmic Chronometers dataset](https://arxiv.org/abs/1604.00183)
has a critical assumption baked in: isotropy. The data averages $H(z)$ across
all sky coordinates. Since our PINN previously discovered an anisotropic
universe expanding at different rates along different axes to mimic Dark Energy,
testing against a 1D, spherically-averaged $H(z)$ curve would explicitly erase
the very physics we just discovered!

### The Tension of the Vacuum

BAO observations give us a standard ruler based on the clustering of matter left
over from the sound waves of the early universe. I added the SDSS BAO dataset to
the training loop to force the network to fit both distance measures
simultaneously.

#### Anisotropic BAO Ray-Tracing

Supernovae only give us a 1D luminosity distance along the light cone. But BAO
is a 3D standard ruler. In a universe where expansion is different along
different directions, we can't just compute a single radial distance. We have to
separate the transverse distance ($D_M$) and the radial distance ($D_H = 1/H$)
and handle the anisotropy directly.

To do this, I implemented an anisotropic ray-tracing system in
src/physics/geodesics.py. Instead of shooting in one direction, the physics
engine shoots null geodesics backwards from the observer along all three
principal spatial axes ($x, y, z$). For each axis, it integrates the geodesic
equations to find:

1. The path trajectories.
2. The redshift $z$ at each point.
3. The transverse distance $D_M$ from the radial coordinate.
4. The directional Hubble parameter $H_{dir}(z)$ from the spatial metric
   derivatives to get the radial distance $D_H = 1/H_{dir}(z)$.

We then average these directions to get the final averaged distances, comparing
them against the SDSS DR12 dataset using its full covariance matrix. 

This directional ray-tracing also solved a major headache. At the very start of
training, the metric is near-Minkowski, meaning the expansion rate $H \approx 0$.
This caused $D_H = 1/H$ to explode to a massive $10^{11}$ code units,
immediately blowing up the gradients and crashing the training. I fixed this by
capping the unitless radial distance $D_H$ at the spatial boundary limit
($10,000$ Mpc in physical scale) to keep the solver stable until it gets
moving.

### The Ricci Contraction Optimisation

Evaluating General Relativity constraints inside a neural network is incredibly
challenging[^2]. To compute the Einstein Field Equations, the physics engine has
to compute second-order derivatives of the network's metric to get both the
Ricci tensor ($R_{\mu\nu}$) and the Ricci scalar ($R$). 

In the earlier version, JAX was computing the Ricci scalar by running a
separate, independent automatic differentiation trace over the metric. This
meant JAX had to compile and run redundant backpropagation sweeps. I optimized
this in src/training/loss.py by contracting the metric inverse with the already
computed Ricci tensor ($R = g^{\mu\nu} R_{\mu\nu}$ using `jnp.einsum`). By
bypassing the redundant AD pass, the step time dropped significantly, and
combined with gentler learning rate kicks, the training run finally became fast
and stable.

But immediately, the training broke down. The loss landscape became incredibly
rugged, and the gradients were fighting each other because we were forcing the
network to do the impossible. The BAO and Supernova data explicitly demand a
specific expansion history. But under General Relativity (the EFE), a pure
vacuum universe _wants_ to decelerate. By forcing the network to fit the data
using only a vacuum geometry, we were making it contort space and time using
extreme shear just to survive. The tension between the data loss and the physics
loss was tearing the optimizer apart.

If we want the model to fit the observations without breaking Einstein's
equations, we have to give it the physical vocabulary to do so[^1]: **Matter**.

### The Tabula Rasa Philosophy

Normally, in $\Lambda$CDM cosmology, we'd plug in $\Omega_m \approx 0.3$ (about
30% matter, most of which is Dark Matter), a standard assumption validated by
the [Planck 2018 results](https://doi.org/10.1051/0004-6361/201833910). But that
violates the core philosophy of this project: The ultimate goal is to find a
geometry that satisfies the data with _no prior assumptions_. We want to weigh
the universe without putting our thumb on the scale.

So, instead of hardcoding $\Omega_m$, I introduced a trainable matter density
parameter, `kappa_rho_0` ($\kappa \rho_0$).

This parameter acts as a pressureless dust field that dilutes as the spatial
volume of the universe changes ($V \propto \sqrt{\gamma}$). By subtracting this
matter field's Stress-Energy tensor ($T_{\mu\nu}$) from the Einstein tensor
($G_{\mu\nu}$), the physics loss can account for the presence of mass.

Because `kappa_rho_0` is a native network parameter, Equinox automatically
calculates its gradients. The network is now free to dynamically "discover" the
matter density that best balances the EFE against the Supernova and BAO data!

### A Balanced Start

To give the network a fair beginning, I initialized `kappa_rho_0` to the
mathematical equivalent of $\Omega_m = 0.1825$: exactly midway between the
baryonic matter density ($\Omega_m = 0.05$) and the standard $\Lambda$CDM value
($\Omega_m = 0.315$). This gave the network a head start, letting it adjust the
matter density up or down to see if it would gravitate[^4] toward standard dark
matter or push down to the baryonic floor.

### The Single-Parameter Bottleneck

During early testing, I ran into a classic PINN optimization problem:
`kappa_rho_0` was updating extremely slowly. The issue stems from the difference
in parameter scales: The neural network contains over 9,000 weights and biases
that define the metric's spatial and temporal variations. In contrast, the
matter density is a single scalar parameter. 

When the Adam optimizer calculates updates, it applies global gradient norm
clipping to prevent numerical instability. Because the L2 norm of the gradient
vector is calculated across all 9,000+ parameters, the gradients of the network
weights completely dominate the norm. This causes the gradient of the single
scalar `kappa_rho_0` to be scaled down by nearly two orders of magnitude. 

If I increase the global learning rate to make the matter density update faster,
the network weights will become unstable and trigger NaN gradients. If I keep
the learning rate low, the matter density remains flat.

To resolve this, I implemented a gradient multiplier in the training loop. After
the standard backpropagation step, we intercept the gradients and scale the
gradient of `kappa_rho_0` by a factor of 80. This allows the global matter
density to adjust rapidly to satisfy the Einstein Field Equations, while the
network weights can adjust slowly and smoothly to form a consistent space-time
metric.

### Results

![Hubble and Shear Plots](kappa_mid_hubble_shear_plots.png)

After letting the training run converge, the model settled on a stable solution,
though the results are more indicative of initialization bias than a physical
conclusion:

1. **The Trainable Matter Density**:
   The matter density parameter `kappa_rho_0` converged to `0.510`, which
   translates to a physical matter density of:
   $$\Omega_m \approx 0.170$$
   Instead of making a clear physical choice, the parameter simply drifted
   slightly from its starting value of $0.1825$ down to $0.170$. This suggests
   the optimizer settled in the local valley of its initialization rather than
   actively searching the parameter landscape.

2. **Directional Expansion Rates $H(t)$**:
   The expansion rates along $x$ and $z$ ($H_x, H_z$) track each other closely,
   peaking at $\approx 0.046$ before settling to $\approx 0.030$ today, while the
   $y$-axis expands more slowly ($H_y \approx 0.012$) and contracts slightly to
   $\approx -0.010$ today. The average Hubble parameter today is $H_{mean}
   \approx 0.017$.

3. **Shear Scalar $\sigma^2(t)$**:
   The shear scalar remains flat at exactly `0.0` in the early universe, peaking at
   a tiny $\sigma^2 \approx 0.0115$ around $t = -0.1$ before decaying to
   $\approx 0.0084$ today. 

#### What This Means Physically

Because we started the universe with a significant matter component, the physics
engine did not need to distort the geometry to fit the distance data. The
presence of matter handles the deceleration and expansion profiles naturally,
meaning the shear is suppressed to a peak of only $0.0115$ (40 times smaller
than the vacuum run). Physically, once matter is present, the metric behaves
almost exactly like a standard isotropic FLRW cosmology. The optimizer did not
need to use anisotropic shear as a mathematical shortcut to satisfy the EFE.

While it is promising that the 4D PINN can recover a stable, near-isotropic
FLRW-like solution without any hardcoded symmetry assumptions, we need to go
further. Because the matter density barely moved from its starting value, we
must force the model out of this comfortable middle-ground to see if it actually
needs Dark Matter, or if it can fit the data using only baryonic matter.

Ultimately, this run proved that when we relax the FLRW symmetry assumptions,
a universe starting from a middle-ground matter density can fit both supernovae
and BAO data (though not perfectly) with less matter and a highly shearing,
anisotropic geometry, while strictly obeying General Relativity. 

### Starving the Universe

Starting the universe with a comfortable middle-ground matter density proved to
be unhelpful. Because the network began in a relatively stable valley, the
optimizer did not feel any pressure to move $\kappa \rho_0$. To make this a
true test of the cosmological physics, I decided to starve the universe, starting
it with $\Omega_m = 0.05$ ($\kappa \rho_0 = 0.15$), representing a universe with
zero Dark Matter.

Either the model would figure out a space-time that satisfies the observable
universe and EFE with only Baryonic matter, or it would create Dark Matter.

In the initial implementation, the baryonic floor was enforced using a hard
minimum in the loss function: $$\kappa \rho_0 = \text{max}(|\theta|, 0.15)$$

If the optimizer tried to push the matter density lower than the floor, the
parameter would drop below $0.15$, but the loss function would cap the value at
exactly $0.15$. Because the loss is completely flat for any value below $0.15$,
the derivative of the loss with respect to the parameter became exactly zero.
The parameter would be trapped with no gradient signal to pull it back up,
freezing it forever.

To keep the gradients alive across the entire training run, I replaced the hard
cap with a smooth ramp using a shifted Softplus function:
$$\kappa \rho_0 = 0.15 + \text{softplus}(\theta)$$

Because `softplus(x)` is always strictly positive, the physical density is
guaranteed to remain above the $0.15$ baryonic floor. And because the
derivative of `softplus(x)` (which is `sigmoid(x)`) is non-zero everywhere, the
parameter never loses its gradient traction. 

To start the universe exactly at the starved baryonic floor, I initialized the
raw parameter $\theta$ to $-5.0$. This maps to an initial physical density of
$0.1567$ ($\Omega_m \approx 0.052$) while keeping the gradient active.

#### Results

[Place results for Run 2 here once training converges, including the final value
of kappa_rho_0, the directional Hubble parameters, and the shear scalar
behavior.]

[^0]: and, more importantly, doesn't give us anywhere to leave our stuff!

[^1]: and ourselves somewhere to live.

[^2]: understatement.

[^3]: but not us, sadly.

[^4]: did you see what i did there?

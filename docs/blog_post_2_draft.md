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

### The Dark Matter Challenge

To make this a true test of the cosmological physics, I decided to start the
universe with $\Omega_m = 0.05$, representing a universe with only Baryonic
(normal) matter and zero Dark Matter.

If the network can satisfy the observable universe and the EFE with only
baryonic matter, it will do so. If not, it will be forced to dynamically
increase `kappa_rho_0` to "discover" Dark Matter.

Enforcing this baryonic floor correctly is a major challenge:

1. **The Gradient Freezing Problem:**
   In early tests, I enforced the floor using a hard minimum in the loss:
   $$\kappa \rho_0 = \text{max}(|\theta|, \text{Floor})$$
   If the optimizer tried to push the matter density lower than the floor, the
   parameter would drop below it, and the loss function would flatline. Because
   the derivative of a flat function is exactly zero, the parameter would be
   trapped without any gradient signal to pull it back up, freezing it forever.

   To keep the gradients alive, I replaced the hard cap with a smooth ramp
   using a shifted Softplus function:
   $$\kappa \rho_0 = \text{Baryonic Floor} + \text{softplus}(\theta)$$
   Because `softplus(x)` is always strictly positive, the physical density is
   guaranteed to remain above the floor, and because its derivative is
   non-zero everywhere, the parameter never loses its gradient traction. To
   start the universe exactly at the baryonic floor, I initialized the raw
   parameter $\theta$ to $-5.0$ (mapping to an initial physical density very
   close to the minimum floor).

2. **The Dynamic Floor Correction:**
   Normally, one might assume a static floor of $\kappa\rho_0 = 0.05 \times 3$
   (using the standard flat FLRW relation $\kappa\rho_0 = 3\Omega_m H_0^2$ with
   $H_0 \approx 1$). However, because the network is dynamically deriving the
   expansion rate $H(t)$, using a static approximation of $H_0$ is a physical
   inconsistency.

   To correct this, I implemented a dynamically scaled baryonic floor:
   $$\text{Baryonic Floor} = 3 \cdot \Omega_b \cdot H_{mean}(1.0)^2$$
   where $\Omega_b = 0.05$ represents the minimum baryonic matter density today
   relative to the critical density, and $H_{mean}(1.0)$ is the model's own
   derived expansion rate today. This ensures that the boundary constraints
   remain physically self-consistent at every training step.

### Results

[Place results here once training converges, including the final value of
kappa_rho_0, the directional Hubble parameters, the shear scalar behavior,
and the physical interpretation of whether the network "invented" Dark
Matter.]

[^0]: and, more importantly, doesn't give us anywhere to leave our stuff!

[^1]: and ourselves somewhere to live.

[^2]: understatement.

[^3]: but not us, sadly.

[^4]: did you see what i did there?

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

### Why Not Cosmic Chronometers?

You might be wondering why we didn't use cosmic chronometers. Cosmic
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

### Starving the Universe

To make this a true test of the physics engine, I initialized `kappa_rho_0` to
the mathematical equivalent of $\Omega_m = 0.05$.

Why 0.05? Because that is roughly the density of _baryonic_ (normal) matter
in our universe. We are starting the network completely starved of Dark Matter.

This sets up a fascinating experiment. Will the network's optimizer realize that
adding matter is the cheapest way to satisfy the equations, causing
`kappa_rho_0` to climb up to the [Planck 2018 value](https://doi.org/10.1051/0004-6361/201833910)
of $\Omega_m \approx 0.315$? In other words, will the AI _invent_ Dark Matter?

Or will the network decide that Dark Matter is unnecessary, and instead find
an empty, highly anisotropic (shearing) universe that perfectly explains the
Supernovae and BAO data[^2]? What if it pushes the density even lower, deleting
baryonic matter entirely and fitting the data with pure, lumpy spacetime?

### The Results

[Conclusions to be added once training converges...]

---

[^0]: and, more importantly, doesn't give us anywhere to leave our stuff!

[^1]: and ourselves somewhere to live.

[^2]: but not us, sadly.

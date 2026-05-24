# Tabula Rasa: Weighing the Universe

## The $\Lambda$CDM Prior Problem
In the standard $\Lambda$CDM cosmological model, the matter density of the universe is measured to be approximately $\Omega_m \approx 0.315$. This measurement is largely derived from observations of the Cosmic Microwave Background (CMB) by satellites like Planck.

However, extracting this value from the raw CMB power spectrum requires assuming a background geometry—specifically, the perfectly isotropic and homogeneous FLRW metric, along with the existence of Dark Energy ($\Lambda$). 

If the core philosophy of **Tabula Rasa** is to challenge the FLRW metric and explore whether anisotropic "weaponized shear" can replace Dark Energy, we cannot logically hardcode $\Omega_m = 0.3$. Doing so would mean importing a constraint derived from the very mathematics we are trying to invalidate.

## Trainable Matter Density ($\kappa \rho_0$)
To solve this, we map the distribution of matter directly into the neural network's metric, but we leave the absolute density as a trainable parameter.

### The Perfect Fluid
We model galaxies and dark matter as a "pressureless dust." In the comoving frame, the fluid is at rest, meaning the Stress-Energy tensor $T_{\mu\nu}$ only has one non-zero component: the energy density $T_{00} = \rho$.

### Volume Dilution
As the universe expands, this dust dilutes. In an anisotropic universe, the volume expands proportional to the square root of the determinant of the $3 \times 3$ spatial metric ($\sqrt{\gamma}$). Therefore, the local density at any point in spacetime is:
$$ \rho(t, x, y, z) = \frac{\rho_0}{\sqrt{\gamma(t, x, y, z)}} $$

### The Physics Objective
By injecting this dynamic tensor into the Einstein Field Equation residual:
$$ G_{\mu\nu} - 8\pi G T_{\mu\nu} = 0 $$

We allow the neural network to calculate the true vacuum curvature required to fit the Supernova and BAO data, while leaning on a dust field to handle the standard deceleration. 

Because $\kappa \rho_0$ ($8\pi G \rho_0$) is defined as a trainable parameter (`eqx.nn.Parameter` equivalent) within the `MetricNN`, the optimizer will automatically dial the total mass of the universe up or down until it finds the perfect balance that satisfies all observational constraints. 

Tabula Rasa is no longer just fitting curves; it is physically weighing the universe.

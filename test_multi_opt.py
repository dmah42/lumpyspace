import equinox as eqx
import jax
import optax

from src.core.metric import MetricNN

key = jax.random.PRNGKey(0)
model = MetricNN(key)


def label_fn(tree):
  labels = jax.tree_util.tree_map(lambda _: "mlp", tree)
  return eqx.tree_at(lambda m: m.omega_m_raw, labels, "omega")


filtered = eqx.filter(model, eqx.is_array)
labels = label_fn(filtered)
print(labels.omega_m_raw)

optimizers = {
  "mlp": optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4)),
  "omega": optax.adam(1e-2),
}
optimizer = optax.multi_transform(optimizers, label_fn)
opt_state = optimizer.init(filtered)
print("Success")

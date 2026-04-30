import jax.numpy as jnp
x = jnp.zeros((32, 64))

import equinox as eqx
layer = eqx.nn.Linear(128, 64, key=key)

from mypackage.utils import transform
a = transform(x)

x = foo(bar(y))

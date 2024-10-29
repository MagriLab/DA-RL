import jax.numpy as jnp


def get_const(std, y):
    return jnp.diag((std * jnp.ones(len(y)))) ** 2


def get_max(std, y):
    return (jnp.diag((std * jnp.ones(len(y)))) * jnp.max(jnp.abs(y), axis=0)) ** 2


def get_prop(std, y):
    return (jnp.diag((std * jnp.abs(y)))) ** 2

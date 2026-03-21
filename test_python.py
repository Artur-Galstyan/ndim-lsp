from jaxtyping import Array, Float


def f(x: Float[Array, "a b"], y: Float[Array, "c a"]):
    return x @ y

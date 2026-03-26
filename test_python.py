from jaxtyping import Array, Float


def f(x: Float[Array, "a b"], y: Float[Array, "b a"]):
    z = x @ y
    return z


def g(a: Float[Array, "a b"], b: Float[Array, "c a"]):
    return a @ b


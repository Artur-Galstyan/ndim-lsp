from jaxtyping import Array, Float


# valid matmul: b matches b
def valid_matmul(x: Float[Array, "a b"], y: Float[Array, "b c"]):
    return x @ y


# invalid matmul: b != c
def invalid_matmul(x: Float[Array, "a b"], y: Float[Array, "c a"]):
    return x @ y


# shape tracing: hover z should show (a, c)
def trace_shape(x: Float[Array, "a b"], y: Float[Array, "b c"]):
    z = x @ y
    return z


# same param names, different shapes across functions
def scoped_a(x: Float[Array, "a b"], y: Float[Array, "b c"]):
    return x @ y




def scoped_b(x: Float[Array, "m n"], y: Float[Array, "p q"]):
    return x @ y

# chained matmul: z = x @ y, then z @ w
def chained(x: Float[Array, "a b"], y: Float[Array, "b c"], w: Float[Array, "c d"]):
    z = x @ y
    return z @ w




# sequential: z traced, then used
def sequential(x: Float[Array, "a b"], y: Float[Array, "b c"], v: Float[Array, "c d"]):
    z = x @ y
    w = z @ v
    return w
 
"""

# chained: x @ y @ w in one expression (won't work yet)
def chained_expr(
    x: Float[Array, "a b"], y: Float[Array, "b c"], w: Float[Array, "c d"]
):
    return x @ y @ w
"""
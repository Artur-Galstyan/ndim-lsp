import foo
import foo.bar as baz
from jax import random
from equinox.nn import Linear as Lin
from ._linear import Linear
from . import layers
from foo import *

class Linear:
    pass

def linear():
    pass

def outer():
    class NestedLinear:
        pass

    def nested_fn():
        pass

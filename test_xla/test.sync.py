# %%

import example
import jax.numpy as jnp
from jaxlib import xla_client

xla_client.register_custom_call_target(b"py_add_xla", example.py_add(), platform="gpu")

# %%

a = jnp.arange(128, dtype=jnp.float32)
b = jnp.arange(128, dtype=jnp.float32)

# %%

from jax import core

py_add_p = core.Primitive("py_add")

def py_add_prim(a, b):
    return py_add_p.bind(a, b)

@py_add_p.def_impl
def py_add_impl(a, b):
    return "hello world"

from jax._src import abstract_arrays

@py_add_p.def_abstract_eval
def py_add_abeval(a, b):
    assert a.shape == b.shape
    assert a.ndim == 1
    assert a.shape[0] == 128
    return abstract_arrays.ShapedArray(a.shape, a.dtype)


# %%

from jax.interpreters import xla
import numpy as np

xops = xla_client.ops

c = xla_client.XlaBuilder("comp_builder")

def py_add_translation(xla_builder, a, b):
    shape = xla_client.Shape.array_shape(np.dtype("float32"), (128,), (0,))
    opaque = b"This is opaque"
    print(f"Type of a is {type(a)}")
    print(f"Type of shape is {type(shape)}")
    return xops.CustomCallWithLayout(
        xla_builder,
        b"py_add_xla",
        operands=(a, b),
        shape_with_layout=shape,
        operand_shapes_with_layout=(shape, shape),
        opaque=opaque,
        # has_side_effect=False,
        # schedule=0,
        # api_version=1,
    )

xla.backend_specific_translations["gpu"][py_add_p] = py_add_translation

"""
    builder: XlaBuilder,
    call_target_name: bytes,
    operands: Sequence[XlaOp],
    shape_with_layout: Shape,
    operand_shapes_with_layout: Sequence[Shape],
    opaque: bytes = ...,
    has_side_effect: bool = ...,
    schedule: CustomCallSchedule = ...,
    api_version: CustomCallApiVersion = ...,
"""

shape = xla_client.Shape.array_shape(np.dtype("float32"), (128,), (0,))
shape

# %%

import jax

c = jax.jit(py_add_prim)(a,b)

# %%

c

# %%

d = py_add_prim(a,b)
d

# %%

d

# %%

!bash compile.sh

import jax
import jax.numpy as jnp
import fmha
from jaxlib import xla_client

xla_client.register_custom_call_target(b"local_softmax", fmha.fmha(), platform="gpu")

# %%

from jax import core

sm_p = core.Primitive("local_softmax")

def local_sm_prim(a):
    return sm_p.bind(a)

@sm_p.def_impl
def py_add_impl(a):
    return "local_softmax"

from jax._src import abstract_arrays

@sm_p.def_abstract_eval
def py_add_abeval(a):
    assert a.shape == (64,64)
    return abstract_arrays.ShapedArray(a.shape, a.dtype)

# %%

from jax.interpreters import xla
import numpy as np

xops = xla_client.ops

c = xla_client.XlaBuilder("comp_builder")

def local_sm_translation(xla_builder, a):
    shape = xla_client.Shape.array_shape(np.dtype("float32"), (64,64), (1,0))
    br_shape = xla_client.Shape.array_shape(np.dtype("float32"),(64,),(0,))
    opaque = b"This is opaque"
    print(f"Type of a is {type(a)}")
    print(f"Type of shape is {type(shape)}")
    return xops.CustomCallWithLayout(
        xla_builder,
        b"local_softmax",
        operands=(a,),
        shape_with_layout=shape,
        operand_shapes_with_layout=(shape,),
        opaque=opaque,
        # has_side_effect=False,
        # schedule=0,
        # api_version=1,
    )

xla.backend_specific_translations["gpu"][sm_p] = local_sm_translation

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

# %%

from jax import random
rng = random.PRNGKey(42)
mat = random.normal(rng,(64,64))
# mat = jnp.ones((64,64)) * 2
b = jax.jit(local_sm_prim)(mat)
c = jax.nn.softmax(mat)

# %%

mat[0]

# %%

b

# %%

c
# %%

jnp.sum(b-c)

# %%

jnp.sum(jnp.exp(mat[0]-jnp.max(mat[0])))

# %%

m = jnp.max(mat[0])
fx = jnp.exp(mat[0]-m)
fx/jnp.sum(fx)

# %%


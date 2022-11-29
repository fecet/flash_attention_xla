# %%

!bash compile.sh

import jax
import jax.numpy as jnp
import fmha
from jaxlib import xla_client
import jaxlib

xla_client.register_custom_call_target(b"local_softmax", fmha.fmha(), platform="gpu")


# %%

from jax import core

fmha_p = core.Primitive("fmha")

def fmha_prim(q,k,v):
    return fmha_p.bind(q,k,v)

@fmha_p.def_impl
def impl(q,k,v):
    print("FMHA")

from jax._src import abstract_arrays

@fmha_p.def_abstract_eval
def abs_eval(q,k,v):
    assert q.shape == v.shape
    assert q.shape[0] == k.shape[1]
    assert q.shape[1] == k.shape[0]
    return abstract_arrays.ShapedArray(q.shape, q.dtype)

# %%

from jax.interpreters import xla
import numpy as np

xops = xla_client.ops

c = xla_client.XlaBuilder("comp_builder")

def local_sm_translation(xla_builder, q, k, v):
    qv_shape = xla_builder.get_shape(q)
    k_shape = xla_builder.get_shape(k)

    qv_dim = qv_shape.dimensions()
    k_dim = k_shape.dimensions()
    qv_shape = xla_client.Shape.array_shape(np.dtype("float32"), qv_dim, (1,0))
    k_shape = xla_client.Shape.array_shape(np.dtype("float32"), k_dim, (1,0))
    # int_shape = xla_client.Shape.array_shape(np.dtype("int32"), (2,), (0,))
    # opaque = b"This is opaque"
    N = qv_dim[0]
    d = qv_dim[1]
    opaque = N.to_bytes(4,"little")
    opaque += d.to_bytes(4,"little")
    
    return xops.CustomCallWithLayout(
        xla_builder,
        b"local_softmax",
        operands=(q,k,v),
        shape_with_layout=qv_shape,
        operand_shapes_with_layout=(qv_shape,k_shape,qv_shape),
        opaque=opaque,
        # has_side_effect=False,
        # schedule=0,
        # api_version=1,
    )

xla.backend_specific_translations["gpu"][fmha_p] = local_sm_translation

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

N = 128
d = 128
shape = (N,d)
q = jnp.ones(shape)
k = jnp.ones(shape).T
v = jnp.ones(shape)


(jax.jit(fmha_prim)(q,k,v)).shape

# %%

jax.jit(fmha_prim)(q,k,v)




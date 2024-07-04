# Having fast chip interconnect makes it much easier to have good performance in multichip
# we were computing batch * E * E * E to get the next activation layer

import numpy as np
import jax
import jax.numpy as jnp

A = jax.numpy.ones((1024, 1024))

print(f'{jax.devices()}')
print(f'{A.devices()}')

# Define the names of each rank in jax.devices()
mesh = jax.sharding.Mesh(jax.devices(), "myaxis")
mesh = jax.sharding.Mesh(np.reshape(jax.devices(), (2,2)), ["myaxis1", "myaxis2"])
# Map our array
# (None, None) = Fully replicated
# (None, "axis") = Sharded over columns
# ("axis", None) OR ("axis") = Sharded over rows

# MULTIPLE AXISES allow you to partion the data in different dimensions.
# ParitionSpec("A", "B") will create AxB grid
# ParitionSpec(["A", "B"], None will create a A+B rows)
# ParitionSpec(["A", "B"], None will create a A+B rows)

p1 = jax.sharding.PartitionSpec("myaxis1","myaxis2")
p2 = jax.sharding.PartitionSpec("myaxis2","myaxis1")
# Create a sharding with the axis names defining how the ranks are ordered
sharding1 = jax.sharding.NamedSharding(mesh, p1)
sharding2 = jax.sharding.NamedSharding(mesh, p2)
# apply the sharding to our input data
sharded_A1 = jax.device_put(A, sharding1)
sharded_A2 = jax.device_put(A, sharding2)

# Jax will decide for us an efficent solution if we combine sharded arrays together.
output = sharded_A1 + sharded_A2

print(f'{jax.debug.visualize_array_sharding(A)}')
print(f'{jax.debug.visualize_array_sharding(sharded_A1)}')
print(f'{jax.debug.visualize_array_sharding(sharded_A2)}')
print(f'{jax.debug.visualize_array_sharding(output)}')
# print(f'{jax.debug.inspect_array_sharding(A)}')
# used behind the scenes of visualize_array_sharding.
# print(f'{jax.debug.visualize_sharding(sharded_A)}')

# What happens if the array doesn't fit into the TPU?
# jax.debug.breakpoint()
print(f'{sharded_A1.addressable_shards[0].data.shape}')

# MATMULs are not easy to parallelize so... how do we handle that?

# we want every activation layer to be sharded the same way
# we know the weights in advance
# we only get act n+1 after we compute it.

# choice:
# Activation Sharding (tensor / Model Parallelism)
# Batch Sharding (data parallelism)
# - We can separate batches easily and each chip can do the computation on its own so it is nice and easy
# - Fails at large models like GPT3 with 700GB.

# - When the model doesn't fit in HBM what do we do?
# -- 1. interchip interconnect (since 2.16 Tb/s speed) or PCIE or DCN is too slow.

#  HBM:  275 TFLOP/s /  9.8 Tb/s =    224 FLOPs/byte
# ○ Fast but there’s too little HBM
# ● ICI:  275 TFLOP/s / 2.16 Tb/s =   1018 FLOPs/byte
# ○ Let’s try it!
# ● PCIE: 275 TFLOP/s /   64 Gb/s =  34375 FLOPs/byte
# ● DCN:  275 TFLOP/s /  25Gb/s = 88000 FLOPs/byte

# -- ICI can work but we have to do more work (1018 FLOPS / byte) to avoid being memory bound / bandwidth bound.

# Trick one: we shard both the batches and the weights. (FSDP)
# Trick two: we parallelize the all gather of weights for the layer n while doing the matmul compuation of n-1.
#            all gather should take much shorter than malmul.
# Notes, there are reduce scatter that needs to be done before the next training step to update at some point.
# When do we run reduce scatter? This seems useful to update the replicas with weights learned from each relica from their unique batches.

# FSDP stops working on very big LLM, if the sequence length is 65k then one replica / device can't handle the size. basically activations are not sharded.
# FSDP doesnt help for inference where latency is key.

# Activation sharding:
# We do sharding on activation and weights per replica. We do a matmul locally but there NEEDs to be a reduce scatter in order to get the right results.
# This is much harder to overlap. Compute the first row, do reduce scatter?
# Somehow pipeline parallelism is needed because row of matrix mat mul, then reducce scatter.
# TP is less attractive...

# Hybrid between FSDP and TP works best.

# TP < FSDP < FSDP + TP

## LETS TEST this with MaxText
export LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

python3 ~/maxtext/pedagogical_examples/shardings.py --ici_fsdp_parallelism 4 --ici_tensor_parallelism 1 --batch_size 4096 --embedding_dimension 16384  --profiler_path /tmp/sharding/
# time is 0.3918472 seconds, TFLOP is 105.553116266496, TFLOP/s is 269.373154297124
python3 ~/maxtext/pedagogical_examples/shardings.py  --ici_fsdp_parallelism 4 --ici_tensor_parallelism 4 --batch_size 4096 --embedding_dimension 16384  --profiler_path /tmp/sharding/
#
 python3 ~/maxtext/pedagogical_examples/shardings.py  --ici_fsdp_parallelism 4 --ici_tensor_parallelism 1 --batch_size 4096 --embedding_dimension 16384

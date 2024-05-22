import jax
import jax.numpy as jnp
from rafi_preclass_prep import timing_util
#  For convenience, JAX provides jax.numpy which closely mirrors the numpy API and provides easy entry into JAX
# numpy is for doing array / matrix operations with optimizations made for CPU
# https://numpy.org/doc/stable/user/quickstart.html. Structure contains ndarray.ndim, .shape, .size, .dtype, .itemsize
import datetime

MATRIX_DIM = 32768
STEPS = 10
TERRA = 10e12
GIGA = 10e9
MILLI = 10e3
A = jax.numpy.ones((MATRIX_DIM, MATRIX_DIM), jnp.float32)
B = jax.numpy.ones((MATRIX_DIM, MATRIX_DIM), jnp.float32)

num_bytes = A.size * A.itemsize
total_num_bytes_crossing = 3 * num_bytes
# 3 because we cross to HBM for:
# get A
# get B
# store C

num_flops = MATRIX_DIM * MATRIX_DIM
# flops are determined by operations done. We are doing one add per element in the matrix.
# ops are done in float32 in TPU case.
# TRACE_NAME = "/tmp/profile_me"
# jax.profiler.start_trace(TRACE_NAME)
# # Timing initially is wrong because not all adds are doing the same thing???
# # Something in jax...
# starttime = datetime.datetime.now()
# for i in range(STEPS):
#     C = A + B
# endtime = datetime.datetime.now()
# time_taken = (endtime - starttime).total_seconds() / STEPS
# print(f"Time taken: {time_taken}")
# jax.profiler.stop_trace()

def f(A, B):
    return A + B

time_taken = timing_util.simple_timeit(f, A, B, task="add_two_nums") / MILLI




# State of the art models will get 60% of the optimal flops.
# but we can do better.
print(
    f"TFlops Per Sec {num_flops / time_taken / TERRA}, GBytes Per Sec {total_num_bytes_crossing / time_taken / GIGA}"
)

# Trace Dir: /tmp/t_add_two_nums_T5KHOWNF2A
# add_two_nums: average time milliseconds: 13.20, trace /tmp/t_add_two_nums_T5KHOWNF2A
# TFlops Per Sec 0.08131881945759273, GBytes Per Sec 975.8258334911127
# Trace Dir: /tmp/t_add_three_nums_H7TFQX8T9L
# add_three_nums: average time milliseconds: 26.21, trace /tmp/t_add_three_nums_H7TFQX8T9L
# Time taken 0.00262147 TFlops Per Sec 0.08191906251072872, GBytes Per Sec 983.0287501287445
# Trace Dir: /tmp/t_add_three_numbers_jit_7ERE3TXAHA
# add_three_numbers_jit: average time milliseconds: 18.95, trace /tmp/t_add_three_numbers_jit_7ERE3TXAHA
# Time taken 0.0018953200000000003 TFlops Per Sec 0.11330454213536499, GBytes Per Sec 906.43633708292

# Open tensorboard
# tensorboard --logdir DIR

# GPUs and TPUs have similar problems where onchip memory is small and need to use
# HBM to store larger things.
# The trend is the FLOPs of the Compute is increasing but the bandwidth is not increasing.


# Now how fast should A+B+C (adding 3 numbers together)

A = jax.numpy.ones((MATRIX_DIM, MATRIX_DIM), jnp.float32)
B = jax.numpy.ones((MATRIX_DIM, MATRIX_DIM), jnp.float32)
C = jax.numpy.ones((MATRIX_DIM, MATRIX_DIM), jnp.float32)

num_flops = MATRIX_DIM * MATRIX_DIM * 2
num_bytes = A.size * A.itemsize
total_num_bytes_crossing = 6 * num_bytes

def f3(A, B, C):
    return A + B + C

time_taken = timing_util.simple_timeit(f3, A, B, C, task="add_three_nums") / MILLI




# State of the art models will get 60% of the optimal flops.
# but we can do better.
print(
    f"Time taken {time_taken} TFlops Per Sec {num_flops / time_taken / TERRA}, GBytes Per Sec {total_num_bytes_crossing / time_taken / GIGA}"
)
# This takes twice as long since this requires 6 transfers over HBM.


# How can we avoid transfers? Well we have some SRAM we can use.
# Pipelining / Partition / Loop Fision (splitting the operations)
# Take the first chuck of them.

# This take 4/6th the time in A + B + C because we transfer 4 things across
# boundaries in total vs 6.

# Simple idea

# for i in range(NUM_PARTS):
#     D[i] = A[i] + B[i] + C[i]
total_num_bytes_crossing = 4 * num_bytes

f3_jit = jax.jit(f3)

time_taken = timing_util.simple_timeit(f3_jit, A, B, C, tries=STEPS, task="add_three_numbers_jit") / MILLI

print(
    f"Time taken {time_taken} TFlops Per Sec {num_flops / time_taken / TERRA}, GBytes Per Sec {total_num_bytes_crossing / time_taken / GIGA}"
)

# Jax JIT is basically allowing us to compile in advance and trace the function.
# Without Jax JIT we are running in "eager" mode sometimes called.
# Jit of f(a+b) will not improve things since there is nothing to fuse
# Will jit results always be shown as fusions?

#pytorch jax etc have the ability to do fusion

# A + B is highly bandwidth bound since we are not using all the flops of the TPU

# Matmul:
#

# Matrix Multiplers

# [ 1 2 ]    [1 2 3]
# [ 3 4 ]  x [4 5 6]
# [ 5 6 ]

# Row * column
# [1 * 1 + 2 * 4, 1 * 2 + 2 * 5, 1 * 3 +  2 * 6]
# [3 * 1 + 4 * 4 ...]
# [...]

# FLOPS = 2xyz, Input 2xy + 2yz, Output 2xz assuming 2 bytes per element

num_bytes = A.size * A.itemsize
total_num_bytes_crossing = 3 * num_bytes
# 3 because we cross to HBM for:
# get A
# get B
# store C

num_flops = MATRIX_DIM * MATRIX_DIM * MATRIX_DIM * 2


def f_matmul(A, B):
    return A@B

time_taken = timing_util.simple_timeit(f_matmul, A, B, tries=STEPS, task="matmul_2_numbers") / MILLI

print(
    f"Time taken {time_taken} TFlops Per Sec {num_flops / time_taken / TERRA}, GBytes Per Sec {total_num_bytes_crossing / time_taken / GIGA}"
)


matmul_jit = jax.jit(f_matmul)

time_taken = timing_util.simple_timeit(matmul_jit, A, B, tries=STEPS, task="matmul_2_numbers_jit") / MILLI

print(
    f"Time taken {time_taken} TFlops Per Sec {num_flops / time_taken / TERRA}, GBytes Per Sec {total_num_bytes_crossing / time_taken / GIGA}"
)

num_flops = MATRIX_DIM * MATRIX_DIM * MATRIX_DIM * 2 + MATRIX_DIM * MATRIX_DIM


def f_relu(A, B):
    return jax.nn.relu(A@B)

time_taken = timing_util.simple_timeit(f_relu, A, B, tries=STEPS, task="relu_2_numbers") / MILLI

print(
    f"Time taken {time_taken} TFlops Per Sec {num_flops / time_taken / TERRA}, GBytes Per Sec {total_num_bytes_crossing / time_taken / GIGA}"
)

f_relu_jit = jax.jit(f_relu)
print(f"hlo for relu jit {f_relu_jit.lower(A,B).as_text()}")

time_taken = timing_util.simple_timeit(f_relu_jit, A, B, tries=STEPS, task="relu_2_numbers_jit") / MILLI

print(
    f"Time taken {time_taken} TFlops Per Sec {num_flops / time_taken / TERRA}, GBytes Per Sec {total_num_bytes_crossing / time_taken / GIGA}"
)

# RELU / MATMUL is flops bound.

print(
    f"Our Arithmetic intensity {num_flops / total_num_bytes_crossing}"
)

# Compare the AI of our model vs the AI of the hardware (275 TB / 1250 GB)


for MATRIX_DIM in [64, 128, 256, 512, 1024, 2048, 4096]:
    STEPS = 10
    NUM_MATRICES = 2**28 // MATRIX_DIM ** 2
    A = jax.numpy.ones((NUM_MATRICES, MATRIX_DIM, MATRIX_DIM), dtype=jax.numpy.bfloat16)
    B = jax.numpy.ones((NUM_MATRICES, MATRIX_DIM, MATRIX_DIM), dtype=jax.numpy.bfloat16)

    num_flops = MATRIX_DIM * MATRIX_DIM * MATRIX_DIM * 2 * NUM_MATRICES
    num_bytes = A.size * A.itemsize
    total_num_bytes_crossing = 3 * num_bytes

    @jax.jit
    def f_batch_matmul(A,B):
        return jax.lax.batch_matmul(A, B)

    average_time_sec_jit  = timing_util.simple_timeit(f_batch_matmul, A, B, task= "jit_f") / MILLI

    print(f"{MATRIX_DIM=}")
    print(
        f"Our Arithmetic intensity {num_flops / total_num_bytes_crossing}"
    )

    print(
        f"Time taken {average_time_sec_jit} TFlops Per Sec {num_flops / average_time_sec_jit / TERRA}, GBytes Per Sec {total_num_bytes_crossing / average_time_sec_jit / GIGA}"
    )
    print("\n\n\n")

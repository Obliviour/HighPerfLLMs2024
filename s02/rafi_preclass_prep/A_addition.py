import datetime
import jax

import timing_util

MATRIX_SIZE = 32768
STEPS = 10
TERRA = 10e12
GIGA = 10e9
MILLI = 10e3

######## EXAMPLE 1 START #########

# Memory is far from the chip in TPUs.
# 160 MB of SRAM, 1.23 TB/s to get to 32 GB of HBM.
# There are cost factors that limit SRAM size.
A = jax.numpy.ones ( (MATRIX_SIZE, MATRIX_SIZE))
B = jax.numpy.ones ( (MATRIX_SIZE, MATRIX_SIZE))

num_bytes = A.size * A.itemsize
total_num_bytes_crossing = 3 * num_bytes
num_flops = MATRIX_SIZE * MATRIX_SIZE


jax.profiler.start_trace("/tmp/wrong_way_3")

s = datetime.datetime.now()
for i in range(STEPS):
    O = A + B
e = datetime.datetime.now()

jax.profiler.stop_trace()

time_taken = (e-s).total_seconds()/STEPS
print( f"Straight addition takes {(e-s).total_seconds()/STEPS:.4f}")
print(
    f"TFlops Per Sec {num_flops / time_taken / TERRA}, GBytes Per Sec {num_bytes / time_taken / GIGA}"
)
######## EXAMPLE 1 END #########


######## EXAMPLE 2 START #########

A = jax.numpy.ones ( (MATRIX_SIZE, MATRIX_SIZE))
B = jax.numpy.ones ( (MATRIX_SIZE, MATRIX_SIZE))
C = jax.numpy.ones ( (MATRIX_SIZE, MATRIX_SIZE))

s = datetime.datetime.now()
for i in range(STEPS):
    O = A + B + C
e = datetime.datetime.now()

print( f"Two additions takes {(e-s).total_seconds()/STEPS:.4f}")


######## EXAMPLE 2 END #########

######## EXAMPLE 3 START #########

def f3(X,Y,Z):
    return X+Y+Z

f3_jit = jax.jit(f3)



A = jax.numpy.ones ( (MATRIX_SIZE, MATRIX_SIZE))
B = jax.numpy.ones ( (MATRIX_SIZE, MATRIX_SIZE))
C = jax.numpy.ones ( (MATRIX_SIZE, MATRIX_SIZE))


s = datetime.datetime.now()
for i in range(STEPS):
    O = f3(A, B, C)
e = datetime.datetime.now()


print( f"Two additions takes {(e-s).total_seconds()/STEPS:.4f} with magic JIT sauce")

######## EXAMPLE 3 END #########

###### EXAMPLE 4 START #####

def f2(A,B):
    return A + B
f2_jit = jax.jit(f2)


timing_util.simple_timeit(f2, A, B, task = "f2")
timing_util.simple_timeit(f2_jit, A, B, task = "f2_jit")


timing_util.simple_timeit(f3, A, B, C, task = "f3")
timing_util.simple_timeit(f3_jit, A, B, C, task = "f3_jit")


###### EXAMPLE 4 END #####

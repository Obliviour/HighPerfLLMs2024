import jax
import numpy as np
import jax.numpy as jnp
import random
import string
import time

MATRIX_DIMS = [128, 256, 512, 1024, 2048, 2560, 3072, 4096, 5121, 6144, 8192, 16384, 32768]
NUM_TRIES = 100

# Enable Tracing
trace_name = f"t_matmul_two_" + "".join(
    random.choice(string.ascii_uppercase + string.digits) for _ in range(10)
)
trace_dir = f"/tmp/{trace_name}"
print(f"Trace Dir: {trace_dir}")
jax.profiler.start_trace(trace_dir)


@jax.jit
def matmul_matrixes(a, b):
    return a @ b


for matrix_dim in MATRIX_DIMS:
    scope_name = f"{matrix_dim} Multiply Test"
    with jax.named_scope(scope_name):
        a = jnp.ones((matrix_dim, matrix_dim), jnp.bfloat16)
        b = jnp.ones((matrix_dim, matrix_dim), jnp.bfloat16)

        # matmul_two_jit = jax.jit(matmul_matrixes(a, b))

        jax.block_until_ready(matmul_matrixes(a, b))
        print(a.size)
        time_taken = 0
        for _ in range(NUM_TRIES):
            start_time = time.time()
            jax.block_until_ready(matmul_matrixes(a, b))
            end_time = time.time()
            time_taken += end_time - start_time
        time_taken = time_taken / NUM_TRIES
        flops_per_s = matrix_dim**3 * 2 / time_taken / 1e12
        bw_per_s = matrix_dim**2 * 3 * 2 / time_taken / 1e9
        bw_per_sec = num_bytes = a.size * a.itemsize * 3
        print(
            f"Matrix Dim: {matrix_dim}: Time taken in seconds: {time_taken}, TFLOPS/sec: {flops_per_s}, GBW/sec: {bw_per_s}"
        )

jax.profiler.stop_trace()


# On v4 we should see that

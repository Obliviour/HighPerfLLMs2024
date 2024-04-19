"""
pip install tensorflow tensorflow_datasets jax flax numpy optax
git clone https://github.com/google/maxtext.git
sudo apt update && sudo apt upgrade
bash maxtext/setup.sh
"""

import tensorflow as tf
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp

# Flax is an open source python neural network library built on top of Jax
# https://flax.readthedocs.io/en/latest/quick_start.html
import flax.linen as nn
from flax.training import train_state

import flax.linen.attention as attention

import numpy as np

import optax

BATCH_IN_SEQUENCES = 384
SEQUENCE_LENGTH = 128

VOCAB_DIM = 256
EMBED_DIM = 512
FF_DIM = 2048

LAYERS = 4

HEAD_DEPTH = 128
NUM_HEADS = 4

LEARNING_RATE = 1e-3

# https://flax.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html#module-basics


class OurSimpleModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        embedding = self.param(
            "embedded", nn.initializers.normal(1), (VOCAB_DIM, EMBED_DIM), jnp.float32
        )
        x = embedding[x]  ## OUTPUT should be [BATCH, SEQUENCE, EMBED]
        return x

class NNModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        embedding = self.param(
            "embedded", nn.initializers.normal(1), (VOCAB_DIM, EMBED_DIM), jnp.float32
        )
        x = embedding[x]  ## OUTPUT should be [BATCH, SEQUENCE, EMBED]


        # Q: So since we set the type to jnp.float32, I guess the nn is doing processing
        # in float32.

        # NN layers
        for i in range(LAYERS):
            feedforward = self.param(
                "feedforward_" + str(i),
                nn.initializers.lecun_normal(),
                (EMBED_DIM, FF_DIM),
                jnp.float32,
            )
            x = embedding[x]
        return x

class OurModel(nn.Module):

    # __call__ is run when the OurModel.apply(inputs) is run.
    @nn.compact
    def __call__(self, x):
        """
        x is [BATCH, SEQUENCE]
        """
        # Each of the characters should be mapped to a vector to determine the relationship to other characters.
        embedding = self.param(
            "embedding",
            nn.initializers.normal(1),
            (VOCAB_DIM, EMBED_DIM),
            jnp.float32,
        )
        x = embedding[x]  ##OUTPUT should be [BATCH, SEQUENCE, EMBED]

        for i in range(LAYERS):
            feedforward = self.param(
                "feedforward_" + str(i),  # include the layer name using str(i).
                nn.initializers.lecun_normal(),
                (EMBED_DIM, FF_DIM),
                jnp.float32,
            )
            x = x @ feedforward  # this is a maxtrix multiply.
            x = jax.nn.relu(x)
            embed = self.param(
                "embed_" + str(i),
                nn.initializers.lecun_normal(),
                (FF_DIM, EMBED_DIM),
                jnp.float32,
            )
            x = x @ embed
            x = jax.nn.relu(x)

        return x @ embedding.T


def convert_to_ascii(string_array, max_length):
    result = np.zeros((len(string_array), max_length), dtype=np.uint8)
    for i, string in enumerate(string_array):
        for j, char in enumerate(string):
            # So for each string array we only grab the first SEQUENCE_LENGTH strings from it.
            # Q: What happens to the rest of the string_array? we lose it?
            if j >= SEQUENCE_LENGTH:
                break
            result[i, j] = char
    return result


# this function provides the expected output which is the input shifted by one.
# WAIT THIS function is the opposite. It actually goes from outputs to inputs.
"""
np_array (output)
[BATCH_IN_SEQUENCE x SEQUENCE_LENGTH]
[[b[0, 0] ... [b[0, SEQUENCE_LENGTH]]]


returns (input)
[0, b[0, 0] ... b[0, SEQUENCE_LENGTH - 1]]
"""


def input_to_output(np_array) -> np.ndarray:
    zero_array = np.zeros((BATCH_IN_SEQUENCES, SEQUENCE_LENGTH), dtype=jnp.uint8)
    zero_array[:, 1:SEQUENCE_LENGTH] = np_array[:, 0 : SEQUENCE_LENGTH - 1]
    return zero_array


def calculate_loss(params, model, inputs, outputs):
    proposed_outputs = model.apply(params, inputs)
    one_hot = jax.nn.one_hot(outputs, VOCAB_DIM)
    loss = optax.softmax_cross_entropy(proposed_outputs, one_hot)
    return jnp.mean(loss)


def main():
    # Load "language model 1 billion benchmark dataset, with train data only."
    # Important to have good data, customers / companies currate their own.

    # Q: What are other ways of loading data?
    # ds = tf.data.Dataset()
    ds = tfds.load("lm1b", split="train", shuffle_files=False)
    # each batch has BATCH_IN_SEQUENCES number of strings.
    # we later will want the token length / sequence length to be SEQUENCE_LENGTH
    ds = ds.batch(BATCH_IN_SEQUENCES)

    # This example will show us BATCH_IN_SEQUENCES strings
    for example in ds:
        # Each tfds dataset batch needs to be parsed by label, and .numpy.
        print(
            f'{example}\n, {example["text"].numpy()}\n {len(example["text"].numpy())},'
        )
        # We determine the outputs by converting the dataset into tokens through a simply ascii function.
        output = convert_to_ascii(example["text"].numpy(), SEQUENCE_LENGTH)
        # The LLM is being ask to predict 'output' based on 'input'.
        # 0 -> 84 (T) -> 104 (h) -> 101 (e)
        # Q: Why do we start with 0?
        input = input_to_output(output)
        print(
            f"OUTPUTS: {output.shape},\n {output} \n INPUTS: {input.shape},\n {input}"
        )
        break

    #### SIMPLE model apply
    rngkey = jax.random.key(0)
    simple_model = OurSimpleModel()
    simple_model_params = simple_model.init(
        rngkey, jnp.empty((BATCH_IN_SEQUENCES, SEQUENCE_LENGTH), dtype=jnp.uint8)
    )

    for example in ds:
        example = example["text"].numpy()
        output = convert_to_ascii(example, SEQUENCE_LENGTH)
        input = input_to_output(output)
        y = simple_model.apply(simple_model_params, input)
        # This applies the embedded to our input. aka giving a vector of values to each character.

        print(f"{y}\n{y.shape}")
        # 384 (number of batches) x (128) SEQUENCE_LENGTH x (512) EMBEDDING
        # Multilayer Perception: basically we 512 is not "right" to figure which of each character (256 choices).
        # So how do we get there? Using Multilayer Perception: aka doing matrix multiplications that change the layer size.
        # Batch, Sequence, Embed x Embed, Vocab = Batch Sequence Vocab.

        # calculating the entrophy?
        break
    ####

    # # We need random number generator. Later sessions describe why.
    # rngkey = jax.random.key(0)
    # model = OurModel()
    # # Init the model with random number, something with the right shape, and the data type we are using.
    # # jnp.empty((BATCH_IN_SEQUENCES, SEQUENCE_LENGTH), dtype = jnp.uint8)) should also work.
    # _params = model.init(rngkey, jnp.ones((BATCH_IN_SEQUENCES, SEQUENCE_LENGTH), dtype = jnp.uint8))

    # tx = optax.adam(learning_rate = LEARNING_RATE)
    # state = train_state.TrainState.create(
    #    apply_fn = model.apply,
    #    params = _params,
    #    tx = tx
    # )

    # iter = 0
    # for example in ds:
    #    outputs = convert_to_ascii(example['text'].numpy(), SEQUENCE_LENGTH)
    #    inputs = input_to_output(outputs)

    #    loss, grad = jax.value_and_grad(calculate_loss)(state.params, model, inputs, outputs)
    #    state = state.apply_gradients(grads = grad)
    #    print(f"{iter} -> {loss}")
    #    iter += 1


if __name__ == "__main__":
    main()

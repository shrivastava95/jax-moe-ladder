# lets make a
# 0. proper random initialization with jax
# 1. linear layer
# 2. ffn
# 3. rmsnorm
# 4. layernorm
# 5. adamw optimizer

"""
cute things i learnt:
key = jr.key(42) -> k1, k2 = jr.split(key)
jnp.ones(shape)
jr.normal(key, shape)
init linear layer with W = jr.normal(key, (in_dim, out_dim)) * jnp.sqrt(2.0 / in_dim)
# question: what kind of initialization is this?

-> note: 
# params in jax are pytrees (lists, dicts, tuples, custom pytree objects)
# jax.tree.map is useful to apply operation on entire pytree!
# use jax.tree.map(jnp.zeros_like, params) in order to get zeros like a pytree.

jnp.arange
jnp.meshgrid
jnp.indices 

"""



import jax
import jax.random as jr
import jax.numpy as jnp
from matplotlib import pyplot as plt

# 0. lets do this random key shit
# randomness is explicit via PRNG keys
key = jr.key(0)
key, subkey = jr.split(key)
x = jnp.ones((2, 4))

# linear layer, mlp, gelu, rmsnorm, layernorm

def init_linear(key, in_dim, out_dim):
    return {"W": jr.normal(key, (in_dim, out_dim,)) * jnp.sqrt(2.0 / in_dim), "b": jnp.zeros((out_dim,))}

def linear(linear_params, x):
    return linear_params["b"] + x @ linear_params["W"]

print(linear(init_linear(key, x.shape[-1], 7), x).shape, f"expected linear output shape: [{x.shape[0]}, 7]")

def init_mlp(key, in_dim, hidden_dim, out_dim):
    k1, k2 = jr.split(key)
    return {
        "l1": init_linear(k1, in_dim, hidden_dim),
        "l2": init_linear(k2, hidden_dim, out_dim),
    }

def mlp(mlp_params, x):
    return linear(mlp_params["l2"], x=jax.nn.gelu(linear(mlp_params["l1"], x=x)))

print(mlp(init_mlp(key, x.shape[-1], 5, 7), x).shape, f"expected mlp output shape: [{x.shape[0]}, 7]")


def init_rmsnorm(dim):
    return {"scale": jnp.ones((dim,))}

def rmsnorm(rmsnorm_params, x, eps=1e-6):
    return rmsnorm_params["scale"] * x / ( jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps) )


print(f"printing rmsnorm on x: \n{x}")
print()
print(f"rmsnorm: \n{rmsnorm(init_rmsnorm(x.shape[-1]), x)}")
print()


def zeros_like_tree(tree):
    return jax.tree.map(jnp.zeros_like, tree)

def init_adamw_state(params):
    return dict(
        step=jnp.array(0),
        m=zeros_like_tree(params),
        v=zeros_like_tree(params),
    )

def adamw_update(params, grads, opt_state, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=1.0):
    step = opt_state["step"] + 1

    m = jax.tree.map(
        lambda m, g: beta1 * m + (1.0 - beta1) * g,
        opt_state["m"],
        grads
    )

    v = jax.tree.map(
        lambda v, g: beta2 * v + (1.0 - beta2) * g**2,
        opt_state["v"],
        grads
    )

    m_hat = jax.tree.map(lambda m: m / (1.0 - beta1 ** step), m)
    v_hat = jax.tree.map(lambda v: v / (1.0 - beta2 ** step), v)

    new_params = jax.tree.map(
        lambda p, m_hat, v_hat: p - lr * (m_hat / (jnp.sqrt(v_hat) + eps) + wd * p),
        params,
        m_hat,
        v_hat,
    )

    return new_params, {"step": step, "m": m, "v": v}



# okay lets define the grokking problem and then fucking do it
import math

P = 113
key = jr.PRNGKey(0)
permutation = jr.permutation(key, P**2)
test_frac = 0.02
split_index = math.ceil(P**2 * test_frac)
assert split_index not in [0, P**2 - 1] # make sure there is atleast one train and test sample

test_indices, train_indices = permutation[:split_index], permutation[split_index:]
train_epochs = 10000

# defining the grokking data
a_grid, b_grid = jnp.meshgrid(jnp.arange(P), jnp.arange(P), indexing='ij')
a, b = [item.reshape(-1) for item in [a_grid, b_grid]]
c = (a + b) % P
data = jnp.stack([a, b, c], axis=1)
data_train, data_test = data[train_indices, :], data[test_indices, :]
print(data_train[:10], data_test[:10])

# cross entropy loss

def cross_entropy_loss(logits, targets):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(
        log_probs,
        targets[..., None],
        axis=-1
    )
    return -jnp.mean(target_log_probs)

# okay lets define the mlp now
def init_embed(key, vocab_size, embed_dim):
    return dict(embeddings=jr.normal(key, (vocab_size, embed_dim)))
    
def get_input_embeddings(embed_params, x):
    return embed_params["embeddings"][x].reshape(x.shape[0], -1)

from tqdm import tqdm

embed_dim = 128
mlp_in_dim = 256
mlp_hidden_dim = 1024
mlp_out_dim = P

embed_params = init_embed(key, P, embed_dim)
mlp_params = init_mlp(key, mlp_in_dim, mlp_hidden_dim, mlp_out_dim)
params = dict(
    mlp_params=mlp_params,
    embed_params=embed_params
)
opt_state = init_adamw_state(params)

def loss_fn(params, x):
    mlp_params = params["mlp_params"]
    embedding_params = params["embed_params"]

    ab, y = x[:, :-1], x[:, -1]
    input_embeds = get_input_embeddings(embedding_params, ab)
    logits = mlp(mlp_params, input_embeds)

    celoss = cross_entropy_loss(logits, y)

    return celoss

@jax.jit
def accuracy(params, x):
    mlp_params = params["mlp_params"]
    embedding_params = params["embed_params"]

    ab, y = x[:, :-1], x[:, -1]
    input_embeds = get_input_embeddings(embedding_params, ab)
    logits = mlp(mlp_params, input_embeds)
    preds = logits.argmax(axis=-1)
    return jnp.sum(y == preds) / jnp.sum(y == y)


@jax.jit
def train_step(params, opt_state, x):
    loss, grads = jax.value_and_grad(loss_fn)(params, x)
    params, opt_state = adamw_update(params, grads, opt_state)
    return params, opt_state, loss





# okay so I guess that I am now ready to do this shit.
# lets write a simple training run with print logging lmao
# fuck matplotlib lets printit the old fashioned way
for step in range(1000):
    params, opt_state, loss = train_step(params, opt_state, data_train)
    test_acc = accuracy(params, data_test)
    train_acc = accuracy(params, data_train)
    print(f"test accuracy: {test_acc}")
    print(f"train accuracy: {train_acc}")
    print(step, loss)
    print()



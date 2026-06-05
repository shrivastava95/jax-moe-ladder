# lets make a
# 0. proper random initialization with jax
# 1. linear layer
# 2. ffn
# 3. rmsnorm
# 4. layernorm
# 5. adamw optimizer

import jax
import jax.random as jr
import jax.numpy as jnp

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
    return linear(mlp_params["l2"], x=linear(mlp_params["l1"], x=x))

print(mlp(init_mlp(key, x.shape[-1], 5, 7), x).shape, f"expected mlp output shape: [{x.shape[0]}, 7]")


def init_rmsnorm(dim):
    return {"scale": jnp.ones((dim,))}

def rmsnorm(rmsnorm_params, x, eps=1e-6):
    return rmsnorm_params["scale"] * x / ( jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps) )


print(f"printing rmsnorm on x: \n{x}")
print()
print(f"rmsnorm: \n{rmsnorm(init_rmsnorm(x.shape[-1]), x)}")
print()



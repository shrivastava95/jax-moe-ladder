"""
Minimal JAX grokking setup for modular addition.

Builds from scratch:
- Linear layer
- MLP
- Optax AdamW/SGD optimizer
- Cross-entropy loss
- JIT train/eval steps
- Train/test accuracy + loss plot
"""

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from optax import adamw, apply_updates, sgd
from tqdm import tqdm


# -----------------------------
# Config
# -----------------------------
P = 113
TEST_FRAC = 0.2
NUM_EPOCHS = 1000
SEED = 0

EMBED_DIM = 128
MLP_IN_DIM = 2 * EMBED_DIM
MLP_HIDDEN_DIM = 256
MLP_OUT_DIM = P

OPTIMIZER_NAME = "sgd"
ADAMW_LR = 1e-3
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
WEIGHT_DECAY = 1.0
SGD_LR = 3.0
SGD_MOMENTUM = 0.9
PLOT_PATH = Path("jax_modular_addition_training.png")


# -----------------------------
# Layers
# -----------------------------
def init_linear(key, in_dim, out_dim):
    """Kaiming/He-style init for a linear layer."""
    w_key, _ = jr.split(key)
    return {
        "W": jr.normal(w_key, (in_dim, out_dim)) * jnp.sqrt(2.0 / in_dim),
        "b": jnp.zeros((out_dim,)),
    }


def linear(params, x):
    return x @ params["W"] + params["b"]


def init_mlp(key, in_dim, hidden_dim, out_dim):
    k1, k2 = jr.split(key)
    return {
        "l1": init_linear(k1, in_dim, hidden_dim),
        "l2": init_linear(k2, hidden_dim, out_dim),
    }


def mlp(params, x):
    x = linear(params["l1"], x)
    x = jax.nn.gelu(x)
    x = linear(params["l2"], x)
    return x


def init_embed(key, vocab_size, embed_dim):
    return {
        "embeddings": jr.normal(key, (vocab_size, embed_dim)) * 0.02,
    }


def get_input_embeddings(embed_params, x):
    # x: (N, 2), output: (N, 2 * embed_dim)
    return embed_params["embeddings"][x].reshape(x.shape[0], -1)


# -----------------------------
# Optimizer
# -----------------------------
def make_optimizer(name=OPTIMIZER_NAME):
    if name == "adamw":
        return adamw(
            learning_rate=ADAMW_LR,
            b1=ADAM_BETA1,
            b2=ADAM_BETA2,
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
        )
    if name == "sgd":
        return sgd(learning_rate=SGD_LR, momentum=SGD_MOMENTUM)
    raise ValueError(f"unknown optimizer: {name}")


optimizer = make_optimizer()


# -----------------------------
# Data
# -----------------------------
def make_modular_addition_data(key, p=P, test_frac=TEST_FRAC):
    a_grid, b_grid = jnp.meshgrid(jnp.arange(p), jnp.arange(p), indexing="ij")
    a = a_grid.reshape(-1)
    b = b_grid.reshape(-1)
    c = (a + b) % p
    data = jnp.stack([a, b, c], axis=1)

    permutation = jr.permutation(key, p**2)
    split_index = math.ceil(p**2 * test_frac)
    assert 0 < split_index < p**2 - 1

    test_indices = permutation[:split_index]
    train_indices = permutation[split_index:]
    return data[train_indices], data[test_indices]


# -----------------------------
# Model + metrics
# -----------------------------
def init_model(key):
    k_emb, k_mlp = jr.split(key)
    return {
        "embed_params": init_embed(k_emb, P, EMBED_DIM),
        "mlp_params": init_mlp(k_mlp, MLP_IN_DIM, MLP_HIDDEN_DIM, MLP_OUT_DIM),
    }


def forward(params, batch):
    ab = batch[:, :-1]
    input_embeds = get_input_embeddings(params["embed_params"], ab)
    logits = mlp(params["mlp_params"], input_embeds)
    return logits


def cross_entropy_loss(logits, targets):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, targets[:, None], axis=-1)
    return -jnp.mean(target_log_probs)


def loss_fn(params, batch):
    y = batch[:, -1]
    logits = forward(params, batch)
    return cross_entropy_loss(logits, y)


@jax.jit
def accuracy(params, batch):
    y = batch[:, -1]
    logits = forward(params, batch)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)


@jax.jit
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = apply_updates(params, updates)
    return params, opt_state, loss


# -----------------------------
# Plotting
# -----------------------------
def save_training_plot(steps, train_losses, test_losses, train_accs, test_accs, path=PLOT_PATH):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(steps, train_losses, label="train loss")
    ax1.plot(steps, test_losses, label="test loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")

    ax2 = ax1.twinx()
    ax2.plot(steps, train_accs, linestyle="--", label="train accuracy")
    ax2.plot(steps, test_accs, linestyle="--", label="test accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.0, 1.0)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

    ax1.set_title("JAX modular addition training")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    key = jr.key(SEED)
    k_data, k_model = jr.split(key)

    data_train, data_test = make_modular_addition_data(k_data)
    params = init_model(k_model)
    opt_state = optimizer.init(params)

    steps = []
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in tqdm(range(NUM_EPOCHS), desc="training"):
        params, opt_state, train_loss = train_step(params, opt_state, data_train)

        # Full-dataset metrics. This is fine here because P^2 is small.
        test_loss = loss_fn(params, data_test)
        train_acc = accuracy(params, data_train)
        test_acc = accuracy(params, data_test)

        steps.append(epoch)
        train_losses.append(float(train_loss))
        test_losses.append(float(test_loss))
        train_accs.append(float(train_acc))
        test_accs.append(float(test_acc))

        if epoch % 100 == 0 or epoch == NUM_EPOCHS - 1:
            print(
                f"epoch={epoch:04d} "
                f"train_loss={train_losses[-1]:.4f} "
                f"test_loss={test_losses[-1]:.4f} "
                f"train_acc={train_accs[-1]:.4f} "
                f"test_acc={test_accs[-1]:.4f}"
            )

    save_training_plot(steps, train_losses, test_losses, train_accs, test_accs)
    print(f"saved plot to: {PLOT_PATH.resolve()}")


if __name__ == "__main__":
    main()

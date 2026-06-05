Yes, do it in **JAX**, but not “just because.”

Do it in JAX because your target is not merely “MoE layer works.” Your target is:

```text
MoE layer works
→ training works
→ routing diagnostics make sense
→ expert parallelism becomes expressible
→ distributed sharding becomes legible
```

JAX is a good fit because its core model forces you to think in transformations like `jit`, `grad`, `vmap`, and explicit array programs; its modern distributed path uses `jax.Array`, `Mesh`, `NamedSharding`, and `PartitionSpec`; and `shard_map` gives explicit per-device control while composing with `jit` and `grad`. ([JAX Documentation][1]) Also: don’t make `pmap` your main mental model. JAX docs now describe `pmap` as implemented in terms of `jit` and `shard_map`, and suggest using `shard_map`/related APIs for newer multi-device work. ([JAX Documentation][2])

## Curriculum principle

For each concept, your implementation test should have three parts:

```text
1. Canonical implementation
2. Invariant tests
3. Mutation test
```

Meaning:

```text
Canonical implementation:
Can I build the standard thing?

Invariant tests:
Can I prove the shapes, probabilities, losses, and gradients are sane?

Mutation test:
Can I change the design and predict what breaks?
```

That is how you make the concept malleable instead of memorized.

---

# Curriculum: implement these in order

## Stage 0 — JAX substrate

### Implement

Write from scratch:

```text
1. Linear layer
2. MLP layer
3. LayerNorm or RMSNorm
4. AdamW / simple optimizer step
5. Cross-entropy next-token loss
6. jit-compiled train step
```

Use only:

```python
jax
jax.numpy as jnp
jax.random
jax.grad / jax.value_and_grad
jax.jit
```

Avoid Flax initially. You need to feel the tensors directly.

### Pass criteria

You pass when you can answer and test:

```text
Given params and batch,
loss(params, batch) returns scalar.

grad(loss)(params, batch) has same tree structure as params.

One train_step updates params without Python-side mutation.
```

### Mutation test

Change:

```text
MLP activation: GELU → SwiGLU
optimizer: SGD → AdamW
sequence length: 64 → 256
vocab size: 128 → 4096
```

You should be able to predict which dimensions, memory terms, and compute terms change.

---

## Stage 1 — Dense next-token model

### Implement

A tiny dense Transformer LM:

```text
token embedding
+ positional encoding or RoPE
+ causal self-attention
+ dense FFN
+ LM head
+ next-token cross-entropy
```

Small is fine:

```text
vocab_size = 256 or byte-level
d_model = 128
n_layers = 2
n_heads = 4
seq_len = 128
```

Do not optimize yet.

### Pass criteria

You pass when:

```text
1. Loss decreases on a tiny corpus.
2. Model can overfit a tiny repeated text sample.
3. Shapes are clear at every block.
4. You can isolate attention loss vs FFN loss by ablation.
```

Concrete tests:

```text
Overfit 1 KB text.
Train until loss drops sharply.
Generate samples.
Verify causal mask prevents future leakage.
```

### Mutation test

Remove the FFN sublayer and observe degradation.

Why this matters:

```text
MoE usually replaces the FFN. 
So you need to know what the dense FFN was contributing before multiplying it.
```

---

## Stage 2 — Expert FFN bank

### Concept being tested

```text
One FFN → many independent FFNs
```

### Implement

Before routing, build an expert bank:

```text
experts: [E] independent FFNs
input:   x [T, d_model]
output:  expert_outputs [T, E, d_model]
```

This is dense-over-experts, not sparse yet.

### Pass criteria

You pass when:

```text
1. Each expert has independent parameters.
2. Expert output shape is [T, E, d_model].
3. Gradients flow into all experts.
4. If all experts have same params, outputs match.
```

### Mutation test

Change:

```text
number of experts E
expert hidden dimension
activation function
shared first layer vs fully independent experts
```

You should be able to predict:

```text
parameter count
active compute
memory use
gradient distribution
```

---

## Stage 3 — Router logits and routing scores

### Concept being tested

```text
Token representation → scores over experts
```

### Implement

Router:

```text
x:             [T, d_model]
router_W:      [d_model, E]
router_logits: [T, E]
router_probs:  [T, E]
```

### Pass criteria

You pass when:

```text
1. router_probs.sum(axis=-1) ≈ 1 for each token.
2. router_logits[t, e] has a clear meaning.
3. Gradients flow into router_W if routing is soft.
```

### Mutation test

Try:

```text
softmax temperature
router noise
router bias
router initialized to favor one expert
```

You should predict:

```text
higher temperature → more uniform routing
lower temperature → sharper routing
biased router → load imbalance
```

---

## Stage 4 — Soft MoE before hard top-k

### Why this stage exists

Do **not** jump directly to top-k. First implement a fully differentiable “soft expert mixture”:

```text
y_t = Σ_e p(t,e) * expert_e(x_t)
```

This is not efficient MoE. It is a conceptual bridge.

### Implement

```text
expert_outputs: [T, E, d_model]
router_probs:   [T, E]
y:              [T, d_model]
```

Using:

```python
y = jnp.einsum("te,ted->td", router_probs, expert_outputs)
```

### Pass criteria

You pass when:

```text
1. Output shape equals dense FFN output shape.
2. Gradients flow into all experts and router.
3. If router gives one-hot weights, output equals selected expert.
```

### Mutation test

Force router_probs to:

```text
uniform
one-hot
biased to expert 0
random sparse
```

Then predict outputs and gradient distribution.

This is where the mental object becomes malleable.

---

## Stage 5 — Top-k routing

### Concept being tested

```text
Soft mixture → sparse mixture
```

### Implement

For each token:

```text
top_k_expert_ids: [T, k]
top_k_logits:     [T, k]
top_k_weights:    [T, k]
```

Then compute:

```text
y_t = Σ_{i=1..k} weight_i * expert_{id_i}(x_t)
```

Initially, you may still compute all experts and gather selected outputs. That is inefficient but conceptually clean.

### Pass criteria

You pass when:

```text
1. k=1 selects exactly one expert.
2. k=2 combines two experts.
3. top_k_weights.sum(axis=-1) ≈ 1.
4. Changing k changes active expert count.
5. Non-selected experts receive no gradient from main loss.
```

### Mutation test

Try:

```text
k = 1
k = 2
k = E
temperature changes
router noise
expert bias
```

You should predict:

```text
k=E recovers soft dense mixture
k=1 is sparse but harsher
k=2 gives smoother training than k=1
```

---

## Stage 6 — Dispatch/combine, local version

This is the first serious MoE implementation test.

### Concept being tested

```text
Do not run every expert for every token.
Group selected tokens by expert.
```

### Implement

Input:

```text
x:          [T, d_model]
expert_ids: [T, k]
weights:    [T, k]
```

Produce:

```text
expert_batches:      [E, capacity, d_model]
expert_token_indices:[E, capacity]
expert_slot_mask:    [E, capacity]
```

Then:

```text
expert_outputs: [E, capacity, d_model]
```

Then scatter/combine back:

```text
y: [T, d_model]
```

### Pass criteria

You pass when:

```text
1. Local dispatch output matches naive top-k implementation.
2. Token order is restored exactly.
3. Empty experts do not break.
4. Duplicate routing is handled or explicitly forbidden.
5. Gradients match the naive reference implementation within tolerance.
```

This is the most important stage.

### Mutation test

Change the internal representation:

```text
[E, capacity, d_model]
vs flattened [total_assignments, d_model]
vs padded dense dispatch mask [T, E, capacity]
```

If you can switch representations, you understand dispatch.

---

## Stage 7 — Capacity factor and overflow

### Concept being tested

```text
Routing creates uneven expert loads.
```

### Implement

For:

```text
T tokens
k selected experts per token
E experts
capacity_factor c
```

compute:

```text
capacity = ceil(c * T * k / E)
```

Then implement overflow policy:

```text
tokens beyond expert capacity are dropped
```

or:

```text
fall back to residual/dense path
```

Start with dropping.

### Pass criteria

You pass when you can log:

```text
tokens_per_expert
capacity
overflow_count
overflow_fraction
dropped_tokens
expert_utilization
```

### Mutation test

Sweep:

```text
capacity_factor = 0.5, 1.0, 1.25, 2.0
E = 4, 8, 16, 64
k = 1, 2
batch size / sequence length
router temperature
```

You should predict:

```text
more experts → lower expected tokens per expert
lower capacity factor → more overflow
sharper router → more imbalance
larger batch → smoother load estimates
```

---

## Stage 8 — Load-balancing loss

### Concept being tested

```text
The router must be trained not only for prediction,
but also for usable expert allocation.
```

### Implement diagnostics first:

```text
importance[e] = sum_t router_probs[t, e]
load[e]       = count_t expert e selected by top-k
```

Then implement an auxiliary loss that penalizes uneven usage.

### Pass criteria

You pass when:

```text
1. Without balancing loss, expert usage can collapse.
2. With balancing loss, usage becomes more uniform.
3. Main NTP loss and balancing loss are logged separately.
4. You can increase/decrease aux coefficient and predict behavior.
```

### Mutation test

Try:

```text
aux_loss_weight = 0
aux_loss_weight = small
aux_loss_weight = too large
```

Prediction:

```text
0       → collapse risk
small   → better utilization
too big → router optimizes uniformity over task usefulness
```

This is a critical causal tradeoff.

---

## Stage 9 — Replace dense FFN with MoE FFN inside Transformer

### Concept being tested

```text
MoE is not a standalone trick.
It is a drop-in sparse FFN replacement.
```

### Implement

Take your dense Transformer:

```text
Attention
Dense FFN
```

Replace some FFNs:

```text
Attention
MoE FFN
```

Variants:

```text
all layers MoE
every other layer MoE
only middle layer MoE
only last layer MoE
```

### Pass criteria

You pass when:

```text
1. Model trains on NTP.
2. Loss decreases.
3. Generation works.
4. Expert usage is logged per layer.
5. Overflow is logged per layer.
6. You can compare dense vs MoE at similar active FLOPs.
```

### Mutation test

Hold active compute roughly fixed and vary:

```text
number of experts
expert width
k
MoE layer frequency
capacity factor
aux loss weight
```

This is where MoE becomes an experimental object.

---

## Stage 10 — MoE experiment harness

### Concept being tested

```text
You cannot claim understanding if you cannot measure failure.
```

### Implement logs:

```text
train loss
eval loss
perplexity
tokens/sec
expert usage histogram
router entropy
overflow fraction
aux loss
z-loss if used
gradient norms
expert parameter norms
router parameter norms
```

### Pass criteria

You pass when you can identify these failure modes from logs:

```text
router collapse
expert starvation
overflow too high
aux loss too strong
training unstable
experts not specializing
MoE slower than dense
```

### Mutation test

Create failure intentionally:

```text
router bias to expert 0
capacity_factor too small
aux_loss_weight = 0
learning rate too high
k = 1 with sharp router
```

Then fix each one.

This is exactly how to make the system malleable.

---

## Stage 11 — Simulated multi-device JAX

Before renting GPUs, use fake CPU devices.

JAX supports configuring multiple CPU devices via `JAX_NUM_CPU_DEVICES` or the XLA flag `--xla_force_host_platform_device_count`, and the `shard_map` tutorial uses this to simulate 8 CPU devices. ([JAX Documentation][3])

### Implement

```text
1. Create a Mesh.
2. Shard a batch over a data axis.
3. Run data-parallel dense training.
4. Use lax.psum for gradient averaging.
5. Verify results match single-device training.
```

Use:

```python
Mesh
PartitionSpec
NamedSharding
jax.device_put
jax.shard_map
jax.lax.psum
```

### Pass criteria

You pass when:

```text
1. You can explain global shape vs local shard shape.
2. You can print per-device shard shapes.
3. You can average gradients across devices.
4. You can recover same-ish loss curve as single-device.
```

---

## Stage 12 — Expert parallelism, simulated

### Concept being tested

```text
Experts live on different devices.
Tokens must move to expert devices.
```

### Implement

Toy setup:

```text
4 devices
8 experts
2 experts per device
```

Input tokens start sharded by batch:

```text
device 0 has tokens 0..T/4
device 1 has tokens T/4..T/2
...
```

But experts are sharded by expert id:

```text
device 0 owns experts 0,1
device 1 owns experts 2,3
...
```

Now implement:

```text
local routing
cross-device dispatch
local expert execution
cross-device combine
```

At first, you can use crude explicit collectives. Performance is not the first goal.

### Pass criteria

You pass when:

```text
1. Distributed output matches local reference output.
2. Each expert runs only on its owning device.
3. Tokens routed to remote experts are handled correctly.
4. Combine restores original token order.
5. You can log communication volume proxy.
```

### Mutation test

Change:

```text
experts per device
top-k
token sharding axis
expert sharding axis
capacity factor
```

You should predict communication patterns.

---

## Stage 13 — Real 2-GPU run

Now rent the cheap 2×3090 instance.

### Implement

Take your simulated expert-parallel MoE and run it on real GPUs.

### Pass criteria

You pass when:

```text
1. Same code path works on 2 GPUs.
2. You can profile tokens/sec.
3. You can compare:
   - dense baseline
   - local MoE
   - expert-parallel MoE
4. You can identify whether bottleneck is compute, memory, or communication.
```

### Mutation test

Sweep:

```text
E = 2, 4, 8
k = 1, 2
capacity_factor = 1.0, 1.25, 2.0
seq_len
batch size
```

Your question at this stage becomes:

```text
When does MoE become slower despite lower active compute?
```

That is the real systems question.

---

# The complete implementation ladder

```text
0. Raw JAX substrate
1. Dense NTP model
2. Dense Transformer LM
3. Expert FFN bank
4. Router logits
5. Soft expert mixture
6. Top-k sparse routing
7. Local dispatch/combine
8. Capacity + overflow
9. Load-balancing loss
10. MoE Transformer LM
11. MoE diagnostics harness
12. Simulated data parallelism
13. Simulated expert parallelism
14. Real 2-GPU expert-parallel MoE
15. Profile and compare dense vs MoE
```

## What not to do yet

Do not start with:

```text
DeepSpeed-MoE
Megatron
Tutel
custom CUDA kernels
Pallas
H100/A100 instances
```

Those are implementation accelerants. They will hide the mechanism you’re trying to acquire.

Also do not start with Flax/linen/NNX unless raw JAX becomes too annoying. Framework abstractions are useful after the graph is in your head, not before.

## Minimum viable project

Your first serious milestone should be:

```text
A tiny byte-level Transformer LM in JAX
with one MoE FFN layer,
top-2 routing,
capacity factor,
load-balancing loss,
and diagnostic logs,
trained on a small text corpus.
```

That is enough to prove you understand the architecture.

Then the second milestone is:

```text
Same MoE layer,
but expert-parallel across 2 devices,
with outputs matching the single-device reference.
```

That is enough to prove you understand the systems version.

[1]: https://docs.jax.dev/en/latest/quickstart.html?utm_source=chatgpt.com "Quickstart: How to think in JAX"
[2]: https://docs.jax.dev/en/latest/_autosummary/jax.pmap.html?utm_source=chatgpt.com "jax.pmap"
[3]: https://docs.jax.dev/en/latest/config_options.html?utm_source=chatgpt.com "Configuration Options"


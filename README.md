# jax-moe-ladder
A tiny byte-level Transformer LM in JAX with one MoE FFN layer, top-2 routing, capacity factor, load-balancing loss, and diagnostic logs, trained on a small text corpus. Expert-parallel across 2 devices, with outputs matching the single-device reference.

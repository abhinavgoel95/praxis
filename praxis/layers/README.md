The changes to `attentions.py` in this branch showcases an example on how to use TE's custom flash attention kernel in a JAX framework.


The TE [self_fused_attn](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/jax/fused_attn.py#L53) API expects a concatenated QKV vector of shape `(b, s, 3, h, d)` where `b` is batch size, `s` is seq len, `h` is #heads, and `d` is head_dim.
The following snippet shows how to concatenate the Q, K, and V tensors.
```
b, s_q, h, d = query_proj.shape
_, s_kv, _, _ = key_proj.shape

q = jnp.reshape(query_proj, (*query_proj.shape[:2], 1, *query_proj.shape[-2:]))
k = jnp.reshape(key_proj, (*query_proj.shape[:2], 1, *query_proj.shape[-2:]))
v = jnp.reshape(value_proj, (*query_proj.shape[:2], 1, *query_proj.shape[-2:]))
qkv = jnp.concatenate((q, k, v), axis=2) # to make it (b, s, 3, h, d)
```

The following snippet is a call to the `self_fused_attn` function inside TE. This will get lowered to the cuDNN flash attention kernel for both forward and backward passes.

```
encoded = fused_attn.self_fused_attn(
        qkv=qkv,
        bias=None,
        mask=jnp.zeros((b, 1, s_q, s_kv)),  # no padding
        seed=None,
        attn_bias_type=fused_attn.AttnBiasType.NO_BIAS,
        attn_mask_type=fused_attn.AttnMaskType.CAUSAL_MASK,
        scaling_factor=1.0/math.sqrt(h),
        dropout_probability=0.,
        is_training=True)
```

The TE API only returns the output of the flash attention kernel (called `encoded` above), it does not return the attention probabilities. Please keep this in mind when using the API.

# utils.py
import re
import torch

def _strip_prefix(state, prefix='module.'):
    return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state.items() }

# ----------------------------
# TARGET ENCODER (backbone.*)
# ----------------------------
def map_target_encoder_state(raw_state: dict, strip_prefix: str = 'module.'):
    """
    Maps a V-JEPA2 'target_encoder' PyTorch state dict into HF-style encoder.* keys.

    Returns:
        new_state (dict): mapped tensors
        report (dict):   diagnostics (leftovers, block indices, sanity checks)
    """
    state = _strip_prefix(raw_state, strip_prefix)

    def map_patch_embed(name):
        tail = name.split('patch_embed.proj.', 1)[1]
        return f'encoder.embeddings.patch_embeddings.proj.{tail}'

    def map_block_base(bi):
        return f'encoder.layer.{bi}'

    def map_norm(name, bi, which):
        tail = name.split(f'blocks.{bi}.norm{which}.', 1)[1]
        return f'{map_block_base(bi)}.norm{which}.{tail}'

    def map_mlp(name, bi, which):
        tail = name.split(f'blocks.{bi}.mlp.fc{which}.', 1)[1]
        return f'{map_block_base(bi)}.mlp.fc{which}.{tail}'

    def map_attn_proj(name, bi):
        tail = name.split(f'blocks.{bi}.attn.proj.', 1)[1]
        return f'{map_block_base(bi)}.attention.proj.{tail}'

    def map_final_norm(name):
        tail = name.split('backbone.norm.', 1)[1]
        return f'encoder.layernorm.{tail}'

    new_state = {}
    consumed = set()

    # 1) Patch embedding
    for wb in ('weight', 'bias'):
        k = f'backbone.patch_embed.proj.{wb}'
        if k in state:
            new_state[map_patch_embed(k)] = state[k]
            consumed.add(k)

    # 2) Blocks
    block_idxs = sorted({
        int(m.group(1))
        for m in (re.search(r'backbone\.blocks\.(\d+)\.', k) for k in state.keys())
        if m
    })
    for bi in block_idxs:
        # norms
        for which in (1, 2):
            for wb in ('weight', 'bias'):
                k = f'backbone.blocks.{bi}.norm{which}.{wb}'
                if k in state:
                    new_state[map_norm(k, bi, which)] = state[k]
                    consumed.add(k)
        # mlp
        for which in (1, 2):
            for wb in ('weight', 'bias'):
                k = f'backbone.blocks.{bi}.mlp.fc{which}.{wb}'
                if k in state:
                    new_state[map_mlp(k, bi, which)] = state[k]
                    consumed.add(k)
        # attn proj
        for wb in ('weight', 'bias'):
            k = f'backbone.blocks.{bi}.attn.proj.{wb}'
            if k in state:
                new_state[map_attn_proj(k, bi)] = state[k]
                consumed.add(k)
        # qkv split
        k_w = f'backbone.blocks.{bi}.attn.qkv.weight'
        k_b = f'backbone.blocks.{bi}.attn.qkv.bias'
        if k_w in state:
            W = state[k_w]
            assert W.dim() == 2 and W.size(0) % 3 == 0, f"Unexpected qkv.weight @ block {bi}: {tuple(W.shape)}"
            D3, _ = W.shape
            D = D3 // 3
            base = map_block_base(bi)
            new_state[f'{base}.attention.query.weight'] = W[:D, :].contiguous()
            new_state[f'{base}.attention.key.weight']   = W[D:2*D, :].contiguous()
            new_state[f'{base}.attention.value.weight'] = W[2*D:, :].contiguous()
            consumed.add(k_w)
        if k_b in state:
            b = state[k_b]
            assert b.dim() == 1 and b.size(0) % 3 == 0, f"Unexpected qkv.bias @ block {bi}: {tuple(b.shape)}"
            D3 = b.size(0)
            D = D3 // 3
            base = map_block_base(bi)
            new_state[f'{base}.attention.query.bias'] = b[:D].contiguous()
            new_state[f'{base}.attention.key.bias']   = b[D:2*D].contiguous()
            new_state[f'{base}.attention.value.bias'] = b[2*D:].contiguous()
            consumed.add(k_b)

    # 3) Final norm
    for wb in ('weight', 'bias'):
        k = f'backbone.norm.{wb}'
        if k in state:
            new_state[map_final_norm(k)] = state[k]
            consumed.add(k)

    leftovers = sorted(k for k in state.keys() if k not in consumed)

    # Sanity: probe first/last block
    def collect_missing_for_block(bi):
        base = f'encoder.layer.{bi}'
        must = [
            f'{base}.norm1.weight', f'{base}.norm1.bias',
            f'{base}.norm2.weight', f'{base}.norm2.bias',
            f'{base}.attention.query.weight', f'{base}.attention.query.bias',
            f'{base}.attention.key.weight',   f'{base}.attention.key.bias',
            f'{base}.attention.value.weight', f'{base}.attention.value.bias',
            f'{base}.attention.proj.weight',  f'{base}.attention.proj.bias',
            f'{base}.mlp.fc1.weight', f'{base}.mlp.fc1.bias',
            f'{base}.mlp.fc2.weight', f'{base}.mlp.fc2.bias',
        ]
        return [m for m in must if m not in new_state]

    sanity = {}
    if block_idxs:
        for probe in (min(block_idxs), max(block_idxs)):
            sanity[str(probe)] = collect_missing_for_block(probe)

    report = {
        'which': 'target_encoder',
        'blocks': block_idxs,
        'leftovers': leftovers,
        'sanity': sanity,
    }
    return new_state, report

# ----------------------------
# PREDICTOR (predictor_* & mask_tokens)
# ----------------------------
def map_predictor_state(raw_state: dict, strip_prefix: str = 'module.'):
    """
    Maps a V-JEPA2 'predictor' PyTorch state dict into HF-style predictor.* keys.

    Returns:
        new_state (dict): mapped tensors
        report (dict):   diagnostics (leftovers, block indices, sanity checks)
    """
    state = _strip_prefix(raw_state, strip_prefix)

    def map_pred_block_base(bi: int):
        return f'predictor.layer.{bi}'

    def map_pred_norm(name, bi, which):
        tail = name.split(f'predictor_blocks.{bi}.norm{which}.', 1)[1]
        return f'{map_pred_block_base(bi)}.norm{which}.{tail}'

    def map_pred_mlp(name, bi, which):
        tail = name.split(f'predictor_blocks.{bi}.mlp.fc{which}.', 1)[1]
        return f'{map_pred_block_base(bi)}.mlp.fc{which}.{tail}'

    def map_pred_attn_proj(name, bi):
        tail = name.split(f'predictor_blocks.{bi}.attn.proj.', 1)[1]
        return f'{map_pred_block_base(bi)}.attention.proj.{tail}'

    new_state = {}
    consumed = set()

    # 1) predictor embeddings
    for wb in ('weight', 'bias'):
        k = f'backbone.predictor_embed.{wb}'
        if k in state:
            new_state[f'predictor.embeddings.predictor_embeddings.{wb}'] = state[k]
            consumed.add(k)

    # 2) mask tokens (stack)
    mask_token_entries = []
    pat = re.compile(r'^backbone\.mask_tokens\.(\d+)$')
    for k in state.keys():
        m = pat.match(k)
        if m:
            mask_token_entries.append((int(m.group(1)), k))
    mask_token_entries.sort(key=lambda x: x[0])
    if mask_token_entries:
        stacked = torch.stack([state[k] for _, k in mask_token_entries], dim=0).contiguous()
        new_state['predictor.embeddings.mask_tokens'] = stacked
        for _, k in mask_token_entries:
            consumed.add(k)

    # 3) blocks
    block_idxs = sorted({
        int(m.group(1))
        for m in (re.search(r'backbone\.predictor_blocks\.(\d+)\.', k) for k in state.keys())
        if m
    })
    for bi in block_idxs:
        # norms
        for which in (1, 2):
            for wb in ('weight', 'bias'):
                k = f'backbone.predictor_blocks.{bi}.norm{which}.{wb}'
                if k in state:
                    new_state[map_pred_norm(k, bi, which)] = state[k]
                    consumed.add(k)
        # mlp
        for which in (1, 2):
            for wb in ('weight', 'bias'):
                k = f'backbone.predictor_blocks.{bi}.mlp.fc{which}.{wb}'
                if k in state:
                    new_state[map_pred_mlp(k, bi, which)] = state[k]
                    consumed.add(k)
        # attn proj
        for wb in ('weight', 'bias'):
            k = f'backbone.predictor_blocks.{bi}.attn.proj.{wb}'
            if k in state:
                new_state[map_pred_attn_proj(k, bi)] = state[k]
                consumed.add(k)
        # qkv split
        k_w = f'backbone.predictor_blocks.{bi}.attn.qkv.weight'
        k_b = f'backbone.predictor_blocks.{bi}.attn.qkv.bias'
        if k_w in state:
            W = state[k_w]
            assert W.dim() == 2 and W.size(0) % 3 == 0, f"Unexpected qkv.weight @ predictor block {bi}: {tuple(W.shape)}"
            D3, _ = W.shape
            D = D3 // 3
            base = map_pred_block_base(bi)
            new_state[f'{base}.attention.query.weight'] = W[:D, :].contiguous()
            new_state[f'{base}.attention.key.weight']   = W[D:2*D, :].contiguous()
            new_state[f'{base}.attention.value.weight'] = W[2*D:, :].contiguous()
            consumed.add(k_w)
        if k_b in state:
            b = state[k_b]
            assert b.dim() == 1 and b.size(0) % 3 == 0, f"Unexpected qkv.bias @ predictor block {bi}: {tuple(b.shape)}"
            D3 = b.size(0)
            D = D3 // 3
            base = map_pred_block_base(bi)
            new_state[f'{base}.attention.query.bias'] = b[:D].contiguous()
            new_state[f'{base}.attention.key.bias']   = b[D:2*D].contiguous()
            new_state[f'{base}.attention.value.bias'] = b[2*D:].contiguous()
            consumed.add(k_b)

    # 4) final norm + proj
    for wb in ('weight', 'bias'):
        k = f'backbone.predictor_norm.{wb}'
        if k in state:
            new_state[f'predictor.layernorm.{wb}'] = state[k]
            consumed.add(k)
        k = f'backbone.predictor_proj.{wb}'
        if k in state:
            new_state[f'predictor.proj.{wb}'] = state[k]
            consumed.add(k)

    leftovers = sorted(k for k in state.keys() if k not in consumed)

    # Sanity: probe first/last block
    def collect_missing_for_pred_block(bi):
        base = f'predictor.layer.{bi}'
        must = [
            f'{base}.norm1.weight', f'{base}.norm1.bias',
            f'{base}.norm2.weight', f'{base}.norm2.bias',
            f'{base}.attention.query.weight', f'{base}.attention.query.bias',
            f'{base}.attention.key.weight',   f'{base}.attention.key.bias',
            f'{base}.attention.value.weight', f'{base}.attention.value.bias',
            f'{base}.attention.proj.weight',  f'{base}.attention.proj.bias',
            f'{base}.mlp.fc1.weight', f'{base}.mlp.fc1.bias',
            f'{base}.mlp.fc2.weight', f'{base}.mlp.fc2.bias',
        ]
        return [m for m in must if m not in new_state]

    sanity = {}
    if block_idxs:
        for probe in (min(block_idxs), max(block_idxs)):
            sanity[str(probe)] = collect_missing_for_pred_block(probe)

    # Embeddings sanity
    emb_missing = []
    for wb in ('weight', 'bias'):
        if f'predictor.embeddings.predictor_embeddings.{wb}' not in new_state:
            emb_missing.append(f'predictor.embeddings.predictor_embeddings.{wb}')
    mask_present = 'predictor.embeddings.mask_tokens' in new_state

    report = {
        'which': 'predictor',
        'blocks': block_idxs,
        'leftovers': leftovers,
        'sanity': sanity,
        'embeddings_missing': emb_missing,
        'mask_tokens_present': mask_present,
    }
    return new_state, report

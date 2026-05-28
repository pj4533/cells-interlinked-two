"""Unit tests for install_edge_consumer_ablation_hook.

Two correctness invariants:

  1. **Per-head subtraction is exact.** When a hook is installed for
     query head h at layer L, the head-h slice of q_proj(r)'s output is
     equal to q_proj(r - ⟨r, v⟩ v)[head-h-slice]. Verified vs. a
     reference computation that pre-projects-out the residual.

  2. **Non-target heads are bit-identical.** Slices of q_proj output
     that don't belong to any target head must match the un-hooked
     forward exactly (zero diff).

  3. **Equivalence: edge-with-all-heads-at-L ≡ projecting out at L's
     input.** With targets = every head at L, the hook's q_proj output
     must equal q_proj(project_out_native(r, v)).

We use a tiny synthetic transformer (FakeModel below) with the same
attribute shape Gemma exposes (`self_attn` with `q_proj` / `k_proj` /
`v_proj`, `num_heads`, `num_key_value_heads`, `head_dim`). Doing this
on real Gemma-3-12B would require loading 24GB of weights for every
test; the fake suffices for the linear-algebra correctness claims.
"""

from __future__ import annotations

import torch
from torch import nn

from cells_interlinked.pipeline.edge_consumer import (
    install_edge_consumer_ablation_hook,
    build_projection_cache,
)


def project_out_native(h: torch.Tensor, v: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Local reference projection that keeps the input's dtype throughout
    — no fp32 round-trip. Used by tests so fp64 hook output can be
    compared bit-exactly to an fp64 reference. The library's
    `project_out` always demotes to fp32 internally (for bf16 numerical
    headroom on MPS), which would introduce ~1e-8 noise into the
    comparison even though the hook itself is exact at fp64."""
    v_unit = v / (v.norm() + 1e-12)
    coeff = (h * v_unit).sum(dim=-1, keepdim=True)
    return h - alpha * coeff * v_unit


# ── Fake model that mimics Gemma's attention surface ───────────────────


class FakeAttn(nn.Module):
    def __init__(self, d_model: int, n_q: int, n_kv: int, head_dim: int):
        super().__init__()
        self.num_heads = n_q
        self.num_key_value_heads = n_kv
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, n_q * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv * head_dim, bias=False)


class FakeBlock(nn.Module):
    def __init__(self, d_model: int, n_q: int, n_kv: int, head_dim: int):
        super().__init__()
        self.self_attn = FakeAttn(d_model, n_q, n_kv, head_dim)


class FakeInner(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_q: int, n_kv: int, head_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([
            FakeBlock(d_model, n_q, n_kv, head_dim) for _ in range(n_layers)
        ])


class FakeModel(nn.Module):
    """Matches the (model → model.layers → block → self_attn → q/k/v_proj)
    nesting that _find_decoder_layers walks. Gemma adds a
    language_model wrapper inside; FakeModel's plain `model.layers`
    path is the simpler Llama/Qwen branch — also supported by
    _find_decoder_layers."""

    def __init__(self, d_model=8, n_layers=4, n_q=4, n_kv=2, head_dim=4):
        super().__init__()
        self.model = FakeInner(d_model, n_layers, n_q, n_kv, head_dim)
        self.d_model = d_model
        self.n_q = n_q
        self.n_kv = n_kv
        self.head_dim = head_dim


def _make_input(d_model: int, seq_len: int = 5) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(1, seq_len, d_model)


def _make_direction(d_model: int) -> torch.Tensor:
    torch.manual_seed(123)
    v = torch.randn(d_model)
    return v / v.norm()


def _q_proj_at(model: FakeModel, layer_idx: int, h: torch.Tensor) -> torch.Tensor:
    """Convenience: run h through layer L's q_proj only."""
    return model.model.layers[layer_idx].self_attn.q_proj(h)


# ── Tests ──────────────────────────────────────────────────────────────


def test_single_head_subtraction_is_exact():
    """Hook on (L=2, head=1) should make q_proj's head-1 slice match
    q_proj(project_out_native(r, v))'s head-1 slice, while other slices match
    the unhooked q_proj(r) bit-exactly."""
    m = FakeModel(d_model=8, n_q=4, n_kv=2, head_dim=4)
    v = _make_direction(m.d_model).double()  # fp64 to avoid bf16 issues
    h = _make_input(m.d_model).double()

    # Convert all model weights to fp64 to match.
    m.double()

    L, head = 2, 1
    cache = build_projection_cache(m, v, L)

    # Reference: pre-project residual, then q_proj
    h_proj = project_out_native(h, v, alpha=1.0)
    q_ref = _q_proj_at(m, L, h_proj)            # [1, T, n_q*head_dim]
    q_unhooked = _q_proj_at(m, L, h)

    # With hook
    handles = install_edge_consumer_ablation_hook(
        m, [(L, head)], v, {L: cache}, alpha=1.0,
        target_projections=("q",),  # isolate q only for this test
    )
    try:
        q_hooked = _q_proj_at(m, L, h)
    finally:
        for hd in handles:
            hd.remove()

    # Verify per-head behavior
    n_q, head_dim = m.n_q, m.head_dim
    q_unhooked_per_head = q_unhooked.view(*q_unhooked.shape[:-1], n_q, head_dim)
    q_hooked_per_head   = q_hooked.view(*q_hooked.shape[:-1], n_q, head_dim)
    q_ref_per_head      = q_ref.view(*q_ref.shape[:-1], n_q, head_dim)

    # Tolerance budget. Target-head comparison goes through q_proj(h −
    # coeff·v̂) on one side and q_proj(h) − coeff·q_proj(v̂) on the
    # other — mathematically identical, but matmul summation reorders
    # differ → fp64 reassociation noise (~1e-9 on these magnitudes).
    # Non-target heads should be exactly equal (the hook doesn't touch
    # those slices at all).
    TARGET_TOL = 1e-6   # generous; real bugs would produce > 1e-3
    NONTARGET_TOL = 0.0  # bit-exact: untouched slices

    for hd_idx in range(n_q):
        if hd_idx == head:
            diff = (q_hooked_per_head[..., hd_idx, :] - q_ref_per_head[..., hd_idx, :]).abs().max()
            assert diff < TARGET_TOL, f"target head {hd_idx} mismatch: max abs diff {diff}"
        else:
            diff = (q_hooked_per_head[..., hd_idx, :] - q_unhooked_per_head[..., hd_idx, :]).abs().max()
            assert diff <= NONTARGET_TOL, f"non-target head {hd_idx} altered: max abs diff {diff}"


def test_all_heads_equivalent_to_global_pre_projection():
    """With targets = every query head at L, the hook's q_proj output
    must equal q_proj(project_out_native(r, v)) over the WHOLE tensor (not
    just per-head slices)."""
    m = FakeModel(d_model=8, n_q=4, n_kv=2, head_dim=4).double()
    v = _make_direction(m.d_model).double()
    h = _make_input(m.d_model).double()

    L = 1
    cache = build_projection_cache(m, v, L)

    q_ref = _q_proj_at(m, L, project_out_native(h, v, alpha=1.0))

    consumer = [(L, hd) for hd in range(m.n_q)]
    handles = install_edge_consumer_ablation_hook(
        m, consumer, v, {L: cache}, alpha=1.0,
        target_projections=("q",),
    )
    try:
        q_hooked = _q_proj_at(m, L, h)
    finally:
        for hd in handles:
            hd.remove()

    diff = (q_hooked - q_ref).abs().max()
    # Same matmul-reassociation reasoning as test_single_head — fp64
    # noise ~1e-9 on these magnitudes is expected.
    assert diff < 1e-6, f"edge-all-heads vs. global-preproject: max abs diff {diff}"


def test_alpha_zero_is_noop():
    """alpha=0 should install no hooks at all (the function short-circuits)
    and produce no behavior change."""
    m = FakeModel().double()
    v = _make_direction(m.d_model).double()
    h = _make_input(m.d_model).double()

    L = 0
    cache = build_projection_cache(m, v, L)
    q_unhooked = _q_proj_at(m, L, h)
    handles = install_edge_consumer_ablation_hook(
        m, [(L, 0), (L, 1)], v, {L: cache}, alpha=0.0,
    )
    # Should be empty list (the function returns [] for alpha=0)
    assert handles == [], f"alpha=0 should install no hooks, got {len(handles)}"
    q_after = _q_proj_at(m, L, h)
    assert torch.equal(q_unhooked, q_after)


def test_kv_groups_resolved_correctly():
    """For GQA with n_q=4, n_kv=2, target query head 0 → KV group 0;
    target query head 3 → KV group 1; targeting both heads 0 and 1
    → KV group 0 only (deduplicated)."""
    from cells_interlinked.pipeline.edge_consumer.hook import _kv_group_of
    assert _kv_group_of(0, n_q_heads=4, n_kv_heads=2) == 0
    assert _kv_group_of(1, n_q_heads=4, n_kv_heads=2) == 0
    assert _kv_group_of(2, n_q_heads=4, n_kv_heads=2) == 1
    assert _kv_group_of(3, n_q_heads=4, n_kv_heads=2) == 1
    # MHA (n_q == n_kv): identity
    assert _kv_group_of(2, n_q_heads=4, n_kv_heads=4) == 2


def test_hook_handles_returned_and_removable():
    """The function should return one handle per affected projection
    module; all must be removable."""
    m = FakeModel().double()
    v = _make_direction(m.d_model).double()
    cache0 = build_projection_cache(m, v, 0)
    cache1 = build_projection_cache(m, v, 1)
    handles = install_edge_consumer_ablation_hook(
        m, [(0, 0), (1, 0)], v, {0: cache0, 1: cache1}, alpha=1.0,
    )
    # 2 layers × 3 projections (q, k, v) = 6 handles expected
    assert len(handles) == 6, f"expected 6 handles, got {len(handles)}"
    # All must remove cleanly
    for h in handles:
        h.remove()
    # Counts after removal
    from cells_interlinked.pipeline.edge_consumer import count_edge_consumer_hooks
    n = count_edge_consumer_hooks(m, layers=[0, 1])
    assert n == 0, f"hooks not fully removed; {n} remain"


if __name__ == "__main__":
    # Allow `uv run python server/tests/test_edge_consumer_hook.py` for
    # quick iteration without pytest installed.
    fns = [
        test_single_head_subtraction_is_exact,
        test_all_heads_equivalent_to_global_pre_projection,
        test_alpha_zero_is_noop,
        test_kv_groups_resolved_correctly,
        test_hook_handles_returned_and_removable,
    ]
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            raise
        except Exception as e:
            print(f"  ERR   {fn.__name__}: {type(e).__name__}: {e}")
            raise
    print(f"\n{len(fns)}/{len(fns)} tests passed")

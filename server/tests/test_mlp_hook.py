"""Unit tests for install_mlp_residual_ablation_hook.

Invariants:

1. Hook subtracts exactly ``⟨mlp_output, v̂⟩ v̂`` from the MLP's
   output (the post-hook only sees the mlp module's return; the
   parent decoder block's residual-add isn't part of the hook scope).
2. MLPs at layers NOT in the target set are bit-identical.
3. alpha=0 is a no-op (no hooks installed).
4. Hook handles can all be cleanly removed.

Uses the same FakeModel scaffold as test_edge_consumer_hook.py, with
a fake MLP module that just returns its input (so we can verify the
hook's subtraction in isolation).
"""

from __future__ import annotations

import torch
from torch import nn

from cells_interlinked.pipeline.edge_consumer.mlp_hook import (
    count_mlp_hooks,
    install_mlp_residual_ablation_hook,
)


class FakeMLP(nn.Module):
    """Minimal MLP: returns its input unchanged. The hook's subtraction
    is then bit-exact comparable to an external subtraction in tests
    (the MLP isn't doing any compute that would introduce its own fp
    rounding). One Linear so the module has a parameter for dtype
    inspection."""

    def __init__(self, d_model: int):
        super().__init__()
        # A no-op linear: identity weights, zero bias. Lets the hook
        # inspect a real parameter for dtype/device, but doesn't alter
        # the forward output, so our equivalence math stays clean.
        self.linear = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(d_model))

    def forward(self, x):
        return self.linear(x)


class FakeBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = FakeMLP(d_model)


class FakeInner(nn.Module):
    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [FakeBlock(d_model) for _ in range(n_layers)]
        )


class FakeModel(nn.Module):
    def __init__(self, d_model: int = 8, n_layers: int = 4):
        super().__init__()
        self.model = FakeInner(d_model, n_layers)
        self.d_model = d_model
        self.n_layers = n_layers


def _make_input(d_model: int, seq_len: int = 5) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(1, seq_len, d_model)


def _make_direction(d_model: int) -> torch.Tensor:
    torch.manual_seed(123)
    v = torch.randn(d_model)
    return v / v.norm()


def _mlp_at(model: FakeModel, L: int):
    return model.model.layers[L].mlp


# ── Tests ──────────────────────────────────────────────────────────────


def test_single_mlp_subtraction_is_exact():
    """Hook on L=2 should make L=2's MLP output equal to
    ``mlp_unhooked(x) - ⟨mlp_unhooked(x), v̂⟩ v̂``."""
    m = FakeModel().double()
    v = _make_direction(m.d_model).double()
    x = _make_input(m.d_model).double()

    # Reference: run unhooked, manually project out v.
    unhooked = _mlp_at(m, 2)(x)
    v_unit = v / v.norm()
    coeff = (unhooked * v_unit).sum(dim=-1, keepdim=True)
    expected = unhooked - coeff * v_unit

    handles = install_mlp_residual_ablation_hook(m, [2], v, alpha=1.0)
    try:
        hooked = _mlp_at(m, 2)(x)
    finally:
        for h in handles:
            h.remove()

    diff = (hooked - expected).abs().max().item()
    # fp64 summation reassociation noise on FakeMLP's identity matmul.
    # Same pattern as test_edge_consumer_hook.py — the math is correct,
    # the floating-point order of operations differs.
    assert diff < 1e-6, f"single MLP mismatch: max abs diff {diff}"


def test_other_mlps_are_bit_identical():
    """L=2 hook must not perturb L=0, 1, 3 MLP outputs at all."""
    m = FakeModel().double()
    v = _make_direction(m.d_model).double()
    x = _make_input(m.d_model).double()

    unhooked_outputs = {L: _mlp_at(m, L)(x).clone() for L in range(m.n_layers)}

    handles = install_mlp_residual_ablation_hook(m, [2], v, alpha=1.0)
    try:
        hooked_outputs = {L: _mlp_at(m, L)(x).clone() for L in range(m.n_layers)}
    finally:
        for h in handles:
            h.remove()

    for L in range(m.n_layers):
        if L == 2:
            continue
        assert torch.equal(unhooked_outputs[L], hooked_outputs[L]), (
            f"non-target L={L} MLP output drifted"
        )


def test_alpha_zero_is_noop():
    m = FakeModel().double()
    v = _make_direction(m.d_model).double()
    handles = install_mlp_residual_ablation_hook(
        m, [0, 1, 2], v, alpha=0.0,
    )
    assert handles == [], f"alpha=0 should install zero hooks, got {len(handles)}"
    # No-op confirmed by handle count + count_mlp_hooks reading 0.
    assert count_mlp_hooks(m, [0, 1, 2]) == 0


def test_multiple_mlps_all_hooked():
    m = FakeModel().double()
    v = _make_direction(m.d_model).double()
    handles = install_mlp_residual_ablation_hook(
        m, [0, 2, 3], v, alpha=1.0,
    )
    assert len(handles) == 3
    assert count_mlp_hooks(m, [0, 1, 2, 3]) == 3
    for h in handles:
        h.remove()
    assert count_mlp_hooks(m, [0, 1, 2, 3]) == 0


def test_empty_layer_list_is_noop():
    m = FakeModel().double()
    v = _make_direction(m.d_model).double()
    handles = install_mlp_residual_ablation_hook(m, [], v, alpha=1.0)
    assert handles == []


def test_v_is_normalized_defensively():
    """A non-unit v passed in should give the same result as the unit
    version (the hook normalizes internally — same convention as
    project_out)."""
    m = FakeModel().double()
    v_unit = _make_direction(m.d_model).double()
    v_scaled = v_unit * 7.5  # non-unit
    x = _make_input(m.d_model).double()

    handles = install_mlp_residual_ablation_hook(m, [1], v_unit, alpha=1.0)
    try:
        out_unit = _mlp_at(m, 1)(x).clone()
    finally:
        for h in handles:
            h.remove()
    handles = install_mlp_residual_ablation_hook(m, [1], v_scaled, alpha=1.0)
    try:
        out_scaled = _mlp_at(m, 1)(x).clone()
    finally:
        for h in handles:
            h.remove()
    diff = (out_unit - out_scaled).abs().max().item()
    assert diff < 1e-6, f"non-unit v gave different result: max abs diff {diff}"


if __name__ == "__main__":
    fns = [
        test_single_mlp_subtraction_is_exact,
        test_other_mlps_are_bit_identical,
        test_alpha_zero_is_noop,
        test_multiple_mlps_all_hooked,
        test_empty_layer_list_is_noop,
        test_v_is_normalized_defensively,
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

"""Unit tests for install_attn_block_residual_ablation_hook.

Mirrors test_mlp_hook.py but with a FakeSelfAttn that returns the
HuggingFace-style tuple ``(hidden_states, attn_weights)`` so we
exercise the tuple-handling branch of the hook. Also includes a
test for the rare plain-Tensor return path.
"""

from __future__ import annotations

import torch
from torch import nn

from cells_interlinked.pipeline.edge_consumer.attn_block_hook import (
    count_attn_block_hooks,
    install_attn_block_residual_ablation_hook,
)


class FakeSelfAttn(nn.Module):
    """Returns ``(hidden_states, attn_weights)``. Identity Linear so
    the per-position math is bit-comparable to an external reference."""

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(d_model))

    def forward(self, x):
        h = self.linear(x)
        # Fake attn_weights (right shape doesn't matter for the hook).
        attn_w = torch.zeros(x.shape[0], 1, x.shape[1], x.shape[1])
        return h, attn_w


class FakeSelfAttnPlainTensor(nn.Module):
    """Returns a plain Tensor (rare in modern HF code paths but the
    hook should still handle it)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(d_model))

    def forward(self, x):
        return self.linear(x)


class FakeBlock(nn.Module):
    def __init__(self, d_model: int, tuple_return: bool = True):
        super().__init__()
        cls = FakeSelfAttn if tuple_return else FakeSelfAttnPlainTensor
        self.self_attn = cls(d_model)


class FakeInner(nn.Module):
    def __init__(self, d_model: int, n_layers: int, tuple_return: bool = True):
        super().__init__()
        self.layers = nn.ModuleList(
            [FakeBlock(d_model, tuple_return=tuple_return) for _ in range(n_layers)]
        )


class FakeModel(nn.Module):
    def __init__(self, d_model: int = 8, n_layers: int = 4, tuple_return: bool = True):
        super().__init__()
        self.model = FakeInner(d_model, n_layers, tuple_return=tuple_return)
        self.d_model = d_model
        self.n_layers = n_layers


def _make_input(d_model: int, seq_len: int = 5) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(1, seq_len, d_model)


def _make_direction(d_model: int) -> torch.Tensor:
    torch.manual_seed(123)
    v = torch.randn(d_model)
    return v / v.norm()


def _attn_at(model: FakeModel, L: int):
    return model.model.layers[L].self_attn


# ── Tests ──────────────────────────────────────────────────────────────


def test_tuple_return_single_layer_subtraction_is_exact():
    """Hook on L=2 should make L=2's hidden_states equal to
    ``attn_unhooked(x) − ⟨attn_unhooked(x), v̂⟩ v̂``, while
    attn_weights (output[1]) passes through unchanged."""
    m = FakeModel(tuple_return=True).double()
    v = _make_direction(m.d_model).double()
    x = _make_input(m.d_model).double()

    unhooked_h, unhooked_w = _attn_at(m, 2)(x)
    v_unit = v / v.norm()
    coeff = (unhooked_h * v_unit).sum(dim=-1, keepdim=True)
    expected_h = unhooked_h - coeff * v_unit

    handles = install_attn_block_residual_ablation_hook(m, [2], v, alpha=1.0)
    try:
        hooked_h, hooked_w = _attn_at(m, 2)(x)
    finally:
        for h in handles:
            h.remove()

    diff_h = (hooked_h - expected_h).abs().max().item()
    # Same fp64 reassociation noise as the MLP hook test.
    assert diff_h < 1e-6, f"hidden_states mismatch: max abs diff {diff_h}"
    # attn_weights should be bit-identical (not touched by the hook).
    assert torch.equal(hooked_w, unhooked_w), "attn_weights altered"


def test_other_attn_blocks_are_bit_identical():
    """L=2 hook must not perturb L=0, 1, 3 attention outputs at all."""
    m = FakeModel(tuple_return=True).double()
    v = _make_direction(m.d_model).double()
    x = _make_input(m.d_model).double()

    unhooked = {
        L: tuple(t.clone() for t in _attn_at(m, L)(x))
        for L in range(m.n_layers)
    }
    handles = install_attn_block_residual_ablation_hook(m, [2], v, alpha=1.0)
    try:
        hooked = {
            L: tuple(t.clone() for t in _attn_at(m, L)(x))
            for L in range(m.n_layers)
        }
    finally:
        for h in handles:
            h.remove()
    for L in range(m.n_layers):
        if L == 2:
            continue
        for i, (a, b) in enumerate(zip(unhooked[L], hooked[L])):
            assert torch.equal(a, b), (
                f"non-target L={L} output[{i}] drifted"
            )


def test_plain_tensor_return_is_supported():
    """If self_attn returns a plain Tensor (not a tuple), the hook
    still works."""
    m = FakeModel(tuple_return=False).double()
    v = _make_direction(m.d_model).double()
    x = _make_input(m.d_model).double()

    unhooked = _attn_at(m, 1)(x)
    v_unit = v / v.norm()
    coeff = (unhooked * v_unit).sum(dim=-1, keepdim=True)
    expected = unhooked - coeff * v_unit

    handles = install_attn_block_residual_ablation_hook(m, [1], v, alpha=1.0)
    try:
        hooked = _attn_at(m, 1)(x)
    finally:
        for h in handles:
            h.remove()
    diff = (hooked - expected).abs().max().item()
    assert diff < 1e-6, f"plain-tensor mismatch: max abs diff {diff}"


def test_alpha_zero_is_noop():
    m = FakeModel().double()
    v = _make_direction(m.d_model).double()
    handles = install_attn_block_residual_ablation_hook(
        m, [0, 1, 2], v, alpha=0.0,
    )
    assert handles == [], f"alpha=0 should install zero hooks, got {len(handles)}"
    assert count_attn_block_hooks(m, [0, 1, 2]) == 0


def test_multiple_layers_all_hooked():
    m = FakeModel().double()
    v = _make_direction(m.d_model).double()
    handles = install_attn_block_residual_ablation_hook(
        m, [0, 2, 3], v, alpha=1.0,
    )
    assert len(handles) == 3
    assert count_attn_block_hooks(m, [0, 1, 2, 3]) == 3
    for h in handles:
        h.remove()
    assert count_attn_block_hooks(m, [0, 1, 2, 3]) == 0


def test_v_normalized_defensively():
    m = FakeModel().double()
    v_unit = _make_direction(m.d_model).double()
    v_scaled = v_unit * 13.7
    x = _make_input(m.d_model).double()

    handles = install_attn_block_residual_ablation_hook(m, [1], v_unit, alpha=1.0)
    try:
        out_u, _ = _attn_at(m, 1)(x)
    finally:
        for h in handles:
            h.remove()
    handles = install_attn_block_residual_ablation_hook(m, [1], v_scaled, alpha=1.0)
    try:
        out_s, _ = _attn_at(m, 1)(x)
    finally:
        for h in handles:
            h.remove()
    diff = (out_u - out_s).abs().max().item()
    assert diff < 1e-6, f"non-unit v gave different output: max abs diff {diff}"


if __name__ == "__main__":
    fns = [
        test_tuple_return_single_layer_subtraction_is_exact,
        test_other_attn_blocks_are_bit_identical,
        test_plain_tensor_return_is_supported,
        test_alpha_zero_is_noop,
        test_multiple_layers_all_hooked,
        test_v_normalized_defensively,
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

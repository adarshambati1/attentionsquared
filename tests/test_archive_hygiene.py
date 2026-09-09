from pathlib import Path

import torch

from src.models.jump_mlp import JumpMLP


ROOT = Path(__file__).resolve().parents[1]


def test_active_python_does_not_import_archived_scripts():
    for directory in (ROOT / "src", ROOT / "scripts"):
        for path in directory.rglob("*.py"):
            if ROOT / "scripts" / "archive" in path.parents:
                continue
            assert "scripts.archive" not in path.read_text(), path


def test_jump_mlp_shapes_and_residual_contract():
    h0 = torch.randn(2, 3, 8)
    x = torch.randn_like(h0)
    direct = JumpMLP(8, residual=False)
    residual = JumpMLP(8, residual=True)
    residual.load_state_dict(direct.state_dict())

    direct_output = direct(h0, x)
    residual_output = residual(h0, x)

    assert direct_output.shape == h0.shape
    assert torch.allclose(residual_output, h0 + direct_output)

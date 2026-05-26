import torch
import pytest
from nutrition5k_pkg.models.backbone import InceptionV3Backbone
from nutrition5k_pkg.models.multitask_head import MultitaskNutritionNet
from nutrition5k_pkg.models.mass_regressor import MassRegressor


def test_backbone_output_shape_3ch():
    model = InceptionV3Backbone(in_channels=3)
    model.eval()
    x = torch.zeros(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 8192)


def test_backbone_output_shape_4ch():
    model = InceptionV3Backbone(in_channels=4)
    model.eval()
    x = torch.zeros(2, 4, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 8192)


def test_backbone_4ch_rgb_weights_preserved():
    """The 4-channel backbone should share RGB weights with the 3-channel pretrained backbone."""
    m3 = InceptionV3Backbone(in_channels=3)
    m4 = InceptionV3Backbone(in_channels=4)
    w3 = m3.conv1.conv.weight.data  # (32, 3, 3, 3)
    w4 = m4.conv1.conv.weight.data  # (32, 4, 3, 3)
    assert torch.allclose(w3, w4[:, :3])
    assert w4[:, 3:].abs().sum() == pytest.approx(0.0)


def test_multitask_direct_output():
    tasks = ['calories', 'mass', 'fat', 'carb', 'protein']
    model = MultitaskNutritionNet(tasks=tasks, in_channels=3)
    model.eval()
    x = torch.zeros(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert set(out.keys()) == set(tasks)
    for t in tasks:
        assert out[t].shape == (2,)


def test_multitask_per_gram_output():
    tasks = ['cal_per_g', 'fat_per_g', 'carb_per_g', 'protein_per_g']
    model = MultitaskNutritionNet(tasks=tasks, in_channels=3)
    model.eval()
    x = torch.zeros(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert set(out.keys()) == set(tasks)


def test_multitask_4ch_input():
    tasks = ['calories', 'mass', 'fat', 'carb', 'protein']
    model = MultitaskNutritionNet(tasks=tasks, in_channels=4)
    model.eval()
    x = torch.zeros(2, 4, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out['calories'].shape == (2,)


def test_mass_regressor_output():
    model = MassRegressor()
    model.eval()
    x = torch.zeros(2, 3, 256, 256)
    volume = torch.tensor([15.0, 22.0])
    with torch.no_grad():
        out = model(x, volume)
    assert out.shape == (2,)


def test_mass_regressor_without_volume():
    """Image-only baseline (volume=None) should also work."""
    model = MassRegressor(use_volume=False)
    model.eval()
    x = torch.zeros(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x, None)
    assert out.shape == (2,)


def test_mass_regressor_volume_required():
    import pytest
    model = MassRegressor(use_volume=True)
    x = torch.zeros(2, 3, 256, 256)
    with pytest.raises(ValueError, match="volume tensor required"):
        model(x, None)


def test_mass_regressor_volume_affects_output():
    model = MassRegressor(use_volume=True)
    model.eval()
    x = torch.zeros(2, 3, 256, 256)
    vol_a = torch.tensor([1.0, 1.0])
    vol_b = torch.tensor([1000.0, 1000.0])
    with torch.no_grad():
        out_a = model(x, vol_a)
        out_b = model(x, vol_b)
    assert not torch.allclose(out_a, out_b), "Different volumes should produce different outputs"

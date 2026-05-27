import torch

from nutrition5k_pkg.models.dpf_nutrition import CrossAttentionBlock, DPFNutritionNet


def test_cross_attention_block_preserves_spatial_size_and_projects_channels():
    block = CrossAttentionBlock(in_channels=8, out_channels=12, reduction=4)
    rgb = torch.randn(2, 8, 21, 28)
    depth = torch.randn(2, 8, 21, 28)

    out = block(rgb, depth)

    assert out.shape == (2, 12, 21, 28)


def test_dpf_nutrition_forward_returns_five_task_vectors():
    tasks = ['calories', 'mass', 'fat', 'carb', 'protein']
    model = DPFNutritionNet(
        tasks=tasks,
        pretrained=False,
        fusion_channels=32,
        head_hidden=16,
        resnet_layers=(1, 1, 1, 1),
    )
    model.eval()
    rgb = torch.zeros(2, 3, 336, 448)
    depth = torch.zeros(2, 1, 336, 448)

    with torch.no_grad():
        out = model(rgb, depth)

    assert set(out.keys()) == set(tasks)
    for task in tasks:
        assert out[task].shape == (2,)


def test_food2k_state_extraction_removes_common_prefixes():
    checkpoint = {
        'state_dict': {
            'module.encoder_q.backbone.conv1.weight': torch.zeros(64, 3, 7, 7),
            'module.encoder_q.backbone.bn1.weight': torch.ones(64),
            'module.encoder_q.backbone.fc.weight': torch.zeros(10, 2048),
        }
    }

    state = DPFNutritionNet._extract_resnet_state(checkpoint)

    assert set(state) == {'conv1.weight', 'bn1.weight'}

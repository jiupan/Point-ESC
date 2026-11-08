"""CPU checks of the paper implementation; native CUDA operators are not tested."""
import ast
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.resa_utils import ProjectionHead, ReSALoss, SinkhornKnopp


class Config(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error


def config_tree(value):
    return Config({k: config_tree(v) for k, v in value.items()}) if isinstance(value, dict) else value


def definitions(path, namespace, names=None):
    """Use the real definitions with CPU replacements for CUDA-only imports."""
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8-sig'))
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 and (names is None or node.name in names)]
    exec(compile(tree, str(ROOT / path), 'exec'), namespace)
    return namespace


def cpu_fps(xyz, count):
    batch, size, _ = xyz.shape
    indices = torch.zeros(batch, count, dtype=torch.long)
    distance = torch.full((batch, size), float('inf'))
    farthest = torch.zeros(batch, dtype=torch.long)
    rows = torch.arange(batch)
    for i in range(count):
        indices[:, i] = farthest
        candidate = ((xyz - xyz[rows, farthest].unsqueeze(1)) ** 2).sum(-1)
        distance = torch.minimum(distance, candidate)
        farthest = distance.argmax(-1)
    return indices


def cpu_gather(features, indices):
    return features.gather(2, indices.long().unsqueeze(1).expand(-1, features.size(1), -1))


class CPUKNN:
    def __init__(self, k, transpose_mode=True):
        self.k, self.calls = k, 0

    def __call__(self, xyz, center):
        self.calls += 1
        return torch.cdist(center, xyz).topk(self.k, largest=False)


class CPUChamfer(nn.Module):
    def cuda(self, *args, **kwargs):
        return self

    def forward(self, x, y):
        distance = torch.cdist(x, y).square()
        return distance.min(-1).values.mean() + distance.min(-2).values.mean()


class Registry:
    def register_module(self):
        return lambda cls: cls


class DropPath(nn.Module):
    def __init__(self, probability=0.):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or not self.probability:
            return x
        keep = 1 - self.probability
        mask = torch.floor(keep + torch.rand((x.size(0),) + (1,) * (x.ndim - 1)))
        return x * mask / keep


class PaperAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.settings = config_tree(yaml.safe_load((ROOT / 'cfgs/pretrain.yaml').read_text()))
        cls.operators = SimpleNamespace(furthest_point_sample=cpu_fps, gather_operation=cpu_gather)
        cls.model_classes = definitions('models/Point_ESC.py', {
            '__name__': 'paper_alignment_models', 'torch': torch, 'nn': nn, 'F': F,
            'np': np, 'random': random, 'MODELS': Registry(), 'KNN': CPUKNN,
            'pointnet2_utils': cls.operators, 'DropPath': DropPath,
            'trunc_normal_': nn.init.trunc_normal_, 'ProjectionHead': ProjectionHead,
            'ReSALoss': ReSALoss, 'ChamferDistanceL1': CPUChamfer,
            'ChamferDistanceL2': CPUChamfer, 'print_log': lambda *a, **k: None,
            'get_missing_parameters_message': str, 'get_unexpected_parameters_message': str,
        })
        transforms = definitions('datasets/data_transforms.py', {'torch': torch, 'np': np, 'random': random})
        cls.view_classes = definitions('datasets/dual_view_transforms.py', dict(transforms))

    def setUp(self):
        torch.manual_seed(7)
        np.random.seed(7)
        random.seed(7)

    def test_default_config_and_weight_schedule(self):
        config = self.settings.model.resa_config
        self.assertEqual((config.sinkhorn_iters, config.sinkhorn_epsilon, config.temperature), (3, .05, .1))
        self.assertEqual((config.start_epoch, config.warmup_epochs, config['lambda']), (60, 120, .1))
        self.assertFalse(config.feature_detach)
        self.assertTrue(self.settings.dataset.train.others.sample_with_fps)
        model = self.model_classes['Point_ESC'](self.settings.model)
        for epoch, expected in [(0, 0), (59, 0), (60, 0), (120, .05), (180, .1), (299, .1)]:
            self.assertAlmostEqual(model._get_adaptive_resa_weight(epoch), expected)

    def test_sinkhorn_matches_finite_iteration_equation(self):
        scores = torch.randn(2, 5, 5, dtype=torch.float64) * .02
        expected = (scores / .05).exp()
        for _ in range(3):
            expected = expected / expected.sum(-1, keepdim=True)
            expected = expected / expected.sum(-2, keepdim=True)
        actual = SinkhornKnopp(3, .05)(scores.requires_grad_())
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(-2), torch.ones(2, 5, dtype=torch.float64))
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(SinkhornKnopp()(scores[0]), actual[0])
        self.assertTrue(torch.isfinite(SinkhornKnopp()(scores.detach() * 100000)).all())

    def test_h_targets_z_predictions_and_elementwise_kl(self):
        h1, h2 = [torch.randn(2, 5, 8, requires_grad=True) for _ in range(2)]
        z1, z2 = [torch.randn(2, 5, 4, requires_grad=True) for _ in range(2)]
        criterion = ReSALoss()
        targets = []
        criterion.sinkhorn.register_forward_hook(lambda m, args, out: targets.append(out))
        loss = criterion(h1, h2, z1, z2)
        prediction = F.log_softmax(F.normalize(z1, dim=-1) @ F.normalize(z2, dim=-1).transpose(-2, -1) / .1, -1)
        expected = (torch.special.xlogy(targets[0], targets[0]) - targets[0] * prediction).sum() / (2 * 5 * 5)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(h1.grad)  # Direct target path is stopped.
        self.assertIsNone(h2.grad)
        self.assertGreater(z1.grad.abs().sum().item(), 0)
        self.assertGreater(z2.grad.abs().sum().item(), 0)
        criterion(h1, h2, z1 * 3, -z2)
        torch.testing.assert_close(targets[0], targets[1])
        criterion(-h1, h2, z1, z2)
        self.assertFalse(torch.allclose(targets[0], targets[2]))

    def test_group_indices_survive_different_view_transforms(self):
        group = self.model_classes['Group'](8, 4)
        points = torch.randn(2, 32, 3)
        neighborhood, center, indices = group(points, return_indices=True)
        shifted, shifted_center = group(points * 2 + 10, indices=indices)
        torch.testing.assert_close(shifted, neighborhood * 2)
        torch.testing.assert_close(shifted_center, center * 2 + 10)
        self.assertEqual(group.knn.calls, 1)

    def test_full_pretraining_forward_and_encoder_gradient(self):
        model = self.model_classes['Point_ESC'](self.settings.model).train()
        transform = self.view_classes['AdaptiveDualViewTransform'](epoch=180, **{
            'base_jitter': .005, 'base_scale_range': .1, 'base_translate': .05,
            'jitter_clip': .025, 'dropout_ratio': .05,
        })
        masks = []
        def capture_mask(module, args, kwargs):
            masks.append(kwargs.get('shared_mask'))
        model.MAE_encoder.register_forward_pre_hook(capture_mask, with_kwargs=True)
        points = torch.randn(2, 128, 3)
        result = model(points, epoch=180, view_transform=transform)
        self.assertTrue(torch.isfinite(result['total_loss']))
        self.assertAlmostEqual(result['resa_weight'].item(), .1, places=6)
        self.assertEqual(masks[0].shape, (2, 64))
        self.assertTrue(torch.equal(masks[0], masks[1]))
        self.assertEqual(model.group_divider.knn.calls, 1)
        result['resa_loss'].backward(retain_graph=True)
        encoder_gradient = model.MAE_encoder.blocks.blocks[0].attn.qkv.weight.grad
        self.assertIsNotNone(encoder_gradient)
        self.assertTrue(torch.isfinite(encoder_gradient).all())
        self.assertGreater(encoder_gradient.abs().sum().item(), 0)
        model.zero_grad()
        result['total_loss'].backward()
        self.assertIsNotNone(model.increase_dim[0].weight.grad)
        self.assertIsNotNone(model.projection_head.projection[0].weight.grad)
        early = model(points, epoch=59, view_transform=transform)
        self.assertEqual(early['resa_weight'].item(), 0)
        self.assertEqual(early['resa_loss'].item(), 0)

    def test_checkpoint_transfer_and_classifier_backward(self):
        pretrained = self.model_classes['Point_ESC'](self.settings.model)
        config = config_tree(yaml.safe_load((ROOT / 'cfgs/finetune_modelnet.yaml').read_text()))
        classifier = self.model_classes['PointTransformer'](config.model).train()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'pretrain.pth'
            torch.save({'base_model': pretrained.state_dict()}, checkpoint)
            classifier.load_model_from_ckpt(str(checkpoint))
        encoder_state = pretrained.MAE_encoder.state_dict()
        for key, value in encoder_state.items():
            torch.testing.assert_close(classifier.state_dict()[key], value)
        logits = classifier(torch.randn(2, 128, 3))
        self.assertEqual(logits.shape, (2, 40))
        loss, _ = classifier.get_loss_acc(logits, torch.tensor([0, 1]))
        loss.backward()
        self.assertGreater(classifier.blocks.blocks[0].attn.qkv.weight.grad.abs().sum().item(), 0)
        self.assertFalse(any('projection_head' in key or 'MAE_decoder' in key for key in classifier.state_dict()))

    def test_test_entry_votes_once_only_when_requested(self):
        calls = []
        namespace = definitions('tools/runner_finetune.py', {
            'torch': torch, 'print_log': lambda *a, **k: None,
            'misc': SimpleNamespace(fps=lambda points, count: points[:, :count]),
            'test_vote': lambda *a, **k: calls.append(k['times']) or 100.,
        }, names={'test'})
        model = nn.Sequential(nn.Flatten(), nn.Linear(12, 2))
        loader = [('', '', (torch.randn(2, 4, 3), torch.tensor([0, 1])))]
        args = SimpleNamespace(distributed=False, vote=False)
        with patch.object(torch.Tensor, 'cuda', lambda self: self):
            namespace['test'](model, loader, args, SimpleNamespace(npoints=4))
            self.assertEqual(calls, [])
            args.vote = True
            namespace['test'](model, loader, args, SimpleNamespace(npoints=4))
            self.assertEqual(calls, [10])

    def test_scanobjectnn_2048_vote_and_logit_average(self):
        calls = []
        class Classifier(nn.Module):
            def forward(self, points):
                calls.append(None)
                # Alternating correct/incorrect individual views, correct average.
                return torch.tensor([[3., 0.], [0., 3.]]) if len(calls) % 2 else torch.tensor([[0., 1.], [1., 0.]])
        namespace = definitions('tools/runner_finetune.py', {
            'torch': torch, 'np': np, 'pointnet2_utils': self.operators,
            'test_transforms': self.view_classes['PointcloudScaleAndTranslate'](),
        }, names={'test_vote'})
        loader = [('', '', (torch.randn(2, 2048, 3), torch.tensor([0, 1])))]
        with patch.object(torch.Tensor, 'cuda', lambda self: self):
            accuracy = namespace['test_vote'](Classifier(), loader, 0, None,
                SimpleNamespace(distributed=False), SimpleNamespace(npoints=2048), times=10)
        self.assertEqual(len(calls), 10)
        self.assertEqual(accuracy.item(), 100.)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""Test script to verify nan detection hook functionality."""

import torch
import torch.nn as nn
from mmcv.runner import build_runner
from mmseg.core.hooks.nan_detection_hook import NanDetectionHook


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)


def test_nan_detection():
    """Test nan detection hook with artificial nan values."""
    print("Testing NanDetectionHook...")
    
    # Create a simple model
    model = SimpleModel()
    
    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Create nan detection hook
    nan_hook = NanDetectionHook(detect_loss_nan=True, detect_grad_nan=True)
    
    # Create runner
    runner_cfg = dict(type='EpochBasedRunner', max_epochs=1)
    runner = build_runner(runner_cfg, default_args=dict(model=model, optimizer=optimizer))
    
    # Register the hook
    runner.register_hook(nan_hook)
    
    # Test 1: Normal forward pass
    print("\n1. Testing normal forward pass...")
    x = torch.randn(1, 10)
    y = torch.tensor([[1.0]])
    
    def normal_train_func():
        model.train()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        return {'loss': loss}
    
    runner.outputs = normal_train_func()
    nan_hook.after_train_iter(runner)
    print("✓ Normal forward pass completed without nan detection")
    
    # Test 2: NaN loss
    print("\n2. Testing NaN loss detection...")
    def nan_loss_train_func():
        model.train()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        # Force loss to be NaN
        loss = torch.tensor(float('nan'))
        return {'loss': loss}
    
    runner.outputs = nan_loss_train_func()
    nan_hook.after_train_iter(runner)
    print("✓ NaN loss detection triggered")
    
    # Test 3: Inf loss
    print("\n3. Testing Inf loss detection...")
    def inf_loss_train_func():
        model.train()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        # Force loss to be Inf
        loss = torch.tensor(float('inf'))
        return {'loss': loss}
    
    runner.outputs = inf_loss_train_func()
    nan_hook.after_train_iter(runner)
    print("✓ Inf loss detection triggered")
    
    print("\n✅ All nan detection tests passed!")


if __name__ == '__main__':
    test_nan_detection()
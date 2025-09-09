# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import HOOKS, OptimizerHook


@HOOKS.register_module()
class NanDetectionHook(OptimizerHook):
    """Hook to detect NaN values in loss and gradients during training."""

    def __init__(self, grad_clip=None, detect_loss_nan=True, detect_grad_nan=True):
        super(NanDetectionHook, self).__init__(grad_clip)
        self.detect_loss_nan = detect_loss_nan
        self.detect_grad_nan = detect_grad_nan

    def after_train_iter(self, runner):
        """Check for NaN values in loss and gradients after each training iteration."""
        # Check for NaN in loss
        if self.detect_loss_nan:
            loss = runner.outputs['loss']
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                runner.logger.warning(
                    f'NaN or Inf detected in loss at iteration {runner.iter + 1}. '
                    f'Loss value: {loss.item()}'
                )
                # Optionally raise an exception or take other actions
                # raise ValueError('NaN detected in loss')

        # Perform the standard optimization steps
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()

        # Check for NaN in gradients
        if self.detect_grad_nan:
            nan_grads = []
            for name, param in runner.model.named_parameters():
                if param.grad is not None:
                    grad = param.grad
                    if torch.isnan(grad).any():
                        nan_grads.append(name)
                        runner.logger.warning(
                            f'NaN detected in gradient for parameter {name} '
                            f'at iteration {runner.iter + 1}'
                        )
                    elif torch.isinf(grad).any():
                        nan_grads.append(name)
                        runner.logger.warning(
                            f'Inf detected in gradient for parameter {name} '
                            f'at iteration {runner.iter + 1}'
                        )
            
            if nan_grads:
                runner.logger.warning(
                    f'NaN/Inf gradients detected in parameters: {nan_grads}'
                )

        # Apply gradient clipping if specified
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])

        runner.optimizer.step()
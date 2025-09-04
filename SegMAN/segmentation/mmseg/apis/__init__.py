# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_segmentor, init_segmentor, show_result_pyplot
from .inference_ref import inference_segmentor_ref, init_segmentor_ref, show_result_pyplot_ref
from .inference_efs import inference_segmentor_efs, init_segmentor_efs, show_result_pyplot_efs
from .test import multi_gpu_test, single_gpu_test
from .train import (get_root_logger, init_random_seed, set_random_seed,
                    train_segmentor)

__all__ = [
    'get_root_logger', 'set_random_seed', 'train_segmentor', 'init_segmentor', 'init_segmentor_ref', 'init_segmentor_efs',
    'inference_segmentor', 'inference_segmentor_ref', 'inference_segmentor_efs', 'multi_gpu_test', 'single_gpu_test',
    'show_result_pyplot', 'show_result_pyplot_ref', 'show_result_pyplot_efs', 'init_random_seed'
]

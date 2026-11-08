import numpy as np
import torch
import random
from .data_transforms import PointcloudRotate, PointcloudJitter, PointcloudScale, PointcloudTranslate, PointcloudRandomInputDropout


class DualViewTransform(object):
    def __init__(self, 
                 jitter_std=0.01, 
                 jitter_clip=0.05,
                 scale_low=0.8, 
                 scale_high=1.2,
                 translate_range=0.1,
                 dropout_ratio=0.1):
        
        self.transform_1 = [
            PointcloudJitter(std=jitter_std*0.5, clip=jitter_clip*0.5),
            PointcloudScale(scale_low=0.9, scale_high=1.1),
            PointcloudRotate(),
        ]
        
        self.transform_2 = [
            PointcloudJitter(std=jitter_std, clip=jitter_clip),
            PointcloudScale(scale_low=scale_low, scale_high=scale_high),
            PointcloudTranslate(translate_range=translate_range),
            PointcloudRotate(),
            PointcloudRandomInputDropout(max_dropout_ratio=dropout_ratio),
        ]
    
    def __call__(self, pc):
        view1 = pc.clone()
        view2 = pc.clone()
        
        for transform in self.transform_1:
            view1 = transform(view1)
            
        for transform in self.transform_2:
            view2 = transform(view2)
            
        return view1, view2


class SingleViewAugment(object):
    def __init__(self, 
                 jitter_std=0.01, 
                 jitter_clip=0.05,
                 scale_low=0.8, 
                 scale_high=1.2,
                 translate_range=0.1):
        
        self.transforms = [
            PointcloudJitter(std=jitter_std, clip=jitter_clip),
            PointcloudScale(scale_low=scale_low, scale_high=scale_high),
            PointcloudTranslate(translate_range=translate_range),
            PointcloudRotate(),
        ]
    
    def __call__(self, pc):
        for transform in self.transforms:
            pc = transform(pc)
        return pc


class AdaptiveDualViewTransform(object):
    def __init__(self,
                 epoch=0,
                 max_epochs=300,
                 base_jitter=0.01,
                 base_scale_range=0.2,
                 base_translate=0.1,
                 jitter_clip=None,
                 dropout_ratio=None):

        progress = min(epoch / max_epochs, 1.0)

        strength_factor = 1.0 - progress * 0.5

        jitter_std = base_jitter * strength_factor
        scale_range = base_scale_range * strength_factor
        translate_range = base_translate * strength_factor

        kwargs = dict(
            jitter_std=jitter_std,
            scale_low=1.0 - scale_range,
            scale_high=1.0 + scale_range,
            translate_range=translate_range,
        )
        if jitter_clip is not None:
            kwargs['jitter_clip'] = jitter_clip
        if dropout_ratio is not None:
            kwargs['dropout_ratio'] = dropout_ratio

        self.dual_transform = DualViewTransform(**kwargs)

        
    def __call__(self, pc):
        return self.dual_transform(pc)

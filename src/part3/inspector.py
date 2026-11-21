"""Use layers 6/12, with the official final-layer PCA mask held constant."""
import numpy as np
import torch
from src.part2.inspector import AnomalyDINOInspector
from src.part3.features import fuse_features


class MultilayerInspector(AnomalyDINOInspector):
    def __init__(self,*args,variant='fusion6_12',**kwargs):
        if variant not in {'layer12','layer6','fusion6_12'}:
            raise ValueError('Unknown preregistered variant')
        super().__init__(*args,**kwargs)
        self.variant=variant
        self.backbone.model.requires_grad_(False)
        original_mask=self.backbone.compute_background_mask
        self._final_features=None

        def extract(tensor):
            with torch.inference_mode():
                layers=self.backbone.model.get_intermediate_layers(tensor.unsqueeze(0).to(self.device),n=[5,11],norm=True)
                mid,final=[x.squeeze(0).cpu().numpy() for x in layers]
            self._final_features=final
            if self.variant=='layer12':return final
            if self.variant=='layer6':return mid
            return fuse_features(mid,final)

        def mask(features,grid,*a,**kw):
            if self._final_features is None or len(features)!=len(self._final_features):
                raise RuntimeError('Mask must follow feature extraction of the same image')
            return original_mask(self._final_features,grid,*a,**kw)

        self.backbone.extract_features=extract
        self.backbone.compute_background_mask=mask

import json
import ivy
import numpy as np
import importlib.util
import os

from mlserver import MLModel
from mlserver.codecs import decode_args
from mlserver.utils import get_model_uri
from typing import Optional


IVY_MODEL = 'regressor.py'
IVY_WEIGHTS = "weights.pkl"

class IvyModel(MLModel):
    async def load(self) -> bool:
        model_uri = await get_model_uri(self._settings)
        model_file = os.path.join(model_uri, IVY_MODEL)
        model_weights = os.path.join(model_uri, IVY_WEIGHTS)
        # TODO: this is very band aid-y. needs a more dynamic approach
        # rishab: look into this after getting this POC running
        # also, lint fixes definitely needed
        spec = importlib.util.spec_from_file_location('models', model_file)
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)
        ivy.set_backend(custom_module.backend)
        self._model = custom_module.Regressor(custom_module.input_dim, 
                                              custom_module.output_dim,
                                              is_training=False)
        self._model.v = ivy.Container.cont_from_disk_as_pickled(model_weights)
        return True

    @decode_args
    async def predict(
        self,
        X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        X = ivy.array(X)
        result = self._model(X).data.detach().cpu().numpy()
        return result

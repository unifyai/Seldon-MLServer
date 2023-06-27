import ivy
import numpy as np
import importlib.util
import os

from mlserver import types
from mlserver.model import MLModel
from mlserver.utils import get_model_uri
from mlserver.codecs import NumpyCodec, NumpyRequestCodec


IVY_MODEL = "regressor.py"
IVY_WEIGHTS = "weights.pkl"

# todo: replace
# WELLKNOWN_MODEL_FILENAMES = ["model"]


class IvyModel(MLModel):
    """
    Implementationof the MLModel interface to load and serve `ivy` models.
    """

    async def load(self) -> bool:
        model_uri = await get_model_uri(
            self._settings, 
            # wellknown_filenames=WELLKNOWN_MODEL_FILENAMES
        )
        model_file = os.path.join(model_uri, IVY_MODEL)
        model_weights = os.path.join(model_uri, IVY_WEIGHTS)
        # TODO: replace logic with ivy.load when done - 
        # https://trello.com/c/LgV2zwyD/2141-add-ivyload-and-ivysave-to-ivy-framework
        # uncomment WELLKNOWN_MODEL_FILENAMES according to specs
        spec = importlib.util.spec_from_file_location('models', model_file)
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)
        ivy.set_backend(custom_module.backend)
        self._model = custom_module.Regressor(custom_module.input_dim, 
                                              custom_module.output_dim,
                                              is_training=False)
        self._model.v = ivy.Container.cont_from_disk_as_pickled(model_weights)
        return True

    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        decoded = self.decode_request(payload, default_codec=NumpyRequestCodec)
        decoded = ivy.array(decoded)
        prediction = self._model(decoded).data.detach().cpu().numpy()

        return types.InferenceResponse(
            model_name=self.name,
            model_version=self.version,
            outputs=[NumpyCodec.encode_output(name="predict", payload=prediction)],
        )

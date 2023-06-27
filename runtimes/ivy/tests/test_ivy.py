import pytest
import os
import ivy

from mlserver.settings import ModelSettings
from mlserver.codecs import CodecError
from mlserver.types import RequestInput, InferenceRequest

from mlserver_ivy import IvyModel


def test_load(model: IvyModel):
    assert model.ready
    assert isinstance(type(model._model), ivy.Module)


async def test_load_folder(model_uri: str, model_settings: ModelSettings):
    model_folder = os.path.dirname(model_uri)
    os.rename(model_uri, model_folder)

    model_settings.parameters.uri = model_folder  # type: ignore

    model = IvyModel(model_settings)
    model.ready = await model.load()

    assert model.ready
    assert isinstance(type(model._model), ivy.Module)


async def test_predict(model: IvyModel, inference_request: InferenceRequest):
    response = await model.predict(inference_request)

    assert len(response.outputs) == 1
    assert 0 <= response.outputs[0].data[0] <= 1


async def test_multiple_inputs_error(
    model: IvyModel, inference_request: InferenceRequest
):
    inference_request.inputs.append(
        RequestInput(name="input-1", shape=[1, 3], data=[[0, 1, 2]], datatype="FP32")
    )

    with pytest.raises(CodecError):
        await model.predict(inference_request)

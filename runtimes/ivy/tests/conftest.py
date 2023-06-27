import pytest
import os
import asyncio
import ivy

from mlserver.settings import ModelSettings, ModelParameters
from mlserver.types import InferenceRequest
from mlserver.utils import install_uvloop_event_loop

from mlserver_ivy import IvyModel
from mlserver_ivy.ivy import IVY_MODEL, IVY_WEIGHTS

TESTS_PATH = os.path.dirname(__file__)
TESTDATA_PATH = os.path.join(TESTS_PATH, "testdata")


@pytest.fixture
def event_loop():
    # By default use uvloop for tests
    install_uvloop_event_loop()
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def model_uri(tmp_path) -> str:
    class Regressor(ivy.Module):
        def __init__(self, input_dim, output_dim, is_training=True):
            self.linear = ivy.Linear(input_dim, output_dim)
            self.dropout = ivy.Dropout(0.5, training=is_training)
            ivy.Module.__init__(self)

        def _forward(self, x, ):
            x = ivy.sigmoid(self.linear(x))
            x = self.dropout(x)
            return x

    ivy.set_backend('torch')  # set backend to PyTorch

    model = Regressor(input_dim=3, output_dim=1)
    optimizer = ivy.Adam(1e-4)

    # generate some random data
    x = ivy.random.random_normal(shape=(100, 3))
    y = ivy.random.random_normal(shape=(100, 1))

    def loss_fn(pred, target):
        return ivy.mean((pred - target)**2)

    for epoch in range(50):
        # forward pass
        pred = model(x)

        # compute loss and gradients
        loss, grads = ivy.execute_with_gradients(lambda v: loss_fn(pred, y), model.v)

        # update parameters
        model.v = optimizer.step(model.v, grads)

        # print current loss
        if (epoch+1) % 10 == 0:
            print(f'Epoch: {epoch + 1:2d} --- Loss: {ivy.to_numpy(loss).item():.5f}')
            
    print('Finished training!')

    model_uri = tmp_path
    model_file = os.path.join(model_uri, IVY_MODEL)
    model_weights = os.path.join(model_uri, IVY_WEIGHTS)
    
    model.v.cont_to_disk_as_pickled(model_weights)
    
    code = """
    import ivy

    class Regressor(ivy.Module):
        def __init__(self, input_dim, output_dim, is_training=True):
            self.linear = ivy.Linear(input_dim, output_dim)
            self.dropout = ivy.Dropout(0.5, training=is_training)
            ivy.Module.__init__(self)

        def _forward(self, x):
            x = ivy.sigmoid(self.linear(x))
            x = self.dropout(x)
            return x

    input_dim = 3
    output_dim = 1
    backend = 'torch'
    """
    with open(model_file, "w") as file:
        file.write(code)

    return model_uri


@pytest.fixture
def model_settings(model_uri: str) -> ModelSettings:
    return ModelSettings(
        name="ivy-model",
        implementation=IvyModel,
        parameters=ModelParameters(uri=model_uri, version="v1.2.3"),
    )


@pytest.fixture
async def model(model_settings: ModelSettings) -> IvyModel:
    model = IvyModel(model_settings)
    model.ready = await model.load()

    return model


@pytest.fixture
def inference_request() -> InferenceRequest:
    payload_path = os.path.join(TESTDATA_PATH, "inference-request.json")
    return InferenceRequest.parse_file(payload_path)

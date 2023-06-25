# Serving a custom model

The `mlserver` package comes with inference runtime implementations for `scikit-learn` and `xgboost` models.
However, some times we may also need to roll out our own inference server, with custom logic to perform inference.
To support this scenario, MLServer makes it really easy to create your own extensions, which can then be containerised and deployed in a production environment.

## Overview

In this example, we will train a [`ivy` model](https://unify.ai/). 
`Ivy` unifies all ML frameworks 💥 enabling you not only to write code that can be used with any of these frameworks as the backend, but also to convert 🔄 any function, model or library written in any of them to your preferred framework!

Out of the box, `mlserver` doesn't provide an inference runtime for `ivy`.
However, through this example we will see how easy is to develop our own.

## Training

The first step will be to train our model.
This will be a very simple feedforward neural network model, based on an example provided in the [`ivy` readme](https://github.com/unifyai/ivy#ivy-as-a-framework).

```python
# Original source code and more details can be found in:
# https://github.com/unifyai/ivy#ivy-as-a-framework


import ivy

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
```

### Saving our trained model

Now that we have _trained_ our model, the next step will be to save it so that it can be loaded afterwards at serving-time.
We will need to save both the weights and model code.

This will get saved in a `artifacts` folder.

```python
import os

model_dir = "./artifacts"
ivy_model = "weights.pkl"
model_file = os.path.join(model_dir, ivy_model)

if not os.path.exists(model_dir): os.makedirs(model_dir)
model.v.cont_to_disk_as_pickled(model_file)
```
```python
import ivy

class Regressor(ivy.Module):
    def __init__(self, input_dim, output_dim, is_training=True):
        self.linear = ivy.Linear(input_dim, output_dim)
        self.dropout = ivy.Dropout(0.5, training=is_training)
        ivy.Module.__init__(self)

    def _forward(self, x, ):
        x = ivy.sigmoid(self.linear(x))
        x = self.dropout(x)
        return x

input_dim = 3
output_dim = 1
backend = 'torch'
```

## Serving

The next step will be to serve our model using `mlserver`.
For that, we will first implement an extension which serve as the _runtime_ to perform inference using our custom `ivy` model.

### Custom inference runtime

Our custom inference wrapper should be responsible of:

- Loading the model we saved previously.
- Running inference using our loaded model.

```python
# %load models.py
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
```

### Settings files

The next step will be to create 2 configuration files:

- `settings.json`: holds the configuration of our server (e.g. ports, log level, etc.).
- `model-settings.json`: holds the configuration of our model (e.g. input type, runtime to use, etc.).

#### `settings.json`

```python
# %load settings.json
{
    "debug": "true"
}

```

#### `model-settings.json`

```python
# %load model-settings.json
{
    "name": "ivy-regressor",
    "implementation": "models.IvyModel",
    "parameters": {
        "uri": "./artifacts"
    }
}

```

### Start serving our model

Now that we have our config in-place, we can start the server by running `mlserver start .`. This needs to either be ran from the same directory where our config files are or pointing to the folder where they are.

```shell
mlserver start .
```

Since this command will start the server and block the terminal, waiting for requests, this will need to be ran in the background on a separate terminal.

### Send test inference request

We now have our model being served by `mlserver`.
To make sure that everything is working as expected, let's send a request from our test set.

For that, we can use the Python types that `mlserver` provides out of box, or we can build our request manually.

```python
import requests
import numpy as np

from mlserver.types import InferenceRequest
from mlserver.codecs import NumpyCodec

x_0 = np.random.randn(4,3).astype(np.float32)
inference_request = InferenceRequest(
    inputs=[
        NumpyCodec.encode_input(name="X", payload=x_0)
    ]
)

endpoint = "http://localhost:8080/v2/models/ivy-regressor/infer"
response = requests.post(endpoint, json=inference_request.dict())

response.json()
```

## Deployment

Now that we have written and tested our custom model, the next step is to deploy it.
With that goal in mind, the rough outline of steps will be to first build a custom image containing our code, and then deploy it.

### Specifying requirements

MLServer will automatically find your requirements.txt file and install necessary python packages

```python
# %load requirements.txt
torch
ivy-core

```

### (optional) install ivy from source instead of pypi as default
extremely useful if you're looking to use cutting-edge updates in ivy that aren't updated on the pypi package yet  
make sure to comment out ivy-core from requirements.txt file!
following command will generate a Dockerfile where we can add the following bash commands to install ivy from source  
```
# Install Ivy
RUN rm -rf ivy && \
    git clone https://github.com/unifyai/ivy && \
    cd ivy && \
    cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    cat optional.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    python3 -m pip install --user -e .
```
note: this has been already done for you in this tutorial (you will find the Dockerfile already exists). feel free to remove the file if required
```bash
%%bash
mlserver dockerfile .
```

### Building a custom image

```{note}
This section expects that Docker is available and running in the background.
```

MLServer offers helpers to build a custom Docker image containing your code.
In this example, we will use the `mlserver build` subcommand to create an image, which we'll be able to deploy later.

Note that this section expects that Docker is available and running in the background, as well as a functional cluster with Seldon Core installed and some familiarity with `kubectl`.

```bash
%%bash
mlserver build . -t 'my-ivy-server:0.1.0'
```

To ensure that the image is fully functional, we can spin up a container and then send a test request. To start the container, you can run something along the following lines in a separate terminal:

```bash
docker run -it --rm -p 8080:8080 my-ivy-server:0.1.0
```

```python
import requests
import numpy as np

from mlserver.types import InferenceRequest
from mlserver.codecs import NumpyCodec

x_0 = np.random.randn(4,3).astype(np.float32)
inference_request = InferenceRequest(
    inputs=[
        NumpyCodec.encode_input(name="X", payload=x_0)
    ]
)

endpoint = "http://localhost:8080/v2/models/ivy-regressor/infer"
response = requests.post(endpoint, json=inference_request.dict())

response.json()
```

As we should be able to see, the server running within our Docker image responds as expected.

### Deploying our custom image

```{note}
This section expects access to a functional Kubernetes cluster with Seldon Core installed and some familiarity with `kubectl`.
```

Now that we've built a custom image and verified that it works as expected, we can move to the next step and deploy it.
There is a large number of tools out there to deploy images.
However, for our example, we will focus on deploying it to a cluster running [Seldon Core](https://docs.seldon.io/projects/seldon-core/en/latest/).

```{note}
Also consider that depending on your Kubernetes installation Seldon Core might expect to get the container image from a public container registry like [Docker hub](https://hub.docker.com/) or [Google Container Registry](https://cloud.google.com/container-registry). For that you need to do an extra step of pushing the container to the registry using `docker tag <image name> <container registry>/<image name>` and `docker push <container registry>/<image name>` and also updating the `image` section of the yaml file to `<container registry>/<image name>`.
```

For that, we will need to create a `SeldonDeployment` resource which instructs Seldon Core to deploy a model embedded within our custom image and compliant with the [V2 Inference Protocol](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2).
This can be achieved by _applying_ (i.e. `kubectl apply`) a `SeldonDeployment` manifest to the cluster, similar to the one below:

```python
%%writefile seldondeployment.yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: ivy-model
spec:
  protocol: v2
  predictors:
    - name: default
      graph:
        name: ivy-regressor
        type: MODEL
      componentSpecs:
        - spec:
            containers:
              - name: ivy-regressor
                image: my-ivy-server:0.1.0
```

```python

```

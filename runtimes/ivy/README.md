# Ivy runtime for MLServer

This package provides a MLServer runtime compatible with Ivy.

## Usage

You can install the runtime, alongside `mlserver`, as:

```bash
pip install mlserver mlserver-ivy
```

For further information on how to use MLServer with Ivy, you can check
out this [worked out example](../../docs/examples/ivy/README.md).

## Content Types

If no [content type](../../docs/user-guide/content-type) is present on the
request or metadata, the Ivy runtime will try to decode the payload as
a [NumPy Array](../../docs/user-guide/content-type).
To avoid this, either send a different content type explicitly, or define the
correct one as part of your [model's
metadata](../../docs/reference/model-settings).

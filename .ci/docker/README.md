## How to build image

With all possible args

```bash
docker build \
    --build-arg HTTP_PROXY="${http_proxy:?}" \
    --build-arg HTTPS_PROXY="${https_proxy:?}" \
    --build-arg NO_PROXY="${no_proxy:?}" \
    --build-arg ACTIONS_RUNNER_VER="$ACTIONS_RUNNER_VER" \
    --tag ci-runner/ov-xai:latest .
```

The simplest way using its defaults

```bash
docker build --tag ci-runner/ov-xai:latest .
```

## How to run a container

```bash
docker run -d --rm \
    --ipc=host \
    -e RUNNER_REPO_URL="https://github.com/openvinotoolkit/openvino_xai" \
    -e RUNNER_NAME="ci-runner-ov-xai" \
    -e RUNNER_LABELS="hello,labels" \
    -e RUNNER_TOKEN= \
    ci-runner/ov-xai:latest
```

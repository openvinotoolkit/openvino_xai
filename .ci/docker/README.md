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

Or, the simplest way using build script

```bash
.ci/docker$ ./build.sh lastest
```

## How to run a container

```bash
docker run -d --rm \
    --ipc=host \
    -e RUNNER_REPO_URL="https://github.com/openvinotoolkit/openvino_xai" \
    -e RUNNER_NAME="ci-runner-ov-xai" \
    -e RUNNER_LABELS="large-disk" \
    -e RUNNER_TOKEN= \
    ci-runner/ov-xai:latest
```

Or, using start script

```bash
.ci/docker$ ./start-runner.sh <gh-token>
```

Use `--help` option to see all available options to the script.

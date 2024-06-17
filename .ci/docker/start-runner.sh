#!/bin/bash

ACTIONS_RUNNER_TAG="latest"
ACTIONS_RUNNER_NAME="ov-xai-ci-runner"
LABELS="large-disk"
MOUNT_PATH=""

POSITIONAL=()

while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -n|--name)
      ACTIONS_RUNNER_NAME="$2"
      shift # past argument
      shift # past value
      ;;
    -l|--labels)
      LABELS="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--tag)
      ACTIONS_RUNNER_TAG="$2"
      shift # past argument
      shift # past value
      ;;
    -m|--mount)
      MOUNT_PATH="$2"
      shift # past argument
      shift # past value
      ;;
    -h|--help)
      DEFAULT="yes"
      break
      shift # past argument
      ;;
    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters

if [ "$#" -lt 1 ] || [ "$DEFAULT" == "yes" ]; then
cat << EndofMessage
    USAGE: $0 <gh-token> [Options]
    Positional args
        <gh-token>          Github token string
    Options
        -n|--name           Specify actions-runner name to be registered to the repository
        -l|--labels         Additional label string to set the actions-runner
        -t|--tag            Specify docker image tag to create container
        -m|--mount          Specify data path on the host machine to mount to the runner container
        -h|--help           Print this message
EndofMessage
exit 0
fi

ENV_FLAGS=""
MOUNT_FLAGS=""

if [ "$MOUNT_PATH" != "" ]; then
  echo "mount path option = $MOUNT_PATH"
    ENV_FLAGS="-e CI_DATA_ROOT=/home/cibot/data"
    MOUNT_FLAGS="-v $MOUNT_PATH:/home/cibot/data:ro"
    LABELS="$LABELS,dmount"
fi

GITHUB_TOKEN=$1

docker inspect "$ACTIONS_RUNNER_NAME"; RET=$?

if [ $RET -eq 0 ]; then
    # if the named container exsiting, stop and remove it first
    docker stop "$ACTIONS_RUNNER_NAME"
    # wait completely stopped the container
    sleep 10
    yes | docker rm "$ACTIONS_RUNNER_NAME"
fi

docker run -d --rm \
    --ipc=host \
    -e RUNNER_REPO_URL="https://github.com/openvinotoolkit/openvino_xai" \
    -e RUNNER_NAME="$ACTIONS_RUNNER_NAME" \
    -e RUNNER_LABELS="$LABELS" \
    -e RUNNER_TOKEN="$GITHUB_TOKEN" \
    ${ENV_FLAGS} \
    ${MOUNT_FLAGS} \
    --name "$ACTIONS_RUNNER_NAME" \
    ci-runner/ov-xai:"$ACTIONS_RUNNER_TAG"; RET=$?

if [ $RET -ne 0 ]; then
    echo "failed to start ci container. $RET"
    exit 1
fi

echo "Successfully started ci container - $ACTIONS_RUNNER_NAME"

#!/bin/bash

ACTIONS_RUNNER_TAG="latest"
ACTIONS_RUNNER_NAME="ov-xai-ci-runner"
LABELS="large-disk"

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
        -h|--help           Print this message
EndofMessage
exit 0
fi

GITHUB_TOKEN=$1

docker run -d --rm \
    --ipc=host \
    -e RUNNER_REPO_URL="https://github.com/openvinotoolkit/openvino_xai" \
    -e RUNNER_NAME="$ACTIONS_RUNNER_NAME" \
    -e RUNNER_LABELS="$LABELS" \
    -e RUNNER_TOKEN="$GITHUB_TOKEN" \
    --name "$ACTIONS_RUNNER_NAME" \
    ci-runner/ov-xai:"$ACTIONS_RUNNER_TAG"; RET=$?

if [ $RET -ne 0 ]; then
    echo "failed to start ci container. $RET"
    exit 1
fi

echo "Successfully started ci container - $ACTIONS_RUNNER_NAME"

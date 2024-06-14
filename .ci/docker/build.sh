#!/bin/bash

ACTIONS_RUNNER_VER="2.317.0"
POSITIONAL=()

while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -v|--version)
      ACTIONS_RUNNER_VER="$2"
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
    USAGE: $0 <tag> [Options]
    Positional args
        <tag>               Tag name to be tagged to newly built image
    Options
        -v|--version        Specify actions-runner version
        -h|--help           Print this message
EndofMessage
exit 0
fi

TAG=$1

docker build \
    --build-arg HTTP_PROXY="${http_proxy:?}" \
    --build-arg HTTPS_PROXY="${https_proxy:?}" \
    --build-arg NO_PROXY="${no_proxy:?}" \
    --build-arg ACTIONS_RUNNER_VER="$ACTIONS_RUNNER_VER" \
    --tag ci-runner/ov-xai:"$TAG" .

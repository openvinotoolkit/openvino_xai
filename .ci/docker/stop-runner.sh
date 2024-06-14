#!/bin/bash
CONTAINER_NAME="ov-xai-ci-runner"
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -n|--name)
      CONTAINER_NAME="$2"
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
    USAGE: $0 <github-token> [Options]
    Options
        -n|--name           Specify container ID or name to be stopped
        -h|--help           Print this message
EndofMessage
exit 0
fi

GITHUB_TOKEN=$1

echo "stopping $CONTAINER_NAME..."

docker inspect "$CONTAINER_NAME"; RET=$?

if [ $RET -eq 0 ]; then
    docker exec -it "$CONTAINER_NAME" bash -c \
        "./config.sh remove \
        --token $GITHUB_TOKEN" ; RET=$?

    if [ $RET -ne 0 ]; then
        echo "failed to stop the runner. $RET"
        exit 1
    fi
else
    echo "cannot find running $CONTAINER_NAME container"
    exit 1
fi

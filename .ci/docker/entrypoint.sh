#!/bin/bash

if [ -z "${RUNNER_NAME}" ] || [ -z "${RUNNER_LABELS}" ] || [ -z "${RUNNER_TOKEN}" ] || [ -z "${RUNNER_REPO_URL}" ]; then
    echo "Missing one or more required environment variables."
    echo "repo url(${RUNNER_REPO_URL}), name(${RUNNER_NAME}), labels(${RUNNER_LABELS}), and token are required."
    exit 1
fi

./config.sh --url ${RUNNER_REPO_URL} \
--unattended \
--replace \
--labels ${RUNNER_LABELS} \
--name ${RUNNER_NAME} \
--token ${RUNNER_TOKEN}

unset RUNNER_LABELS RUNNER_NAME RUNNER_TOKEN RUNNER_REPO_URL

./run.sh
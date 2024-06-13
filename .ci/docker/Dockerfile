# need to check whether ubuntu22.04 is supported operation system for the self-hosted runners
# https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners#linux
FROM ubuntu:20.04@sha256:0b897358ff6624825fb50d20ffb605ab0eaea77ced0adb8c6a4b756513dec6fc
LABEL maintainer="OpenVINO XAI Toolkit Development Team"

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# Setup proxies
ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV no_proxy=$NO_PROXY

# to ignore interactive setup while running 'apt-get install'
ENV DEBIAN_FRONTEND="noninteractive"

# hadolint ignore=DL3008
RUN apt-get update && apt-get install --no-install-recommends -y \
        software-properties-common \
        curl \
        unzip \
        cron \
        ffmpeg \
        libpython3.10 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# set /tmp folder cleaning schedule at 7PM every day which is older than a day
RUN echo "0 19 * * * find /tmp/* -mtime +1 -exec rm -rf {} \;" >> ./cron_clean_tmp.txt && \
    crontab cron_clean_tmp.txt && \
    # create a non-root user to run gh actions-runner
    adduser --disabled-login cibot


COPY entrypoint.sh /home/cibot
RUN chown cibot:cibot /home/cibot/entrypoint.sh
RUN chmod +x /home/cibot/entrypoint.sh

USER cibot
WORKDIR /home/cibot

RUN mkdir actions-runner && mv entrypoint.sh actions-runner
WORKDIR /home/cibot/actions-runner

ARG ACTIONS_RUNNER_VER=2.317.0
RUN curl -o actions-runner.tar.gz -L https://github.com/actions/runner/releases/download/v${ACTIONS_RUNNER_VER}/actions-runner-linux-x64-${ACTIONS_RUNNER_VER}.tar.gz && \
    tar xzf ./actions-runner.tar.gz && rm actions-runner.tar.gz

ENTRYPOINT [ "/bin/bash", "-c" ]

CMD ["/home/cibot/actions-runner/entrypoint.sh"]

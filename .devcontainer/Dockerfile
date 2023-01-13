FROM nvcr.io/nvidia/pytorch:20.12-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get -y install --no-install-recommends apt-utils dialog 2>&1
RUN apt-get -y install --no-install-recommends \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
        openssh-client \
        software-properties-common \
        shellcheck \
        sudo \
        vim \
        zsh
RUN apt-get autoremove -y
RUN apt-get clean -y
RUN rm -rf /var/lib/apt/lists/*

# Install hadolint.
RUN wget -O /usr/bin/hadolint https://github.com/hadolint/hadolint/releases/download/v1.16.3/hadolint-Linux-x86_64 && \
    chmod +x /usr/bin/hadolint

# Configure users
ARG USERNAME="user"
ARG USER_UID="1005"
ARG USER_GID="${USER_UID}"
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd -s /usr/bin/zsh --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} && \
    echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}" && \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"

ENV DEBIAN_FRONTEND=dialog


FROM nvcr.io/nvidia/isaac-lab:2.3.2

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    iputils-ping iputils-tracepath traceroute \
    net-tools iproute2 \
    dnsutils curl wget ca-certificates \
    nodejs npm \
    && npm install -g @anthropic-ai/claude-code \
    && curl -L https://github.com/Sarfflow/rtunnel/releases/download/v1.0.0/rtunnel-linux -o /usr/local/bin/fkqz \
    && chmod +x /usr/local/bin/fkqz \
    && rm -rf /var/lib/apt/lists/*

RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip config set global.index-url http://nexus.sii.shaipower.online/repository/pypi/simple \
    && /workspace/isaaclab/_isaac_sim/python.sh -m pip config set global.trusted-host nexus.sii.shaipower.online \
    && printf '%s\n' \
        'Types: deb' \
        'URIs: http://nexus.sii.shaipower.online/repository/ubuntu/' \
        'Suites: noble noble-updates noble-backports' \
        'Components: main universe restricted multiverse' \
        'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
        '' \
        'Types: deb' \
        'URIs: http://nexus.sii.shaipower.online/repository/ubuntu/' \
        'Suites: noble-security' \
        'Components: main universe restricted multiverse' \
        'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
        > /etc/apt/sources.list.d/ubuntu.sources

RUN printf 'root: \n' | chpasswd \
    && printf '%s\n' \
        'PermitRootLogin yes' \
        'PasswordAuthentication yes' \
        'KbdInteractiveAuthentication yes' \
        'UsePAM yes' \
        > /etc/ssh/sshd_config.d/99-root-password-login.conf

FROM dudekw/siu-20.04
ARG WORKSPACE_FOLDER=/workspace
WORKDIR "/workspace"

ENV DEBIAN_FRONTEND=noninteractive
RUN rm /etc/apt/sources.list.d/google-chrome.list
RUN apt-get update
RUN apt-get upgrade -qy
RUN apt-get install -qy --no-install-recommends build-essential curl python3 python3-dev python3-pip

COPY ./ ./

RUN ln -fs "$WORKSPACE_FOLDER/roads.png" /roads.png

RUN pip install -r requirements.cpu_only.txt
RUN pip install -U flask

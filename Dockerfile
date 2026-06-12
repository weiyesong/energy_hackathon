FROM mambaorg/micromamba:1.5.10

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ARG USERNAME=app
ARG USER_UID=1000
ARG USER_GID=1000

USER root

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        tini \
        vim-tiny \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes

WORKDIR /workspace
COPY --chown=${USERNAME}:${USERNAME} . /workspace

RUN mkdir -p /workspace/data/raw /workspace/data/processed /workspace/data/output /workspace/notebooks \
    && chown -R ${USERNAME}:${USERNAME} /workspace

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR \
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.nc,.h5,.hdf,.grib,.grib2,.zarr" \
    PROJ_NETWORK=OFF

USER ${USERNAME}
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]

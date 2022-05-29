FROM quay.io/donchesworth/rapids-dask-pytorch:py38-cuda10.2-rapids21.10-pytorch1.9-ubi8

# Labels
LABEL maintainer="Don Chesworth<donald.chesworth@gmail.com>"
LABEL org.label-schema.schema-version="0.0"
LABEL org.label-schema.name="hugging-quik-test"
LABEL org.label-schema.description="Helper package to make loading/unloading huggingface models quik-er"

RUN pip install matplotlib sklearn

# Project installs
WORKDIR /opt/hq
COPY ./ /opt/hq/
RUN pip install .

RUN chgrp -R 0 /opt/hq/ && \
    chmod -R g+rwX /opt/hq/ && \
    chmod +x /opt/hq/entrypoint.sh

ENTRYPOINT ["/opt/hq/entrypoint.sh"]

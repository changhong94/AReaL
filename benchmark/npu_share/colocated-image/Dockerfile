ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG BENCH_PYTHON
COPY awex-source /opt/awex-ascend-src
# Editable installation preserves upstream subpackages omitted by its wheel.
# This installs no dependencies and does not replace torch/vLLM/CANN.
RUN ${BENCH_PYTHON} -m pip install --no-deps --no-build-isolation -e /opt/awex-ascend-src
ARG AWEX_COMMIT
LABEL areal.awex.commit=${AWEX_COMMIT}

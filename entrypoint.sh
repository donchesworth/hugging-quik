#!/bin/sh

cd /opt/pq
source ./.env
# pytest
pytest --cov=/opt/hq/hugging_quik --cov-config=.coveragerc
echo $(xmllint --xpath "string(//coverage/@line-rate)" coverage_cpu.xml)
# curl https://codecov.io/bash

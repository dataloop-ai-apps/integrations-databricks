FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.10_opencv

RUN pip install --user \
    databricks-api \
    databricks-sql-connector \
    databricks-sdk \
    pandas \
    dtlpy

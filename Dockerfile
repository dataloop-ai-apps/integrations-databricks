
FROM dataloopai/dtlpy-agent:cpu.py3.10.opencv

RUN pip install --user \
    databricks-api \
    databricks-sql-connector \
    databricks-sdk \
    pandas \
    dtlpy

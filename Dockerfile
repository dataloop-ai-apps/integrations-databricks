
FROM dataloopai/dtlpy-agent:cpu.py3.10.opencv

COPY requirements.txt .
RUN pip install --user -r requirements.txt
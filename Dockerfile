FROM python:3.8-slim-bullseye

# --------------------------------------------------------------------------------------------
# Install dependencies
# for 'pgrep', useful when joblib needs to kill children processes cleanly with loky backend
RUN apt-get update && apt-get install procps -y

# Python dependencies
COPY requirements.txt /tmp/
RUN pip install --upgrade pip
RUN pip3 install -r /tmp/requirements.txt

# --------------------------------------------------------------------------------------------
ADD descriptor.json /app/descriptor.json
ADD run.py /app/run.py

ENTRYPOINT ["python", "/app/run.py"]

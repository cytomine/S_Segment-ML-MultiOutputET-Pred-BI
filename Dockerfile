FROM python:3.6.9-stretch

# --------------------------------------------------------------------------------------------
# Install Cytomine python client
RUN git clone https://github.com/cytomine-uliege/Cytomine-python-client.git && \
    cd /Cytomine-python-client && \
# git checkout tags/v2.7.1 &&
    pip install . && \
    rm -r /Cytomine-python-client

# --------------------------------------------------------------------------------------------
# Instal Pyxit, Skimage and SLDC
RUN pip install pyxit==1.1.5 scikit-image==0.17.2 sldc==1.2.4 sldc-cytomine==1.2.2

# --------------------------------------------------------------------------------------------
# Instal Pyxit
ADD descriptor.json /app/descriptor.json
ADD run.py /app/run.py

ENTRYPOINT ["python", "/app/run.py"]

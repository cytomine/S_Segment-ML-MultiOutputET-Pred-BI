FROM cytomine/software-python3-base:v2.2.2

# --------------------------------------------------------------------------------------------
# Instal Pyxit, Skimage and SLDC
RUN pip install pyxit==1.1.5 scikit-image==0.17.2 sldc==1.2.4 #sldc-cytomine==1.3.0

# --------------------------------------------------------------------------------------------
# Install custom SLDC-Cytomine
ADD image_adapter.py /image_adapter.py

RUN git clone https://github.com/waliens/sldc-cytomine && cd /sldc-cytomine && git checkout f1be4611192a5c6d40c04b09c6bab6cc01adbed8 && mv /image_adapter.py /sldc-cytomine/sldc_cytomine/image_adapter.py && python setup.py build && python setup.py install

ADD descriptor.json /app/descriptor.json
ADD run.py /app/run.py

ENTRYPOINT ["python", "/app/run.py"]

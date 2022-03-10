FROM cytomine/software-python3-base:v2.2.2

# --------------------------------------------------------------------------------------------
# Instal Pyxit, Skimage and SLDC
RUN pip install pyxit==1.1.5 scikit-image==0.17.2 sldc==1.4.1 sldc-cytomine==1.3.3

# --------------------------------------------------------------------------------------------
ADD descriptor.json /app/descriptor.json
ADD run.py /app/run.py

ENTRYPOINT ["python", "/app/run.py"]

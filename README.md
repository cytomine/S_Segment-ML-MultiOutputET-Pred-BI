# Cytomine Software - Segment-ML-MultiOutputET-Pred-BI

Cytomine (https://cytomine.org) app developed by ULiège Cytomine Research team (https://uliege.cytomine.org) for segmentation of samples using a binary segmentation (ML) model (see [S_Segment-ML-MultiOutputET-Pred-BI](https://github.com/cytomine/S_Segment-ML-MultiOutputET-Pred-BI)).

This implementation follows Cytomine (> v3.0) external app conventions based on container technology.

* **Summary:** It uses a binary segmentation model based on subwindow and multiple output extra-trees to segment an image.

To launch such an analysis, a user first specifies the algorithm that will be applied on the image (Segment-ML-MultiOutputET-Pred-BI) but also the model on which the segmentation is based (in this case, Segment-ML-MultiOutputET-Train).

* **Typical application:** Predict the regions of interest from chosen areas that corresponds to a certain term (*e.g.* tumor regions in histology slides).

* **Based on:** pixel classification model to detect regions of interest, the methodology is presented [here](https://ieeexplore.ieee.org/document/6868017).


* **Parameters:** 
  * *cytomine_host*, *cytomine_public_key*, *cytomine_private_key*, *cytomine_id_project*, *cytomine_id_images* and *cytomine_id_software* are parameters needed for Cytomine external app. They will allow the app to be run on the Cytomine instance (determined with its host), connect and communicate with the instance (with the key pair). An app is always run into a project (*cytomine_id_project*) and to be run, the app must be previously declared to the plateform (*cytomine_id_software*). Note that if *cytomine_id_images* is not specified, the segmentation will be applied on all images from the project.
  * *cytomine_id_roi_term* : the term corresponding to the region(s) of interest (ROI) on which the segmentation is run and annotations will be created. If not specified, the segmentation process is applied on the entire selected images.
  * *cytomine_id_predict_term* : The detected components will be associated to a term corresponding to the given id. If not specified, no term will be associated to the created annotation(s).
  * *cytomine_reviewed_roi* : if set to *True*, the prediction algorithm will fetch only reviewed annotations.
  * *cytomine_zoom_level* : the zoom level at which the algorithm will be run.
  * *cytomine_tile_size* : The choosen tile size. It has to be equal to 256 if the zoom level is different from 0.
  * *cytomine_tile_overlap* : the choosen size of the overlap between slides. It has to be 0 if the zoom level is different from 0.
  * *n_jobs* : the choosen number of threads to execute the algoSrithm.
  * *union_enabled* : if set to *True*, annotations created by the algorithm that are adjacent are joined together and only one annotation will be considered.
  * *min_annotation_area* : the minimum area of the annotations created by the prediction algorithm.
  * *max_annotation_area* : the maximum area of the annotations created by the prediction algorithm. It is equal to -1 if there is no maximum area.
  * *tile_filter_min_stddev* : minimum standard deviation for tile filtering.
  * *tile_filter_max_mean*: maximum mean for tile filtering.
  * *pyxit_prediction_step*: number of steps before computing the prediction output.
  * Finally, you can modify the *log_level* by setting one of these values 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL' or unset.

-----------------------------------------------------------------------------

Copyright 2010-2022 University of Liège, Belgium, https://uliege.cytomine.org

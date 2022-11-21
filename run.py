import os
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

import joblib
import numpy as np
import pyxit
from cytomine import CytomineJob
from cytomine.models.collection import CollectionPartialUploadException
from cytomine.models import Annotation, ImageInstance, Job, PropertyCollection
from cytomine.models import AnnotationCollection, ImageInstanceCollection, AttachedFileCollection
from cytomine.utilities.software import parse_domain_list, str2bool
from shapely import wkt
from shapely.affinity import affine_transform
from shapely.geometry import GeometryCollection, Polygon
from skimage.color import rgb2gray, rgb2hsv
from skimage.util.shape import view_as_windows
from sldc.locator import flatten_geoms
from sldc import (Logger, SemanticSegmenter, SSLWorkflowBuilder,
                  StandardOutputLogger, TileBuilder, TileTopology)
from sldc_cytomine import CytomineTileBuilder
from sldc_cytomine.dump import load_region_tiles
from sldc_cytomine.autodetect import infer_protocols


def flatten(_polygon):
  if not hasattr(_polygon, "geoms"):
    if _polygon.area > 0:
      return [_polygon]
    else: 
      return []
  _out = []
  for geom in _polygon.geoms:
    _out.extend(flatten(geom))
  return _out


def extract_windows_and_identifiers(image, identifiers, dims, step):
    window_n_pixels = np.product(dims)
    subwindows = view_as_windows(image, dims, step=step)
    subwindows = subwindows.reshape([-1, window_n_pixels])
    window_ids = view_as_windows(identifiers, dims[:2], step=step)
    return subwindows, window_ids[:, :, 0, 0].reshape([-1])


def extract_windows(image, dims, step):
    """Return a set of windows over the image where all pixels are covered at least once"""
    n_pixels = int(np.prod(image.shape[:2]))
    identifiers = np.arange(n_pixels).reshape(image.shape[:2])
    subwindows, subwindows_ids = extract_windows_and_identifiers(image, identifiers, dims, step)

    # add missing windows
    fits_horizontally = (image.shape[1] - (dims[1] - step)) % step == 0 
    fits_vertically = (image.shape[0] - (dims[0] - step)) % step == 0
    y_start, x_start  = -dims[0], -dims[1]
    if not fits_horizontally:
        w_right, id_right = extract_windows_and_identifiers(image[:, x_start:], identifiers[:, x_start:], dims, step)
        subwindows = np.vstack([subwindows, w_right])
        subwindows_ids = np.hstack([subwindows_ids, id_right])
    if not fits_vertically:
        w_bottom, id_bottom = extract_windows_and_identifiers(image[y_start:], identifiers[y_start:], dims, step)
        subwindows = np.vstack([subwindows, w_bottom])
        subwindows_ids = np.hstack([subwindows_ids, id_bottom])
    if not (fits_horizontally or fits_vertically): 
        w_bottom_right = np.array([image[y_start:, x_start:].flatten()])
        id_bottom_right = np.array([identifiers[y_start, x_start]])
        subwindows = np.vstack([subwindows, w_bottom_right])
        subwindows_ids = np.hstack([subwindows_ids, id_bottom_right])

    return subwindows, subwindows_ids


def count_pred_per_pixel(step, sw_height, sw_width):
	return TileTopology.tile_count_1d(sw_height, step) * TileTopology.tile_count_1d(sw_width, step)


def determine_best_step(target_pred_per_pxl, sw_height, sw_width):
	"""Determine the best prediction step for a given number of requested predictions per pixel.
	Take the largest step that generates the requested number of predictions per pixel (or slightly more if no exact match).
	"""
	prev_step = None
	# look for best step
	for step in np.arange(1, min(sw_height, sw_width) + 1): 
		pred_per_pxl = count_pred_per_pixel(step, sw_height, sw_width)
		if pred_per_pxl < target_pred_per_pxl:
			return prev_step
		prev_step = step

	return prev_step


def change_referential(p, offset=None, zoom_level=0, height=None):
  """
  Parameters
  ----------
  p: Polygon
    A shapely polygon
  offset: tuple
    (x, y) an offest to apply to the polygon
  height: int
    The height of the region containing the polygon (post-offset) 
  zoom_level: float
    Scale factor (exponent)
  """
  if offset is not None:
    p = affine_transform(p, [1, 0, 0, 1, -offset[0], -offset[1]])
  if height is not None:
    p = affine_transform(p, [1, 0, 0, -1, 0, height])
  if zoom_level != 0:
    p = affine_transform(p, [1 / 2 ** zoom_level, 0, 0, 1 / 2 ** zoom_level, 0, 0])
  return p


class ExtraTreesSegmenter(SemanticSegmenter):
    def __init__(self, pyxit, classes=None, background=0, min_std=0, max_mean=255, prediction_step=1):
        super(ExtraTreesSegmenter, self).__init__(classes=classes)
        self._pyxit = pyxit
        self._prediction_step = prediction_step
        self._min_std = min_std
        self._max_mean = max_mean
        self._background = background

    def _process_tile(self, image):
        channels = [image]
        if image.ndim > 2:
            channels = [image[:, :, i] for i in range(image.shape[2])]
        return np.any([
            np.std(c) > self._min_std or np.mean(c) < self._max_mean
            for c in channels
        ])

    def _convert_colorspace(self, image):
        colorspace = self._pyxit.colorspace
        if colorspace == pyxit.estimator.COLORSPACE_GRAY:
            return rgb2gray(image)
        elif colorspace == pyxit.estimator.COLORSPACE_RGB:
            return image
        elif colorspace == pyxit.estimator.COLORSPACE_HSV:
            return rgb2hsv(image)
        else:
            raise ValueError("unsupported color space use HSV, Gray or RGB")

    def segment(self, image):
        # extract mask
        mask = np.ones(image.shape[:2], dtype="bool")
        if image.ndim == 3 and image.shape[2] == 2 or image.shape[2] == 4:
            mask = image[:, :, -1].astype("bool")
            image = np.copy(image[:, :, :-1])  # remove mask from image

        # skip processing if tile is supposed background (checked via mean & std) or not in the mask
        if not (self._process_tile(image) and np.any(mask)):
            return np.full(image.shape[:2], self._background)

        # change colorspace
        image = self._convert_colorspace(image).reshape(image.shape)

        # prepare windows
        target_height = self._pyxit.target_height
        target_width = self._pyxit.target_width
        w_dims = [target_height, target_width]
        if image.ndim > 2 and image.shape[2] > 1:
            w_dims += [image.shape[2]]
        subwindows, w_identifiers = extract_windows(image, w_dims, self._prediction_step)

        # predict
        y = np.array(self._pyxit.base_estimator.predict_proba(subwindows))

        cm_dims = list(image.shape[:2]) + [self._pyxit.n_classes_]
        confidence_map = np.zeros(cm_dims, dtype="float")
        pred_count_map = np.zeros(cm_dims[:2], dtype="int32")

        for row, w_index in enumerate(w_identifiers):
            im_width = image.shape[1]
            pred_dims = [target_height, target_width, self._pyxit.n_classes_]
            x_off, y_off = w_index % im_width, w_index // im_width
            confidence_map[y_off:(y_off+target_height), x_off:(x_off+target_width)] += y[:, row, :].reshape(pred_dims)
            pred_count_map[y_off:(y_off+target_height), x_off:(x_off+target_width)] += 1

        # average over multiple predictions
        confidence_map /= np.expand_dims(pred_count_map, axis=2)

        # remove class where there is no mask
        class_map = np.take(self._pyxit.classes_, np.argmax(confidence_map, axis=2))
        class_map[np.logical_not(mask)] = self._background
        return class_map


class AnnotationAreaChecker(object):
    def __init__(self, min_area, max_area):
        self._min_area = min_area
        self._max_area = max_area

    def check(self, annot):
        min_area = self._min_area
        max_area = self._max_area
        if max_area < 0:
            return min_area < annot.area
        else:
            return min_area < (annot.area) < max_area


def validate_train_job(job: Job, properties: dict):
    """Check whether the selected training job is valid."""
    if job.status != Job.SUCCESS:
        raise ValueError(f"training job {job.id} has not successfully terminated its execution")
    if "binary" not in properties:
        raise ValueError(f"job {job.id} is missing the 'binary' property, maybe it is not a training job ?")
    if "classes" not in properties:
        raise ValueError(f"job {job.id} is missing the 'classes' property, maybe it is not a training job ?")


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        import warnings
        warnings.filterwarnings("error", category=RuntimeWarning)
        # use only images from the current project
        cj.job.update(progress=1, statusComment="Preparing execution (creating folders,...).")
        root_path = str(Path.home())
        working_path = os.path.join(root_path, "images")
        os.makedirs(working_path, exist_ok=True)

        # load training information
        cj.job.update(progress=5, statusComment="Extract properties from training job.")
        train_job = Job().fetch(cj.parameters.cytomine_id_job)
        if not train_job:
          raise ValueError(f"cannot retrieve training job {cj.parameters.cytomine_id_job}")
        properties = PropertyCollection(train_job).fetch().as_dict()
        validate_train_job(train_job, properties)
        binary = str2bool(properties["binary"].value)
        classes = parse_domain_list(properties["classes"].value)

        cj.job.update(progress=10, statusComment="Download the model file.")
        attached_files = AttachedFileCollection(train_job).fetch()
        model_file = attached_files.find_by_attribute("filename", "model.joblib")
        model_filepath = os.path.join(root_path, "model.joblib")
        model_file.download(model_filepath, override=True)
        pyxit = joblib.load(model_filepath)

        # set n_jobs
        pyxit.base_estimator.n_jobs = cj.parameters.n_jobs
        pyxit.n_jobs = cj.parameters.n_jobs

        cj.job.update(progress=25, statusComment="Identify regions to process.")
        images = ImageInstanceCollection()
        if cj.parameters.cytomine_id_images is not None:
            id_images = parse_domain_list(cj.parameters.cytomine_id_images)
            images.extend([ImageInstance().fetch(_id) for _id in id_images])
        else:
            images = images.fetch_with_filter("project", cj.project.id)
        
        if len(images) == 0:
            raise ValueError("no image found for processing")

        # so that the software works with different Cytomine core & ims versions
        sldc_slide_class, sldc_tile_class = infer_protocols(images[0])
        zoom_level = cj.parameters.cytomine_zoom_level
        sldc_slides = [sldc_slide_class(img, zoom_level) for img in images]
        sldc_slides_map = {slide.image_instance.id: slide for slide in sldc_slides}

        # fetch ROI annotations, all users and algo
        rois_fetch_params = { "terms": [cj.parameters.cytomine_id_roi_term], "project": cj.project.id, "showWKT": True }
        rois = AnnotationCollection(**rois_fetch_params, includeAlgo=True).fetch()
        kept_rois = list()
        
        regions = list()
        for roi in rois:
            if roi.image not in sldc_slides_map:
                continue
            slide = sldc_slides_map[roi.image]
            roi_polygon = change_referential(wkt.loads(roi.location), zoom_level=zoom_level, height=slide.image_instance.height)
            regions.append(slide.window_from_polygon(roi_polygon, mask=True))
            kept_rois.append(roi)

        if len(regions) == 0:
            raise ValueError(f"no regions found with ROI term identifier {cj.parameters.cytomine_id_roi_term}")

        cj.job.update(progress=45, statusComment="Build workflow.")
        builder = SSLWorkflowBuilder()
        builder.set_tile_size(cj.parameters.cytomine_tile_size, cj.parameters.cytomine_tile_size)
        builder.set_overlap(cj.parameters.cytomine_tile_overlap)
        tile_builder = CytomineTileBuilder(working_path, tile_class=sldc_tile_class, n_jobs=1)
        builder.set_tile_builder(tile_builder)
        builder.set_logger(StandardOutputLogger(level=Logger.INFO))
        builder.set_n_jobs(joblib.cpu_count() if cj.parameters.n_jobs < 0 else max(1, cj.parameters.n_jobs))
        builder.set_background_class(0)
        builder.set_distance_tolerance(1) # only merge overlapping polygons  

        # determine prediction step
        pyxit_prediction_step = determine_best_step(
            cj.parameters.pyxit_predictions_per_pixel, 
            pyxit.target_height, 
            pyxit.target_width
        )

        builder.set_segmenter(ExtraTreesSegmenter(
            pyxit=pyxit,
            classes=classes,
            prediction_step=pyxit_prediction_step,
            background=0,
            min_std=cj.parameters.tile_filter_min_stddev,
            max_mean=cj.parameters.tile_filter_max_mean
        ))
        workflow = builder.get()

        min_area = cj.parameters.min_annotation_area 
        max_area = cj.parameters.max_annotation_area
        area_checker = AnnotationAreaChecker(
            min_area=0 if min_area is None else min_area,
            max_area=-1 if max_area is None else max_area
        )
        
        def get_term(label):
            if binary:
                if "cytomine_id_predict_term" not in cj.parameters or not cj.parameters.cytomine_id_predict_term:
                    return []
                else:
                    return [int(cj.parameters.cytomine_id_predict_term)]
            # multi-class
            return [int(label)]

        for roi, region in cj.monitor(zip(kept_rois, regions), start=50, end=90, period=0.05, prefix="Segmenting images/ROIs"):
            # pre-download tiles and delete tile after the region has been processed
            with TemporaryDirectory() as tmpdir:
                load_region_tiles(
                    roi, tmpdir,
                    zoom_level=zoom_level,
                    slide_class=sldc_slide_class,
                    tile_class=sldc_tile_class,
                    n_jobs=0 if cj.parameters.n_jobs < 0 else max(1, cj.parameters.n_jobs))
                tile_builder._working_path = tmpdir
                results = workflow.process(region)
                
            annotations = AnnotationCollection()
            for obj in results:

                # move back to max zoom, whole image and lower corner referential
                polygon = obj.polygon.intersection(region.polygon_mask)
                polygon = change_referential(polygon, offset=[-region.offset[0], -region.offset[1]]) # at provided zoom level 
                polygon = change_referential(polygon, zoom_level=-zoom_level)
                polygon = change_referential(polygon, height=region.base_image.image_instance.height) # at zoom 0
                
                for geom in flatten(polygon):
                    if not area_checker.check(geom):
                        continue
                    annotations.append(Annotation(
                        location=geom.wkt,
                        id_terms=get_term(obj.label),
                        id_project=cj.project.id,
                        id_image=region.base_image.image_instance.id
                    ))

            annotations.save(n_workers=0 if cj.parameters.n_jobs < 0 else max(1, cj.parameters.n_jobs))


        cj.job.update(status=Job.TERMINATED, status_comment="Finish", progress=100)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

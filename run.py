import os
import joblib
import numpy as np
from pathlib import Path

from pyxit.estimator import COLORSPACE_RGB, COLORSPACE_TRGB, COLORSPACE_HSV, COLORSPACE_GRAY, _raw_to_trgb, _raw_to_hsv
import shapely
from shapely import wkt
from shapely.affinity import affine_transform, translate
from skimage.util.shape import view_as_windows

from cytomine import CytomineJob, Cytomine
from cytomine.models import ImageInstanceCollection, ImageInstance, AttachedFileCollection, Job, PropertyCollection, \
    AnnotationCollection, Annotation, JobCollection
from cytomine.utilities.software import parse_domain_list, str2bool
from sldc import SemanticSegmenter, SSLWorkflowBuilder, StandardOutputLogger, Logger, ImageWindow, Image
from sldc_cytomine import CytomineTileBuilder
from sldc_cytomine.image_adapter import CytomineDownloadableTile


class CytomineSlide(Image):
    """
    A slide from a cytomine project
    """

    def __init__(self, img_instance, zoom_level=0):
        """Construct CytomineSlide objects

        Parameters
        ----------
        img_instance: ImageInstance
            The the image instance
        zoom_level: int
            The zoom level at which the slide must be read. The maximum zoom level is 0 (most zoomed in). The greater
            the value, the lower the zoom.
        """
        self._img_instance = img_instance
        self._zoom_level = zoom_level

    @classmethod
    def from_id(cls, id_img_instance, zoom_level=0):
        return cls(ImageInstance.fetch(id_img_instance), zoom_level=zoom_level)

    @property
    def image_instance(self):
        return self._img_instance

    @property
    def np_image(self):
        raise NotImplementedError("Disabled due to the too heavy size of the images")

    @property
    def width(self):
        return self._img_instance.width // (2 ** self.zoom_level)

    @property
    def height(self):
        return self._img_instance.height // (2 ** self.zoom_level)

    @property
    def channels(self):
        return 3

    @property
    def zoom_level(self):
        return self._zoom_level

    @property
    def api_zoom_level(self):
        """The zoom level used by cytomine api uses 0 as lower level of zoom (most zoomed out). This property
        returns a zoom value that can be used to communicate with the backend."""
        return self._img_instance.depth - self.zoom_level

    def __str__(self):
        return "CytomineSlide (#{}) ({} x {}) (zoom: {})".format(self._img_instance.id, self.width, self.height, self.zoom_level)


def extract_windows(image, dims, step):
    # subwindows on input image
    subwindows = view_as_windows(image, dims, step=step)
    subwindows = subwindows.reshape([-1, np.product(dims)])
    # generate tile identifierss
    n_pixels = int(np.prod(image.shape[:2]))
    window_ids = np.arange(n_pixels).reshape(image.shape[:2])
    identifiers = view_as_windows(window_ids, dims[:2], step=step)
    identifiers = identifiers[:, :, 0, 0].reshape([-1])
    return subwindows, identifiers


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
        flattened = image.reshape([-1] if image.ndim == 2 else [-1, image.shape[2]])
        if colorspace == COLORSPACE_RGB:
            return image
        elif colorspace == COLORSPACE_TRGB:
            return _raw_to_trgb(flattened).reshape(image.shape)
        elif colorspace == COLORSPACE_HSV:
            return _raw_to_hsv(flattened).reshape(image.shape)
        elif colorspace == COLORSPACE_GRAY:
            return _raw_to_hsv(flattened).reshape(image.shape[:2])
        else:
            raise ValueError("unknown colorspace code '{}'".format(colorspace))

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

        # remove classe where there is no mask
        class_map = np.take(self._pyxit.classes_, np.argmax(confidence_map, axis=2))
        class_map[np.logical_not(mask)] = self._background
        return class_map


class AnnotationAreaChecker(object):
    def __init__(self, min_area, max_area, level):
        self._min_area = min_area
        self._max_area = max_area
        self._level = level

    def check(self, annot):
        if self._max_area < 0:
            return self._min_area < annot.area * (2**self._level)
        else:
            return self._min_area < (annot.area * (2**self._level)) < self._max_area


def change_referential(p, height):
    return affine_transform(p, [1, 0, 0, -1, 0, height])


def get_iip_window_from_annotation(slide, annotation, zoom_level):
    """generate a iip-compatible roi based on an annotation at the given zoom level"""
    roi_polygon = change_referential(wkt.loads(annotation.location), slide.image_instance.height)
    if zoom_level == 0:
        return slide.window_from_polygon(roi_polygon, mask=True)
    # recompute the roi so that it matches the iip tile topology
    zoom_ratio = 1 / (2 ** zoom_level)
    scaled_roi = affine_transform(roi_polygon, [zoom_ratio, 0, 0, zoom_ratio, 0, 0])
    min_x, min_y, max_x, max_y = (int(v) for v in scaled_roi.bounds)
    diff_min_x, diff_min_y = min_x % 256, min_y % 256
    diff_max_x, diff_max_y = max_x % 256, max_y % 256
    min_x -= diff_min_x
    min_y -= diff_min_y
    max_x = min(slide.width, max_x + 256 - diff_max_x)
    max_y = min(slide.height, max_y + 256 - diff_max_y)
    return slide.window((min_x, min_y), max_x - min_x, max_y - min_y, scaled_roi)


def extract_images_or_rois(parameters):
    # work at image level or ROIs by term
    images = ImageInstanceCollection()
    if parameters.cytomine_id_images is not None:
        id_images = parse_domain_list(parameters.cytomine_id_images)
        images.extend([ImageInstance().fetch(_id) for _id in id_images])
    else:
        images = images.fetch_with_filter("project", parameters.cytomine_id_project)

    slides = [CytomineSlide(img, parameters.cytomine_zoom_level) for img in images]
    if parameters.cytomine_id_roi_term is None:
        return slides

    # fetch ROI annotations, all users
    collection = AnnotationCollection(
        terms=[parameters.cytomine_id_roi_term],
        reviewed=parameters.cytomine_reviewed_roi,
        project=parameters.cytomine_id_project,
        showWKT=True,
        includeAlgo=True
    ).fetch()

    slides_map = {slide.image_instance.id: slide for slide in slides}
    regions = list()
    for annotation in collection:
        if annotation.image not in slides_map:
            continue
        slide = slides_map[annotation.image]
        regions.append(get_iip_window_from_annotation(slide, annotation, parameters.cytomine_zoom_level))

    return regions


class CytomineOldIIPTile(CytomineDownloadableTile):
    def download_tile_image(self):
        slide = self.base_image
        col_tile = self.abs_offset_x // 256
        row_tile = self.abs_offset_y // 256
        _slice = slide.image_instance
        response = Cytomine.get_instance().get('imaging_server.json', None)
        imageServerUrl = response['collection'][0]['url']
        return Cytomine.get_instance().download_file(imageServerUrl + "/image/tile", self.cache_filepath, False, payload={
            "zoomify": _slice.fullPath,
            "mimeType": _slice.mime,
            "x": col_tile,
            "y": row_tile,
            "z": slide.api_zoom_level
        })


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        # use only images from the current project
        cj.job.update(progress=1, statusComment="Preparing execution")

        # extract images to process
        if cj.parameters.cytomine_zoom_level > 0 and (cj.parameters.cytomine_tile_size != 256 or cj.parameters.cytomine_tile_overlap != 0):
            raise ValueError("when using zoom_level > 0, tile size should be 256 "
                             "(given {}) and overlap should be 0 (given {})".format(
                cj.parameters.cytomine_tile_size, cj.parameters.cytomine_tile_overlap))

        cj.job.update(progress=1, statusComment="Preparing execution (creating folders,...).")
        # working path
        root_path = str(Path.home())
        working_path = os.path.join(root_path, "images")
        os.makedirs(working_path, exist_ok=True)

        # load training information
        cj.job.update(progress=5, statusComment="Extract properties from training job.")
        train_job = Job().fetch(cj.parameters.cytomine_id_job)
        properties = PropertyCollection(train_job).fetch().as_dict()
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

        cj.job.update(progress=45, statusComment="Build workflow.")
        builder = SSLWorkflowBuilder()
        builder.set_tile_size(cj.parameters.cytomine_tile_size, cj.parameters.cytomine_tile_size)
        builder.set_overlap(cj.parameters.cytomine_tile_overlap)
        builder.set_tile_builder(CytomineTileBuilder(working_path, tile_class=CytomineOldIIPTile, n_jobs=cj.parameters.n_jobs))
        builder.set_logger(StandardOutputLogger(level=Logger.INFO))
        builder.set_n_jobs(1)
        builder.set_background_class(0)
        # value 0 will prevent merging but still requires to run the merging check
        # procedure (inefficient)
        builder.set_distance_tolerance(2 if cj.parameters.union_enabled else 0)
        builder.set_segmenter(ExtraTreesSegmenter(
            pyxit=pyxit,
            classes=classes,
            prediction_step=cj.parameters.pyxit_prediction_step,
            background=0,
            min_std=cj.parameters.tile_filter_min_stddev,
            max_mean=cj.parameters.tile_filter_max_mean
        ))
        workflow = builder.get()

        area_checker = AnnotationAreaChecker(
            min_area=cj.parameters.min_annotation_area,
            max_area=cj.parameters.max_annotation_area,
            level=cj.parameters.cytomine_zoom_level
        )

        def get_term(label):
            if binary:
                if "cytomine_id_predict_term" not in cj.parameters or not cj.parameters.cytomine_id_predict_term:
                    return []
                else:
                    return [int(cj.parameters.cytomine_id_predict_term)]
            # multi-class
            return [label]

        zoom_mult = (2 ** cj.parameters.cytomine_zoom_level)
        zones = extract_images_or_rois(cj.parameters)
        for zone in cj.monitor(zones, start=50, end=90, period=0.05, prefix="Segmenting images/ROIs"):
            results = workflow.process(zone)

            if cj.parameters.cytomine_id_roi_term is not None:
                ROI = change_referential(translate(zone.polygon_mask, zone.offset[0], zone.offset[1]), zone.base_image.height)
                if cj.parameters.cytomine_zoom_level > 0:
                    ROI = affine_transform(ROI, [zoom_mult, 0, 0, zoom_mult, 0, 0])

            annotations = AnnotationCollection()
            for obj in results:
                if not area_checker.check(obj.polygon):
                    continue
                polygon = obj.polygon
                if isinstance(zone, ImageWindow):
                    polygon = affine_transform(polygon, [1, 0, 0, 1, zone.abs_offset_x, zone.abs_offset_y])


                polygon = change_referential(polygon, zone.base_image.height)
                if cj.parameters.cytomine_zoom_level > 0:
                    polygon = affine_transform(polygon, [zoom_mult, 0, 0, zoom_mult, 0, 0])


                if cj.parameters.cytomine_id_roi_term is not None:
                    polygon = polygon.intersection(ROI)

                if not polygon.is_empty:
                    annotations.append(Annotation(
                        location=polygon.wkt,
                        id_terms=get_term(obj.label),
                        id_project=cj.project.id,
                        id_image=zone.base_image.image_instance.id
                    ))
            """        
            Some annotations found thanks to the algorithm are Geometry Collections,
            containing more than one element -> keep only polygons from those 
            GeometryCollections
            """
            annotations_ok = AnnotationCollection()
            for annotation in annotations:
                if isinstance(shapely.wkt.loads(annotation.location), shapely.geometry.collection.GeometryCollection):
                    for geom in shapely.wkt.loads(annotation.location):
                        if isinstance(geom, shapely.geometry.Polygon):
                            annotations_ok.append(Annotation(
                                location=geom.wkt,
                                id_terms=annotation.term,
                                id_project=annotation.project,
                                id_image=annotation.image
                            ))
                elif isinstance(shapely.wkt.loads(annotation.location), shapely.geometry.Polygon):
                    annotations_ok.append(annotation)
                else:
                    continue
                    
            annotations_ok.save()

        cj.job.update(status=Job.TERMINATED, status_comment="Finish", progress=100)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

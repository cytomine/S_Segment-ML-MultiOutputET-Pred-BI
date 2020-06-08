import os
import joblib
import numpy as np
from pathlib import Path
from skimage.util.shape import view_as_windows
from skimage.transform import resize

from cytomine import CytomineJob
from cytomine.models import ImageInstanceCollection, ImageInstance, AttachedFileCollection, Job, PropertyCollection, \
    AnnotationCollection
from cytomine.utilities.software import parse_domain_list, str2bool
from sldc import SemanticSegmenter, SSLWorkflowBuilder, StandardOutputLogger, Logger
from sldc_cytomine import CytomineTileBuilder, CytomineSlide


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
            channels = [image[:, :, i] for i in image.shape[2]]
        return np.any([
            np.std(c) > self._min_std or np.mean(c) < self._max_mean
            for c in channels
        ])

    def segment(self, image):
        if not self._process_tile(image):
            return np.full(image.shape[:2], self._background)
        # prepare windows
        target_height = self._pyxit.target_height
        target_width = self._pyxit.target_width
        subwindows_dims = [target_height, target_width]
        if image.ndim > 2 and image.shape[2] > 1:
            subwindows_dims += [image.shape[2]]
        subwindows = view_as_windows(image, subwindows_dims, step=self._prediction_step)
        grid_dims = subwindows.shape[:2]
        subwindows = subwindows.reshape([-1, np.product(subwindows_dims)])

        # predict
        X = np.array([""] * subwindows.shape[0])
        y = self._pyxit.predict(X, _X=subwindows)

        # propagate in whole image
        return resize(y.reshape(grid_dims), image.shape[:2], order=0)


class AnnotationAreaChecker(object):
    def __init__(self, min_area, max_area):
        self._min_area = min_area
        self._max_area = max_area

    def check(self, annot):
        return self._min_area < annot.area < self._max_area


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        # use only images from the current project
        cj.job.update(progress=1, statuscomment="Preparing execution (creating folders,...).")

        if cj.parameters.cytomine_zoom_level != 0:
            raise ValueError("Zoom level > 0 not supported.")

        # working path
        root_path = Path.home()
        working_path = os.path.join(root_path, "images")

        # load training information
        cj.job.update(progress=5, statusComment="Extract properties from training job.")
        train_job = Job().fetch(cj.parameters.cytomine_id_job)
        properties = PropertyCollection(train_job).fetch().as_dict()
        binary = str2bool(properties["binary"].value)
        foreground_classes = parse_domain_list(properties["foreground_classes"].value)

        cj.job.update(progress=10, statusComment="Download the model file.")
        attached_files = AttachedFileCollection(train_job).fetch()
        model_file = attached_files.find_by_attribute("filename", "model.joblib")
        model_filepath = os.path.join(root_path, "model.joblib")
        model_file.download(model_filepath, override=True)
        pyxit = joblib.load(model_filepath)

        # set n_jobs
        pyxit.base_estimator.n_jobs = cj.parameters.n_jobs
        pyxit.n_jobs = cj.parameters.n_jobs

        # extract images to process
        images = ImageInstanceCollection()
        if cj.parameters.cytomine_id_images is not None:
            id_images = parse_domain_list(cj.parameters.id_images)
            images.extend([ImageInstance().fetch(_id) for _id in id_images])
        else:
            images = images.fetch_with_filter("project", cj.project.id)

        builder = SSLWorkflowBuilder()
        builder.set_tile_size(cj.parameters.cytomine_tile_size, cj.parameters.cytomine_tile_size)
        builder.set_overlap(cj.parameters.cytomine_tile_overlap)
        builder.set_tile_builder(CytomineTileBuilder(working_path))
        builder.set_logger(StandardOutputLogger(level=Logger.INFO))
        builder.set_n_jobs(1)
        builder.set_background_class(0)
        # value 0 will prevent merging but still requires to run the merging check
        # procedure (inefficient)
        builder.set_distance_tolerance(1 if cj.parameters.union_enabled else 0)
        builder.set_segmenter(ExtraTreesSegmenter(
            pyxit=pyxit,
            classes=[0] + ([cj.parameters.cytomine_predict_term] if binary else foreground_classes),
            prediction_step=cj.parameters.cytomine_prediction_step,
            background=0,
            min_std=cj.parameters.tile_filter_min_std,
            max_mean=cj.parameters.tile_filter_max_mean
        ))
        workflow = builder.get()

        annotations = AnnotationCollection()
        area_checker = AnnotationAreaChecker(
            min_area=cj.parameters.min_annotation_area,
            max_area=cj.parameters.max_annotation_area
        )
        for image in images:
            wsi = CytomineSlide(image.id)
            results = workflow.process(wsi)
            annotations.extend([r.polygon for r in results if area_checker.check(r.polygon)])

        annotations.save()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

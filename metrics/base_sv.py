from dataset.basergbdataset import BaseRGBDataet, TrackerResult


class Metric_sv:
    def __init__(self) -> None:
        pass

    def __call__(self, dataset: BaseRGBDataet, res: TrackerResult):
        pass

    def __call__(self, dataset: BaseRGBDataet):
        self(dataset, dataset.trackers)

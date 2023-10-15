from .base_sv import Metric_sv
from dataset.basergbdataset import BaseRGBDataet, TrackerResult
from vot.region.shapes import Rectangle
from vot.cal_enus import compute_ENUS1
from tool.utils import *


class Sequence():
    def __init__(self, gt):
        self.groundtruth = [Rectangle(per_gt[0], per_gt[1], per_gt[2], per_gt[3]) for per_gt in gt]
        self.size = (320, 320)


class MPR(Metric_sv):
    """
    NOTE
    ---------
    Maximum Precision Rate (MPR). PR is the percentage of frames whose output location
    is within the given threshold distance of ground truth. That is to say, it computes
    the average Euclidean distance between the center locations of the tracked target
    and the manually labeled ground-truth positions of all the frames. Although our
    alignment between two modalities is highly accurate, there still exist small alignment
    errors. Therefore, we use maximum precision rate (MPR) instead of PR in this paper.
    Specifically, for each frame, we compute the above Euclidean distance on both RGB and
    thermal modalities, and adopt the smaller distance to compute the precision.
    We set the threshold to be 20 pixels to obtain the representative MPR.
    """

    def __init__(self, thr=np.linspace(0, 50, 51)) -> None:
        super().__init__()
        self.thr = thr

    def __call__(self, dataset: BaseRGBDataet, result: TrackerResult, seqs: list):
        pr = []
        for seq_name in seqs:
            gt_v = dataset[seq_name]
            serial = result[seq_name]
            res = np.array(serial_process(CLE, serial, gt_v))

            pr_cell = []
            for i in self.thr:
                pr_cell.append(np.sum(res <= i) / len(res))
            pr.append(pr_cell)
        pr = np.array(pr)
        pr_val = pr.mean(axis=0)[20]
        return pr_val, pr


class ENUS(Metric_sv):
    """
    NOTE
    ---------
    Maximum Precision Rate (MPR). PR is the percentage of frames whose output location
    is within the given threshold distance of ground truth. That is to say, it computes
    the average Euclidean distance between the center locations of the tracked target
    and the manually labeled ground-truth positions of all the frames. Although our
    alignment between two modalities is highly accurate, there still exist small alignment
    errors. Therefore, we use maximum precision rate (MPR) instead of PR in this paper.
    Specifically, for each frame, we compute the above Euclidean distance on both RGB and
    thermal modalities, and adopt the smaller distance to compute the precision.
    We set the threshold to be 20 pixels to obtain the representative MPR.
    """

    def __init__(self, thr=np.linspace(0, 1, 51)) -> None:
        super().__init__()

        self.thr = thr

    def __call__(self, dataset: BaseRGBDataet, result: TrackerResult, seqs: list):
        enus = []
        for seq_name in seqs:
            gt_v = dataset[seq_name]
            serial = result[seq_name]
            gt = Sequence(gt_v)
            serial = [Rectangle(per_serial[0], per_serial[1], per_serial[2], per_serial[3]) for per_serial in serial]
            # res = np.array(serial_process(IoU, serial, gt_v))

            enus.append(compute_ENUS1(serial, gt, thr=self.thr)[0])

        enus = np.array(enus)
        enus_val = enus.mean()
        return enus_val, enus


class MSR(Metric_sv):
    """
    NOTE
    ---------
    Maximum Success Rate (MSR). SR is the ratio of the number of successful frames whose
    overlap is larger than a threshold. Similar to MPR, we also define maximum success
    rate (MSR) to measure the tracker results. By varying the threshold, the MSR plot can
    be obtained, and we employ the area under curve of MSR plot to define the representative MSR.
    """

    def __init__(self, thr=np.linspace(0, 1, 21)) -> None:
        super().__init__()
        self.thr = thr

    def __call__(self, dataset: BaseRGBDataet, result: TrackerResult, seqs: list):

        sr = []
        for seq_name in seqs:
            gt_v = dataset[seq_name]
            serial = result[seq_name]
            res = np.array(serial_process(IoU, serial, gt_v))

            sr_cell = []
            for i in self.thr:
                sr_cell.append(np.sum(res > i) / len(res))
            sr.append(sr_cell)

        sr = np.array(sr)
        sr_val = sr.mean()
        return sr_val, sr


class PR(Metric_sv):
    """
    Precision Rate.
    """

    def __init__(self, thr=np.linspace(0, 50, 51)) -> None:
        super().__init__()
        self.thr = thr

    def __call__(self, dataset: BaseRGBDataet, result: TrackerResult, seqs: list):
        pr = []
        for seq_name in seqs:
            try:
                gt = dataset[seq_name]
                serial = result[seq_name]
                serial[0] = gt[0]  # ignore the first frame
            except:
                gt = dataset[seq_name]['groundTruth']
                serial = result[seq_name]
                serial[0] = gt[0]  # ignore the first frame
            res = np.array(serial_process(CLE, serial, gt))

            pr_cell = []
            for i in self.thr:
                pr_cell.append(np.sum(res <= i) / len(res))
            pr.append(pr_cell)

        pr = np.array(pr)
        pr_val = pr.mean(axis=0)[20]
        return pr_val, pr


class SR(Metric_sv):
    """
    Success Rate.
    """

    def __init__(self, thr=np.linspace(0, 1, 21)) -> None:
        super().__init__()
        self.thr = thr

    def __call__(self, dataset: BaseRGBDataet, result: TrackerResult, seqs: list):

        sr = []
        for seq_name in seqs:
            try:
                gt = dataset[seq_name]
                serial = result[seq_name]
                serial[0] = gt[0]  # ignore the first frame
            except:
                gt = dataset[seq_name]['groundTruth']
                serial = result[seq_name]
                serial[0] = gt[0]  # ignore the first frame
            res = np.array(serial_process(IoU, serial, gt))

            sr_cell = []
            for i in self.thr:
                sr_cell.append(np.sum(res > i) / len(res))
            sr.append(sr_cell)

        sr = np.array(sr)
        sr_val = sr.mean()
        return sr_val, sr


class NPR(Metric_sv):
    """
    Normalized Precision Rate.
    """

    def __init__(self, thr=np.linspace(0, 0.5, 51)) -> None:
        super().__init__()
        self.thr = thr

    def __call__(self, dataset: BaseRGBDataet, result: TrackerResult, seqs: list):
        pr = []
        for seq_name in seqs:
            try:
                gt = dataset[seq_name]
                serial = result[seq_name]
                serial[0] = gt[0]  # ignore the first frame
            except:
                gt = dataset[seq_name]['groundTruth']
                serial = result[seq_name]
                serial[0] = gt[0]  # ignore the first frame
            # cut off tracking result
            serial = serial[:len(gt)]
            # handle the invailded tracking result
            for i in range(1, len(gt)):
                if serial[i][2] <= 0 or serial[i][3] <= 0:
                    serial[i] = serial[i - 1].copy()
            res = np.array(serial_process(normalize_CLE, serial, gt))

            for i in range(len(gt)):
                if sum(gt[i] <= 0):
                    res[i] = -1

            pr_cell = []
            for i in self.thr:
                pr_cell.append(np.sum(res <= i) / len(res))
            pr.append(pr_cell)
        pr = np.array(pr)
        pr_val = pr.mean(axis=0)[20]
        return pr_val, pr

# from .basedataset import BaseRGBTDataet, _basepath
from .basergbdataset import BaseRGBDataet
import os
from tool.utils import *
from metrics import MPR, MSR, ENUS

_basepath = os.path.dirname(os.path.dirname(__file__))


class SV248S(BaseRGBDataet):
    """
    Publication: `Learning collaborative sparse representation for grayscale-thermal tracking` 2016\\
    IEEE Transactions on Image Processing \\
    [Download Dataset.](https://github.com/mmic-lcl/Datasets-and-benchmark-code)

    NOTE: this is not support attribute test. [Just here, not SV248s]
    """

    def __init__(self, gt_path=f"{_basepath}/gt_file/SV248S/sv248s_gt/",
                 seq_name="01") -> None:
        seq_name_path = f"{_basepath}/gt_file/SV248s/{seq_name}.txt"
        seqs = load_text(seq_name_path, dtype=str)
        super().__init__(gt_path=gt_path, seqs=seqs, bbox_type='ltwh', v_name=None)
        # super().__init__(gt_path=gt_path, seqs=seqs, bbox_type='ltwh', v_name='init.txt', i_name='init.txt')

        self.name = 'SV248S'
        self.MPR_fun = MPR()
        self.MSR_fun = MSR()
        self.ENUS_fun = ENUS()

        # Challenge attributes
        self._attr_list = ('STO', 'LTO', 'DS', 'IV', 'BCH', 'SM', 'ND', 'CO', 'BCL', 'IPR')
        self.STO = self.choose_serial_by_att('STO')
        self.LTO = self.choose_serial_by_att('LTO')
        self.DS = self.choose_serial_by_att('DS')
        self.IV = self.choose_serial_by_att('IV')
        self.BCH = self.choose_serial_by_att('BCH')
        self.SM = self.choose_serial_by_att('SM')
        self.ND = self.choose_serial_by_att('ND')
        self.CO = self.choose_serial_by_att('CO')
        self.BCL = self.choose_serial_by_att('BCL')
        self.IPR = self.choose_serial_by_att('IPR')
        self._type_list = (
            'Airplane', 'Car', 'Car_Large', 'Hard', 'Large_Vehicle', 'Normal', 'Plane', 'Ship', 'Simple', 'Vehicle')
        self.Airplane = self.choose_serial_by_type('Airplane')
        self.Car = self.choose_serial_by_type('Car')
        self.Car_Large = self.choose_serial_by_type('Car_Large')
        self.Hard = self.choose_serial_by_type('Hard')
        self.Large_Vehicle = self.choose_serial_by_type('Large_Vehicle')
        self.Normal = self.choose_serial_by_type('Normal')
        self.Plane = self.choose_serial_by_type('Plane')
        self.Ship = self.choose_serial_by_type('Ship')
        self.Simple = self.choose_serial_by_type('Simple')
        self.Vehicle = self.choose_serial_by_type('Vehicle')

    def get_attr_list(self):
        return self._attr_list

    def get_type_list(self):
        return self._type_list

    def choose_serial_by_att(self, attr):
        if attr == self.ALL:
            return self.seqs_name
        else:
            seqs = []
            for seq in self.seqs_name:
                i = self.get_attr_list().index(attr)
                path = os.path.join(self.gt_path, '..', 'AttriSeqsTxt', seq + '.txt')
                p = load_text(path)[i]
                if p == 1.:
                    seqs.append(seq)
            return seqs

    def choose_serial_by_type(self, type):
        seqs = []
        for seq in self.seqs_name:
            i = self.get_type_list().index(type)
            path = os.path.join(self.gt_path, '..', 'TypeSeqsTxt', seq + '.txt')
            p = load_text(path)[i]
            if p == 1.:
                seqs.append(seq)
        return seqs

    def MPR(self, tracker_name=None, seqs=None):
        """
        Parameters
        ----------
        [in] tracker_name - str
            Default is None, evaluate all registered trackers.
        [in] seqs - list
            Sequence to be evaluated, default is all.

        Returns
        -------
        [out0] When evaluating a single tracker, return MPR and the precision Rate at different thresholds.
        [out1] Other cases return a dictionary with all tracker results.
        """
        if seqs == None:
            seqs = self.seqs_name

        if tracker_name != None:
            return self.MPR_fun(self, self.trackers[tracker_name], seqs)
        else:
            res = {}
            for k, v in self.trackers.items():
                res[k] = self.MPR_fun(self, v, seqs)
            return res

    def MSR(self, tracker_name=None, seqs=None):
        """
        NOTE
        ---------
        > Maximum Success Rate (MSR). SR is the ratio of the number of successful frames whose
        overlap is larger than a threshold. Similar to MPR, we also define maximum success
        rate (MSR) to measure the tracker results. By varying the threshold, the MSR plot can
        be obtained, and we employ the area under curve of MSR plot to define the representative MSR.

        Parameters
        ----------
        [in] tracker_name - str
            Default is None, evaluate all registered trackers.
        [in] seqs - list
            Sequence to be evaluated, default is all.

        Returns
        -------
        [out0] When evaluating a single tracker, return MSR and the Success Rate at different thresholds.
        [out1] Other cases return a dictionary with all tracker results.
        """
        if seqs == None:
            seqs = self.seqs_name

        if tracker_name != None:
            return self.MSR_fun(self, self.trackers[tracker_name], seqs)
        else:
            res = {}
            for k, v in self.trackers.items():
                res[k] = self.MSR_fun(self, v, seqs)
            return res

    def ENUS(self, tracker_name=None, seqs=None):

        if seqs == None:
            seqs = self.seqs_name

        if tracker_name != None:
            return self.ENUS_fun(self, self.trackers[tracker_name], seqs)
        else:
            res = {}
            for k, v in self.trackers.items():
                res[k] = self.ENUS_fun(self, v, seqs)
            return res

    def draw_plot(self, metric_fun, filename=None, title=None, seqs=None):
        assert metric_fun == self.MSR or metric_fun == self.MPR or metric_fun == self.ENUS
        if filename == None:
            filename = self.name
            if metric_fun == self.MPR:
                filename += "_MPR"
                axis = self.MPR_fun.thr
                loc = "lower right"
                x_label = "Location error threshold"
                y_label = "Precision"
            elif metric_fun == self.MSR:
                filename += "_MSR"
                axis = self.MSR_fun.thr
                loc = "lower left"
                x_label = "overlap threshold"
                y_label = "Success Rate"
            elif metric_fun == self.ENUS:
                filename += "_ENUS"
                axis = self.ENUS_fun.thr
                loc = "lower left"
                x_label = "ENUS threshold"
                y_label = "ENUS"
            filename += "_plot.png"
        else:
            if metric_fun == self.MPR:
                filename += "_MPR"
                axis = self.MPR_fun.thr
                loc = "lower right"
                x_label = "Location error threshold"
                y_label = "Precision"
            elif metric_fun == self.MSR:
                filename += "_MSR"
                axis = self.MSR_fun.thr
                loc = "lower left"
                x_label = "overlap threshold"
                y_label = "Success Rate"
            elif metric_fun == self.ENUS:
                filename += "_ENUS"
                axis = self.ENUS_fun.thr
                loc = "lower left"
                x_label = "ENUS threshold"
                y_label = "ENUS"
            filename += "_plot.png"

        if title == None:
            if metric_fun == self.MPR:
                title = "Precision Plot"
            elif metric_fun == self.MSR:
                title = "Success Plot"
            elif metric_fun == self.ENUS:
                title = "ENUS Plot"

        return super().draw_plot(axis=axis,
                                 metric_fun=metric_fun,
                                 filename=filename,
                                 title=title,
                                 seqs=seqs, y_max=1.0, y_min=0.0, loc=loc,
                                 x_label=x_label, y_label=y_label)

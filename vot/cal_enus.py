from typing import List, Tuple, Any
import numpy as np
from vot.region import Region, is_special, Special, calculate_overlaps, calculate_location_errs, calculate_ps, \
    calculate_rs
from vot.region.shapes import Rectangle


# from vot.dataset import Sequence





def compute_ENUS1(trajectory: List[Region], sequence, burnin: int = 1,thr = None,
                   ignore_unknown: bool = True, bounded: bool = True) -> float:
    if isinstance(trajectory[1], Rectangle):
        gt_rect = []
        for gt_poly in sequence.groundtruth:
            if len(gt_rect) == 0:
                gt_rect.append(Special(1))
                continue
            # points = np.array(gt_poly)
            # print(points)
            # x1, y1, x2, y2 = np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])
            x1, y1, x2, y2 = gt_poly.x, gt_poly.y, gt_poly.x + gt_poly.width, gt_poly.y + gt_poly.height
            gt_rect.append(Rectangle(x1, y1, x2 - x1, y2 - y1))
        p_standard = calculate_ps(gt_rect, sequence.groundtruth, (sequence.size) if bounded else None)
        p_standard[0] = 1
    else:
        p_standard = 1
    # print(p_standard)

    P = np.array(calculate_ps(trajectory, sequence.groundtruth, (sequence.size) if bounded else None))
    R = np.array(calculate_rs(trajectory, sequence.groundtruth, (sequence.size) if bounded else None))
    overlaps = R * (1 - np.power(np.abs(P / p_standard - 1), 1))
    location_errs = np.array(
        calculate_location_errs(trajectory, sequence.groundtruth, (sequence.size) if bounded else None))

    overlaps = overlaps * location_errs

    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(trajectory):
        if is_special(region, Special.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Special.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region, Special.FAILURE):
            mask[i] = False

    overlaps = overlaps[mask]

    # step = np.arange(0, 1.01, 0.02)
    # step = np.arange(0, 1.01, 0.01)
    step = thr
    output = []
    number_frame = np.sum(mask)
    for s in step:
        output.append(np.sum(overlaps > s) / (number_frame + 0.000001))

    if any(mask):
        return output, np.sum(mask)
    else:
        return [], 0


    # if any(mask):
    #     return np.mean(overlaps[mask]), np.sum(mask)
    # else:
    #     return 0, 0


# if __name__ == '__main__':
#     trajectory = [Rectangle(5, 5, 2, 2), Rectangle(9, 9, 6, 6)]
#     enus, _ = compute_ENUS1(trajectory, sequence_new())
#
#     print(enus)

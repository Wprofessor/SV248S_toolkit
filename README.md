# SV248S_toolkit

## Acknowledgment
- This repo is based on [RGBT_toolkit](https://github.com/opacity-black/RGBT_toolkit) which is an excellent work.


## Evaluate
- We provide the ground truth file, so you can directly call it.
- The relevant code is in eval_sv248s.py

```python
from dataset import SV248S

seq_name = 'all'  # 01 02 03 04 05 06 all

sv248s = SV248S(seq_name=seq_name)

# Register your tracker
sv248s(
    tracker_name="test1",  # 跟踪器的名字，可自定义
    result_path="./test1",  # 结果存放的地址
    bbox_type="ltwh",
    prefix="")

sv248s(
    tracker_name="test2",  # 跟踪器的名字，可自定义
    result_path="./test2",  # 结果存放的地址
    bbox_type="ltwh",
    prefix="")

# STO,LTO,DS,IV,BCH,SM,ND,CO,BCL,IPR     属性
# Airplane, Car, Car_Large, Hard, Large_Vehicle, Normal, Plane, Ship, Simple, Vehicle 类型

# Draw a curve plot （可以更换不同的属性和类型，若不添加属性和类型，则绘制普通的分数图)
sv248s.draw_plot(metric_fun=sv248s.MPR, filename='sv248s_' + seq_name, seqs=sv248s.Hard)
sv248s.draw_plot(metric_fun=sv248s.MSR, filename='sv248s_' + seq_name, seqs=sv248s.Hard)
sv248s.draw_plot(metric_fun=sv248s.ENUS, filename='sv248s_' + seq_name, seqs=sv248s.Hard)


#  Draw a radar chart of all challenge attributes
sv248s.draw_attributeRadar(metric_fun=sv248s.ENUS, filename='sv248s_' + seq_name)
sv248s.draw_attributeRadar(metric_fun=sv248s.MSR, filename='sv248s_' + seq_name)
sv248s.draw_attributeRadar(metric_fun=sv248s.MPR, filename='sv248s_' + seq_name)
```

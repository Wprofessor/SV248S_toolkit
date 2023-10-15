from dataset import SV248S

seq_name = 'all'  # 01 02 03 04 05 06 all

sv248s = SV248S(seq_name=seq_name)

# Register your tracker
sv248s(
    tracker_name="test1",  # Result in paper: 90.2, 73.2
    result_path="./test1",
    bbox_type="ltwh",
    prefix="")

# sv248s(
#     tracker_name="test2",  # Result in paper: 90.2, 73.2
#     result_path="./test2",   # 结果存放的地址
#     bbox_type="ltwh",
#     prefix="")





# STO,LTO,DS,IV,BCH,SM,ND,CO,BCL,IPR
# Airplane, Car, Car_Large, Hard, Large_Vehicle, Normal, Plane, Ship, Simple, Vehicle

# Draw a curve plot
sv248s.draw_plot(metric_fun=sv248s.MPR, filename='sv248s_' + seq_name, seqs=sv248s.Hard)
sv248s.draw_plot(metric_fun=sv248s.MSR, filename='sv248s_' + seq_name, seqs=sv248s.Hard)
sv248s.draw_plot(metric_fun=sv248s.ENUS, filename='sv248s_' + seq_name, seqs=sv248s.Hard)

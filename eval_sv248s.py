from dataset import SV248S

seq_name = '01'  # 01 02 03 04 05 06

sv248s = SV248S(seq_name=seq_name)

# Register your tracker
sv248s(
    tracker_name="test1",  # Result in paper: 90.2, 73.2
    result_path="./test1",
    bbox_type="ltwh",
    prefix="")
sv248s(
    tracker_name="test2",  # Result in paper: 90.2, 73.2
    result_path="./test2",
    bbox_type="ltwh",
    prefix="")

# Evaluate multiple trackers
pr_dict = sv248s.MPR()
print("test1 MPR: ", pr_dict["test1"][0])

# Evaluate single tracker
test_sr, _ = sv248s.MSR("test1")
print("test1 MSR:\t", test_sr)

# Draw a curve plot
sv248s.draw_plot(metric_fun=sv248s.MPR, filename='sv248s_' + seq_name)
sv248s.draw_plot(metric_fun=sv248s.MSR, filename='sv248s_' + seq_name)

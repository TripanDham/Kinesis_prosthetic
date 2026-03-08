import joblib

path = "kit_test_motion_dict.pkl"


data = joblib.load(path)

# ['pose_quat_global', 'pose_quat', 'trans_orig', 'root_trans_offset', 'beta', 'gender', 'pose_aa', 'fps']
print(data['0-KIT_167_turn_right09_poses']['pose_quat'].shape)
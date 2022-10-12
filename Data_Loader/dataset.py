import numpy as np


def processing_data(features):
    # remove score of keypoint
    features = features[:, :, :, :2]
    # ********************************* COMPUTE ANGLE *********************************************
    # angle knee right
    knee_hip = features[:, :, 14:15, :] - features[:, :, 12:13, :]
    knee_ankle = features[:, :, 14:15, :] - features[:, :, 16:17, :]
    a = np.sum(knee_hip * knee_ankle, axis=3)
    b = np.sqrt(np.sum(knee_hip ** 2, axis=3)) * np.sqrt(np.sum(knee_ankle ** 2, axis=3))
    b = np.where(b == 0, 1, b)
    angle_knee_right = a / b

    # angle knee left
    knee_hip = features[:, :, 13:14, :] - features[:, :, 11:12, :]
    knee_ankle = features[:, :, 13:14, :] - features[:, :, 15:16, :]
    a = np.sum(knee_hip * knee_ankle, axis=3)
    b = np.sqrt(np.sum(knee_hip ** 2, axis=3)) * np.sqrt(np.sum(knee_ankle ** 2, axis=3))
    b = np.where(b == 0, 1, b)
    angle_knee_left = a / b

    # angle hip right
    hip_shoulder = features[:, :, 12:13, :] - features[:, :, 6:7, :]
    hip_knee = features[:, :, 12:13, :] - features[:, :, 14:15, :]
    a = np.sum(hip_shoulder * hip_knee, axis=3)
    b = np.sqrt(np.sum(hip_shoulder ** 2, axis=3)) * np.sqrt(np.sum(hip_knee ** 2, axis=3))
    b = np.where(b == 0, 1, b)
    angle_hip_right = a / b

    # angle hip left
    hip_shoulder = features[:, :, 11:12, :] - features[:, :, 5:6, :]
    hip_knee = features[:, :, 11:12, :] - features[:, :, 13:14, :]
    a = np.sum(hip_shoulder * hip_knee, axis=3)
    b = np.sqrt(np.sum(hip_shoulder ** 2, axis=3)) * np.sqrt(np.sum(hip_knee ** 2, axis=3))
    b = np.where(b == 0, 1, b)
    angle_hip_left = a / b

    # remove 4 point 1,2,3,4 with eye, ear
    features = np.concatenate([features[:, :, 0:1, :], features[:, :, 5:, :]], axis=2)

    # ***************************************** NORMALIZE ************************************

    def scale_pose(xy):
        """
        Normalize pose points by scale with max/min value of each pose.
        xy : (frames, parts, xy) or (parts, xy)
        """
        if xy.ndim == 2:
            xy = np.expand_dims(xy, 0)
        xy_min = np.nanmin(xy, axis=2).reshape(xy.shape[0], xy.shape[1], 1, 2)
        xy_max = np.nanmax(xy, axis=2).reshape(xy.shape[0], xy.shape[1], 1, 2)
        xy = (xy - xy_min) / (xy_max - xy_min) * 2 - 1
        return xy

    features = scale_pose(features)
    # flatten
    features = features[:, :, :, :].reshape(len(features), features.shape[1], features.shape[2] * features.shape[3])

    # ****************************************** Concatenate ********************************************************
    features = np.concatenate([features, angle_hip_left, angle_hip_right, angle_knee_left, angle_knee_right], axis=2)
    return features

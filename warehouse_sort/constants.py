"""Constants shared by the environment and the policy loaders.

Kept free of simulator imports so `warehouse_sort.il_policy` can be imported (and unit-tested)
on a machine without ManiSkill/SAPIEN.
"""

# Robot home configuration used by WarehouseSortEnv._initialize_episode (7 arm joints + 2 fingers).
START_QPOS = [0.0, 0.3927, 0.0, -1.9635, 0.0, 2.356, 0.7854, 0.04, 0.04]

# Layout of the proprioception vector obs["state"] on the rgb track (26 dims). Produced by
# FlattenRGBDObservationWrapper / build_state_obs_extractor in this order:
#   agent.qpos (9) | agent.qvel (9) | extra.tcp_pose (7: xyz + quat wxyz) | extra.is_grasped (1)
PROPRIO_DIM = 26
QPOS_SLICE = slice(0, 9)
QVEL_SLICE = slice(9, 18)
TCP_POSE_SLICE = slice(18, 25)
IS_GRASPED_INDEX = 25

# Action layout (pd_ee_delta_pos): [dx, dy, dz, gripper], all in [-1, 1].
ACTION_DIM = 4
GRIPPER_INDEX = 3

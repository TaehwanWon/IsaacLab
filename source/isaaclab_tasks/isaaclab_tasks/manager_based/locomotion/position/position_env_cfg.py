# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
import random


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.ladder.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING

    # objects
    left_pole: RigidObjectCfg = MISSING
    right_pole: RigidObjectCfg = MISSING
    # rung_0 ~ rung_4 필드 정의
    rung_0: RigidObjectCfg = MISSING
    rung_1: RigidObjectCfg = MISSING
    rung_2: RigidObjectCfg = MISSING
    rung_3: RigidObjectCfg = MISSING
    rung_4: RigidObjectCfg = MISSING
    rung_5: RigidObjectCfg = MISSING
    rung_6: RigidObjectCfg = MISSING
    rung_7: RigidObjectCfg = MISSING
    rung_8: RigidObjectCfg = MISSING
    rung_9: RigidObjectCfg = MISSING
    rung_10: RigidObjectCfg = MISSING
    rung_11: RigidObjectCfg = MISSING
    rung_12: RigidObjectCfg = MISSING
    rung_13: RigidObjectCfg = MISSING
    rung_14: RigidObjectCfg = MISSING

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

    # target position generated from the base frame of the robot
    # target_torso_position = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name="torso_link",  # 로봇 torso의 링크 이름에 맞게 설정 (예: "base", "torso", "base_link")
    #     resampling_time_range=(10.0, 10.0),  # 10초마다 한 번 명령 갱신
    #     debug_vis=True,  # 시각화 마커 표시
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.0, 0.0),  # 사다리 기준으로 고정된 위치
    #         pos_y=(0.0, 0.0),
    #         pos_z=(1.5, 1.5),  # 사다리 높이에 해당하는 z range 1.5로 fixed
    #         roll=(0.0, 0.0),
    #         pitch=(0.0, 0.0),
    #         yaw=(0.0, 0.0),
    #     )
    # )

    # target position generated from the world frame of the robot
    target_torso_position = mdp.FixedPoseCommandCfg(
        asset_name="robot",
        body_name="torso_link",  # 로봇 torso의 링크 이름에 맞게 설정 (예: "base", "torso", "base_link")
        resampling_time_range=(10.0, 10.0),  # 10초마다 한 번 명령 갱신
        debug_vis=True,  # 시각화 마커 표시
        pos=(2.0, 0.0, 1.5),  # 월드 프레임에서 타겟 위치
        quat=(1.0, 0.0, 0.0, 0.0),  # 정방향을 보는 단위 quaternion
    )



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # for standing on ladder (offset of joint positions)
    # reset_robot_hip_pitch_joints = EventTerm(
    #     func=mdp.reset_selected_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (0.3, 0.3),
    #         "velocity_range": (0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_pitch_joint"),
    #     },
    # )
    # #
    # reset_robot_shoulder_pitch_joints = EventTerm(
    #     func=mdp.reset_selected_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (0.0, 0.0),
    #         "velocity_range": (0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_pitch_joint"),
    #     },
    # )
    #
    # reset_robot_elbow_joints = EventTerm(
    #     func=mdp.reset_selected_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (0.0, 0.0),
    #         "velocity_range": (0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*_elbow_joint"),
    #     },
    # )
    # reset_robot_left_wrist_roll_joints = EventTerm(
    #     func=mdp.reset_selected_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (0.0, 0.0),
    #         "velocity_range": (0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("robot", joint_names="left_wrist_roll_joint"),
    #     },
    # )
    #
    # reset_robot_right_wrist_roll_joints = EventTerm(
    #     func=mdp.reset_selected_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (0.0, 0.0),
    #         "velocity_range": (0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("robot", joint_names="right_wrist_roll_joint"),
    #     },
    # )
    # reset_robot_right_middle_0_joints = EventTerm(
    #     func=mdp.reset_selected_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (0.0, 0.0),
    #         "velocity_range": (0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("robot", joint_names="right_hand_middle_0_joint"),
    #     },
    # )
    # reset_robot_right_middle_1_joints = EventTerm(
    #     func=mdp.reset_selected_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (0.0, 0.0),
    #         "velocity_range": (0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("robot", joint_names="right_hand_middle_1_joint"),
    #     },
    # )
    # reset_robot_left_middle_0_joints = EventTerm(
    #     func=mdp.reset_selected_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (0.0, 0.0),
    #         "velocity_range": (0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("robot", joint_names="left_hand_middle_0_joint"),
    #     },
    # )
    # reset_robot_left_middle_1_joints = EventTerm(
    #     func=mdp.reset_selected_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (0.0, 0.0),
    #         "velocity_range": (0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("robot", joint_names="left_hand_middle_1_joint"),
    #     },
    # )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    target_position_tracking = RewTerm(
        func=mdp.sparse_pos_traget_reward, weight = 2.0, params={
            "command_name": "target_torso_position",
            "Tr" : 1.0
        }
    )
    # hand_contact_reward = RewTerm(
    #     func=mdp.hand_contact_time_reward, weight = 1.0, params={
    #         "threshold" : 1.0,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hand_.*"]),
    #     }
    # )


    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    stationary_penalty = RewTerm(
        func=mdp.penalty_stationary,
        weight=1.0,
        params={
            "threshold": 0.2,
            "penalty_val": -1.0
        }
    )
    torso_tilt_penalty = RewTerm(
        func=mdp.torso_rotation_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("robot")}
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionLadderEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
from __future__ import annotations

import math
# from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import ext_loco.tasks.loco_manipulation.EE_pose.mdp as mdp

##
# Pre-defined configs
##
# from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
# from .terrain.terrain import ROUGH_TERRAINS_CFG
from ext_loco.assets.limx import LIMX_SF_TRON1A_ARM

##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = LIMX_SF_TRON1A_ARM.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore
    print('ENV_REGEX_NS:', '{ENV_REGEX_NS}')
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_Link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=5.0,
    )
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # Training command: random world-frame target pose (point-wise command)
    EE_pose = mdp.UniformWorldPoseCommandCfg(
        asset_name="robot",
        body_name="eef_link",
        resampling_time_range=(6.0, 15.0),
        resampling_time_scale=(0.5, 5.0),
        make_quat_unique=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.8, 0.8),
            pos_y=(-0.8, 0.8),
            pos_z=(0.1, 2.0),       # 0726 by Edwin
            roll=(-1.5, 1.5),
            pitch=(-0.8, 0.8),
            yaw=(-1.5, 1.5),
        ),
        se3_decrease_vel_range=(0.5, 1.4),
        # Precision diagnostics only affect logging; they do not change observations or rewards.
        precision_metrics_enabled=True,
        precision_window_s=1.0,
        precision_trend_edge_s=0.2,
        precision_hold_s=0.5,
        precision_position_threshold=0.05,
        precision_orientation_threshold=math.radians(5.0),
        precision_mani_scale_threshold=0.1,
        debug_vis=True,
    )


@configclass
class CommandsCfgPlay:
    """Play command: full EE trajectory playback from pickle."""

    EE_pose = mdp.PicklePoseSequenceCommandCfg(
        asset_name="robot",
        body_name="eef_link",
        # World frame EE pose pkl from umi
        file_path="/home/phi5090ii/NYX/umi-on-tron-lab/IsaacLab_RFM/data/pushing.pkl",
        planar_center=True,
        # add_random_height_range=(-0.05, 0.05),
        # eef_link is a zero-offset alias of link6/J6, so no extra tip offset is applied.
        tip_offset_pos=(0.0, 0.0, 0.0),
        tip_offset_rpy=(0.0, 0.0, 0.0),
        episode_length_s=10,
        pose_latency=0.1,
        history_buffer_length=100,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        # from UniformPoseCommandCfg, must provide even for sequence task
        class_type=mdp.PicklePoseSequenceCommand,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.5, 0.5),
            pos_y=(-0.5, 0.5),
            pos_z=(0.7, 1.6),
            roll=(-3.2, 3.2),
            pitch=(-3.2, 3.2),
            yaw=(-3.2, 3.2),
        ),
    )


@configclass
class CommandsCfgCommandPlay:
    """Play command: fixed EE target without loading a pickle trajectory."""

    EE_pose = mdp.UniformWorldPoseCommandCfg(
        asset_name="robot",
        body_name="eef_link",
        resampling_time_range=(1.0e9, 1.0e9),
        resampling_time_scale=(1.0, 1.0),
        make_quat_unique=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.5, -0.5),
            pos_y=(0.0, 0.0),
            pos_z=(1.6, 1.6),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        se3_decrease_vel_range=(0.0, 0.0),
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["abad_[RL]_Joint","hip_[RL]_Joint","knee_[RL]_Joint","ankle_[RL]_Joint","J.*"], 
                                           scale=1.0, use_default_offset=True)
    # joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["wheel_[RL]_Joint"], 
    #                                        scale=5.0, use_default_offset=True)
    # joint_pos_ankle = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["ankle_[RL]_Joint"], 
    #                                        scale=1.0, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)) # 3 dim

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ) # 3 dim
        EE_pose_commands = ObsTerm(func=mdp.EE_commands_b) # 9 dim
        # ee_tracking_error = ObsTerm(
        #     func=mdp.ee_target_error_with_latency,
        #     params={"command_name": "EE_pose", "pos_scale": 10.0, "orn_scale": 1.5},
        #     noise=Unoise(n_min=-0.01, n_max=0.01)
        # ) # 9 dim (3 pos + 6 rot)

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_exclude_wheel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!ankle_).+")},
        )  # 12 dim
        
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)) # 14 dim
        actions = ObsTerm(func=mdp.last_action) # 14 dim
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )
        EE_pose = ObsTerm(
            func=mdp.EE_current_pose_b,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={"asset_cfg": SceneEntityCfg("robot", body_names="eef_link")},
        )  # 9
        EE_se3_distance_reference = ObsTerm(func=mdp.EE_se3_distance_ref) # 1

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    @configclass
    class ContactNetCfg(ObsGroup):
        """Observations for contactNet group."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))  # 3
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))  # 3
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_exclude_wheel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!ankle_).+")},
        )  # 12
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.8, n_max=0.8))  # 14
        
        joint_torque = ObsTerm(func=mdp.joint_torque, noise=Unoise(n_min=-0.8, n_max=0.8))  # 14
        EE_pose = ObsTerm(
            func=mdp.EE_current_pose_b,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={"asset_cfg": SceneEntityCfg("robot", body_names="eef_link")},
        )  # 9

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = False
    
    @configclass
    class NextObsCfg(ObsGroup):
        """Observations for contactNet group."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)  # 3
        projected_gravity = ObsTerm(func=mdp.projected_gravity)  # 6
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_exclude_wheel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!ankle_).+")},
        )  # 33
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)  # 55
        EE_pose = ObsTerm(
            func=mdp.EE_current_pose_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="eef_link")},
        )  # 86
        foot_position_b = ObsTerm(
            func=mdp.foot_position_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="ankle_[RL]_Link")},
        )  # 197

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for policy group."""

        # privileged observation terms (order preserved)
        
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # 112
        joint_torque = ObsTerm(func=mdp.joint_torque)  # 134
        # feet_contact_index = ObsTerm(
        #     func=mdp.feet_contact_index,
        #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.+"), "threshold": 10.0},
        # )  
        base_pos_z = ObsTerm(func=mdp.base_pos_z_rel)  # 145
        joint_acc = ObsTerm(func=mdp.joint_acc)  # 167
        EE_lin_vel = ObsTerm(
            func=mdp.body_any_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="eef_link"), "vel_type": "linear"},
        )  
        EE_ang_vel = ObsTerm(
            func=mdp.body_any_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="eef_link"), "vel_type": "angular"},
        )  
        
        # EE_se3_cb_error = ObsTerm(func=mdp.EE_se3_cb_error, scale=3.0)  # 152
        foot_position_b = ObsTerm(
            func=mdp.foot_position_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="ankle_.*")},
        )  
        # stationary terms
        joint_kp = ObsTerm(func=mdp.joint_kp)  # 219
        joint_kd = ObsTerm(func=mdp.joint_kd)  # 241
        base_mass_rel = ObsTerm(
            func=mdp.body_mass_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            },
        )
        rigid_body_materials = ObsTerm(
            func=mdp.rigid_body_materials,
            params={"asset_cfg": SceneEntityCfg(name="robot", body_names=["ankle_.+", "base_Link", "link6"])},
        ) 
 

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    next_obs: NextObsCfg = NextObsCfg()
    contactNet: ContactNetCfg = ContactNetCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1),       # by CZY
            "dynamic_friction_range": (0.4, 0.9),    # by CZY
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            # "mass_distribution_params": (-5.0, 5.0),
            "mass_distribution_params": (-3, 3.0),   # by CZY
            "operation": "add",
        },
    )

    prepare_quantity_for_tron1_piper = EventTerm(
        func=mdp.prepare_quantity_for_tron1_piper,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
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

    reset_tron_joints = EventTerm(
        func=mdp.reset_selected_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "abad_[RL]_Joint",
                    "hip_[RL]_Joint",
                    "knee_[RL]_Joint",
                    "ankle_[RL]_Joint",
                ],
            ),
            # Randomize only the TRON leg and ankle joints.
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.2, 0.2),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.1, 0.1), "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Safety
    safety_exp = RewTerm(
        func=mdp.safety_reward_exp, weight=2.0, params={"base_height_target": 0.8, "std": math.sqrt(0.5)}
    )
    # pose_product = RewTerm(
    #     func=mdp.pose_product_reward,
    #     weight=3.0,
    #     params={"pos_sigma": 0.6, "orn_sigma": 2.0, "command_name": "EE_pose"},
    # )
    # EE Tracking
    track_EE_position_exp = RewTerm(func=mdp.track_EE_position_exp, weight=4.0, params={"command_name": "EE_pose", "std": math.sqrt(0.5)})            #2.0 by Edwin
    track_EE_orientation_exp = RewTerm(func=mdp.track_EE_orientation_exp, weight=5.0, params={"command_name": "EE_pose", "std": math.sqrt(0.5)})      #3.0 by Edwin
    track_EE_pb = RewTerm(func=mdp.track_EE_pb, weight=15.0)
    track_EE_reference_exp = RewTerm(func=mdp.track_EE_reference_exp, weight=5.0, params={"std": math.sqrt(0.5), "init_value": 0.98})

    # Penalties
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_weighted_torques_l2 = RewTerm(
        func=mdp.weighted_joint_torques_l2,
        weight=-4.0e-5,
        params={
            "torque_weight": {
                "abad_L_Joint": 0.2,
                "hip_L_Joint": 0.2,
                "knee_L_Joint": 0.2,
                # "foot_L_Joint": 0.2,
                "abad_R_Joint": 0.2,
                "hip_R_Joint": 0.2,
                "knee_R_Joint": 0.2,
                # "foot_R_Joint": 0.2,
                "ankle_L_Joint": 3.0,
                "ankle_R_Joint": 3.0,
                "J1": 5.0,
                "J2": 5.0,
                "J3": 5.0,
                "J4": 15.0,
                "J5": 15.0,
                "J6": 15.0,
            }
        },
    )

    dof_weighted_power_l1 = RewTerm(
        func=mdp.weighted_joint_power_l1,
        weight=-2.5e-4,
        params={
            "power_weight": {
                "abad_L_Joint": 1.0,
                "hip_L_Joint": 1.0,
                "knee_L_Joint": 1.0,
                # "foot_L_Joint": 1.0,
                "abad_R_Joint": 1.0,
                "hip_R_Joint": 1.0,
                "knee_R_Joint": 1.0,
                # "foot_R_Joint": 1.0,
                "ankle_L_Joint": 1.0,
                "ankle_R_Joint": 1.0,
                "J1": 5.0,
                "J2": 5.0,
                "J3": 5.0,
                "J4": 5.0,
                "J5": 5.0,
                "J6": 5.0,
            }
        },
    )

    # dof_acc_l2 = RewTerm(
    #     func=mdp.joint_acc_l2, weight=-2.0e-6, params={"asset_cfg": SceneEntityCfg("robot", joint_names="J.*")}
    # )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.75) #1.0 By Edwin
    # action_smoothness = RewTerm(func=mdp.action_smoothness_penalty, weight=-5.0e-4)

    # -- optional penalties
    dof_vel_ankle_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=-5.0e-4, params={"asset_cfg": SceneEntityCfg("robot", joint_names="ankle_.+")}
    )
    dof_vel_non_ankle_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!ankle_).*")},
    )
    dof_vel_arm_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=-5.0e-4, params={"asset_cfg": SceneEntityCfg("robot", joint_names="J.*")}
    )
    # dof_power_l1 = RewTerm(func=mdp.dof_power_l1, weight=0.0)
    dof_non_ankle_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!ankle_).*")},
    )
    # joint_deviation_l1 = RewTerm(func=mdp.joint_deviation_l1, weight=0.0)
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    # base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=0.0, params={"target_height": 0.3})
    body_ee_alignment = RewTerm(
        func=mdp.body_ee_alignment,
        weight=-0.5,
        params={"joint_names": ["J1", "J5"]},
    )
    feet_contacts_reg = RewTerm(
        func=mdp.feet_contacts_reg,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_.*"), "threshold": 5.0},
    )
    # 2026-07-18 first-round foot-flat optimization:
    # Transplant the senior implementation's world-frame foot-normal error, but apply it only
    # to feet currently in contact and attenuate it with the real-time near-target standing gate.
    # This targets toe/heel loading during manipulation without forcing the swing foot to remain
    # flat during normal locomotion.  Keep the explicit L/R order for diagnostic logging.
    # W&B exposes Episode_Reward/foot_flat_l2 and Metrics/EE_pose/precision/feet/* after resets.
    foot_flat_l2 = RewTerm(
        func=mdp.foot_flat_l2,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["ankle_L_Link", "ankle_R_Link"],
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["ankle_L_Link", "ankle_R_Link"],
                preserve_order=True,
            ),
            "threshold": 5.0,
            "command_name": "EE_pose",
        },
    )
    foot_slip_l2 = RewTerm(
        func=mdp.foot_slip_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_.*"),
            "threshold": 5.0,
            "contact_grace_period": 0.1,
        },
    )
    legs_min_separation = RewTerm(
        func=mdp.legs_min_separation,
        weight=-2.0,
        params={"min_distance": 0.2, "body_names": ("ankle_L_Link", "ankle_R_Link"), "axis": "y"},#0.2
    )

    base_height = RewTerm( # by CZY
        func=mdp.base_height_rough_l2,
        weight=-0.5,#-2
        params={"target_height": 0.85, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1000)
    # alive = RewTerm(func=mdp.stay_alive, weight=2.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"), "threshold": 1.0},
    )
    knee_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="knee_[LR]_Link"),
            "threshold": 5.0,
        },
    )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation_stochastic,
        params={
            "limit_angle": math.pi * 0.4,
            "probability": 0.1,
        },  # Expect step = 1 / probability
    )

    ee_contact = DoneTerm(
        func=mdp.overcontact_stochastic,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="link6"),
            "threshold": 10.0,
            "probability": 0.1,
        },  # Expect step = 1 / probability
    )

    bad_height = DoneTerm(
        func=mdp.bad_height_stochastic,
        params={
            "limit_height": 0.4,
            "probability": 0.1,
        },  # Expect step = 1 / probability
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # tighten_ee_reward_sigma = CurrTerm(
    #     func=mdp.tighten_ee_reward_sigma,  # type: ignore
    #     params={
    #         "update_interval": 4000 * 24,  # every 80 iterations (24 steps each)
    #         "decay": 0.9,
    #         "pos_min": 0.05,
    #         "orn_min": 0.1,
    #     },
    # )
    pos_commands_ranges_level = CurrTerm(
        func=mdp.pos_commands_ranges_level,  # type: ignore
        params={
            "max_range": {"pos_x": (-0.8, 0.8), "pos_y": (-0.8, 0.8), "pos_z": (0.1, 2.0)},
            "update_interval": 80 * 24,  # 80 iterations * 24 steps per iteration
            "command_name": "EE_pose",
        },
    )
    orient_commands_ranges_level = CurrTerm(
        func=mdp.orient_commands_ranges_level, # type: ignore
        params={
            "update_interval": 80 * 24,  # 80 iterations * 24 steps per iteration
            "command_name": "EE_pose",
        },
    )
##
# Environment configuration
##


@configclass
class LimxEEposeRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs = 4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg | None = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
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
                
@configclass
class LimxEEposeRoughEnvCfg_PLAY(LimxEEposeRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Use trajectory command only for play/eval.
        self.commands = CommandsCfgPlay()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable observation noise for play
        self.observations.policy.enable_corruption = False

        # remove random disturbances
        del self.events.base_external_force_torque
        del self.events.push_robot

        # reset robot to a fixed default pose (no randomization) so the trajectory
        # is always anchored from the same EE starting position
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.reset_tron_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_tron_joints.params["velocity_range"] = (0.0, 0.0)

        # only end episodes on time_out (= trajectory complete); disable all other
        # terminations that would interrupt trajectory playback mid-way
        del self.terminations.base_contact
        del self.terminations.knee_contact
        del self.terminations.bad_orientation
        del self.terminations.ee_contact
        del self.terminations.bad_height

        # synchronise env episode length with the trajectory length from the command
        self.episode_length_s = self.commands.EE_pose.episode_length_s


@configclass
class LimxEEposeCommandEnvCfg_PLAY(LimxEEposeRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Use a fixed EE command for play/eval instead of a pickle trajectory.
        self.commands = CommandsCfgCommandPlay()

        # This task uses one fixed target pose.  The inherited training
        # curricula expand symmetric command ranges at reset, which turns the
        # fixed negative x target into an invalid (min > max) range.
        self.curriculum = None

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable observation noise for play
        self.observations.policy.enable_corruption = False

        # remove random disturbances
        del self.events.base_external_force_torque
        del self.events.push_robot

        # reset robot to a fixed default pose.
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.reset_tron_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_tron_joints.params["velocity_range"] = (0.0, 0.0)

        # keep the fixed-command play running until the user closes the app.
        del self.terminations.base_contact
        del self.terminations.knee_contact
        del self.terminations.bad_orientation
        del self.terminations.ee_contact
        del self.terminations.bad_height
        self.episode_length_s = 10

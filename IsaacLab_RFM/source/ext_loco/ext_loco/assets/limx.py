import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

import os

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))

##
# Configuration - Articulation.
##

LIMX_WF_TRON1A = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSETS_DIR}/WF_TRON1A/robot.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7 + 0.25),
        joint_pos={
            ".*_Joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=["abad_[RL]_Joint","hip_[RL]_Joint","knee_[RL]_Joint"],
            effort_limit=80.0,
            velocity_limit=20.0,
            stiffness=40.0,
            damping=2.5,
            friction=0.0
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_[RL]_Joint"],
            effort_limit=40.0,
            velocity_limit=40.0,
            stiffness=0.0,
            damping=2.5,
            friction=0.0
        ),
    },
)
"""Configuration of ANYmal-B robot using actuator-net."""

LIMX_WF_TRON1A_ARM = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ASSETS_DIR}/WF_TRON1A_ARM/robot_with_arm.usd",
        usd_path=f"{ASSETS_DIR}/WF_TRON1A_ARXR5ARM/robot_with_arxr5arm_usd/robot.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "wheel_L_Joint": 0.0,
            "wheel_R_Joint": 0.0,
            # arm
            "J1": 0.0,  # [rad]
            "J2": 0.5,  # [rad]
            "J3": 0.0,  # [rad]
            "J4": 0.0,  # [rad]
            "J5": 0.0,  # [rad]
            "J6": 0.0,  # [rad]
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=["abad_[RL]_Joint","hip_[RL]_Joint","knee_[RL]_Joint"],
            effort_limit=80.0,
            velocity_limit=20.0,
            stiffness=40.0,
            damping=1.8,
            friction=0.0
        ),
        "base_wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_[RL]_Joint"],
            effort_limit=40.0,
            velocity_limit=40.0,
            stiffness=0.0,
            damping=0.5,
            friction=0.33
            # min_delay=0,  # physics time steps (min: 5.0*0=0.0ms)
            # max_delay=3,  # physics time steps (max: 5.0*8=16.0ms)
        ),
        "arm_former_three": ImplicitActuatorCfg(
            joint_names_expr=["J1", "J2", "J3"],
            effort_limit=18.0,
            velocity_limit=3.14,
            stiffness=18.0,
            damping=1.0,
            friction=0.0,
            # min_delay=0,  # physics time steps (min: 5.0*0=0.0ms)
            # max_delay=3,  # physics time steps (max: 5.0*8=16.0ms)
        ),
    },
)

LIMX_SF_TRON1A_ARM = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ASSETS_DIR}/SF_TRON1A_ARXR5ARM/assembly.urdf",
        activate_contact_sensors=True,
        fix_base=False,
        make_instanceable=True,
        merge_fixed_joints=False,  # keep the camera-center eef_link body for EE tracking
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            drive_type="force",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0,
            fix_root_link=False
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,

            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "ankle_R_Joint": 0.0,
            "ankle_L_Joint": 0.0,
            # arm
            "J1": 0.0,  # [rad]
            "J2": 0.5,  # [rad]
            "J3": 0.0,  # [rad]
            "J4": 0.0,  # [rad]
            "J5": 0.0,  # [rad]
            "J6": 0.0,  # [rad]
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=["abad_[RL]_Joint","hip_[RL]_Joint","knee_[RL]_Joint"],
            effort_limit=80.0,
            velocity_limit=20.0,
            stiffness=40.0,
            damping=1.8,
            friction=0.0
        ),
        "base_ankles": ImplicitActuatorCfg(
            joint_names_expr=["ankle_[RL]_Joint"],
            effort_limit=40.0,
            velocity_limit=40.0,
            stiffness=45.0, # Position control for SF robot ankles
            damping=0.8,    # Increased damping for position control
            friction=0.33
        ),
        "arm_former_three": ImplicitActuatorCfg(
            joint_names_expr=["J1", "J2", "J3"],
            effort_limit=18.0,
            velocity_limit=3.14,
            stiffness=18.0,
            damping=1.0,
            friction=0.0,
        ),
        "arm_later_three": ImplicitActuatorCfg(
            joint_names_expr=["J4", "J5", "J6"],
            effort_limit=3.0,
            velocity_limit=3.9,
            stiffness=4.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)

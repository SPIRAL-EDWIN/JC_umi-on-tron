from __future__ import annotations

# 说明:
# 1) 本文件定义了 EE_pose 任务的奖励/惩罚函数（reward terms）。
# 2) 这些函数会在配置文件中的 RewardsCfg 里通过 RewTerm(func=...) 被调用并乘以对应权重。
# 3) 本文件只负责“计算每一项 reward 的原始值”，最终总奖励由 RewardManager 做加权求和。
# 4) 常见缩写:
#    - EE: End-Effector，末端执行器。
#    - SE3: 位姿空间（位置 + 朝向）。
#    - std/sigma: 容忍尺度，越小越严格（同样误差下 reward 会更低）。
#    - _loco_mani_scale: 机动/操作混合因子，0 偏操作，1 偏机动。
# 5) 张量约定:
#    - 第一维通常是并行环境数 num_envs。
#    - 函数返回 shape 一般为 (num_envs,)，表示每个并行环境一个标量 reward。

import torch
from typing import TYPE_CHECKING, Any, Dict, Sequence

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from ext_loco.utils.math import generate_sigmoid_scale, relexed_barrier_func, command_duration_mask


def tracking_EE_cb_penalty_l1(
    env: ManagerBasedRLEnv,
):
    """直接返回 EE 累积跟踪误差（L1 风格惩罚项）。

    参数:
        env: IsaacLab 的强化学习环境对象，内部保存了当前 step 的各类状态和缓存。

    返回:
        env.EE_se3_cb_error: 由其他模块提前计算好的累计 SE3 误差。
    """
    # 这里不再重复计算，直接使用环境中缓存好的误差。
    return env.EE_se3_cb_error


def safety_reward_exp(
    env: ManagerBasedRLEnv,
    std: float,
    base_height_target: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """计算“安全性”奖励（指数核）。

    这项奖励融合了足端几何偏差、车轮速度、底盘速度、姿态、底盘高度、机械臂中位姿态、EE 线/角速度。
    误差越小，exp(-error/std^2) 越接近 1；误差越大，越接近 0。

    参数:
        env: 环境对象。
        std: 指数核尺度，越小越严格。
        base_height_target: 底盘目标高度（世界系 z）。
        asset_cfg: 从 scene 里取哪个机器人资产，默认 "robot"。

    返回:
        shape=(num_envs,) 的安全尺度，按 _loco_mani_scale 混合locomotion安全和manipulation安全。
    """
    # 从场景中取出机器人对象（可为 RigidObject 或 Articulation）。
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # 该函数依赖 env._foot_link_ids（通常在 observation 里初始化）。
    # 若未准备好，直接抛错，避免静默用错数据。
    if not hasattr(env, "_foot_link_ids"):
        raise AttributeError("Foot link ids not set on env. Ensure observations.foot_position_b is called first.")
    # root_link 的四元数/位置扩展到 2 个脚，便于后续做逐脚坐标变换。
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    base_position = asset.data.root_link_pos_w.unsqueeze(1).expand(-1, 2, -1)

    # check if root_state equal
    # if asset.data.root_quat_w != base_quat:
    #     raise ValueError("Root state is not equal to base state")

    # 读取脚在世界系的位置，再变换到底盘坐标系（base frame）。
    foot_position = asset.data.body_pos_w[:, env._foot_link_ids, :]
    foot_position_b = math_utils.quat_rotate_inverse(base_quat, foot_position - base_position)
    # base_height = -foot_position_b[:, :, 2].mean(dim=-1) + env._foot_radius
    # 底盘高度（世界坐标 z）。
    base_height = asset.data.root_link_pos_w[:, 2]

    # 计算足端在 base frame 的平面误差（x,y）。
    foot_pos_error_b = foot_position_b[:, :, :2] - env._nominal_foot_position_b[:, :2]
    # inner_eight_condition:
    # 若脚位于名义位置“内侧”（y 方向跨过名义符号），则给予更强惩罚（后面 /0.1）。
    inner_eight_condition = ((env._nominal_foot_position_b[:, 1] > 0.0) * (foot_pos_error_b[:, :, 1] < 0.0)) | (
        (env._nominal_foot_position_b[:, 1] < 0.0) * (foot_pos_error_b[:, :, 1] > 0.0)
    )

    # y 方向误差分段归一化：内侧更严格，外侧相对宽松。
    foot_pos_error_b[:, :, 1] = torch.where(
        inner_eight_condition, foot_pos_error_b[:, :, 1] / 0.1, foot_pos_error_b[:, :, 1] / 0.2
    )
    # x 方向误差归一化。
    foot_pos_error_b[:, :, 0] = foot_pos_error_b[:, :, 0] / 0.2

    # 对两只脚、xy 两方向做 L1 汇总并裁剪，防止极端值主导梯度。
    foot_pos_error_b = torch.sum(torch.sum(foot_pos_error_b.abs(), dim=-1), dim=-1)
    foot_pos_error_b = torch.clamp(foot_pos_error_b, max=8.0)
    # 底盘姿态误差（通过 projected_gravity_b 近似 roll/pitch 偏移）。
    base_orient_error_roll = torch.abs(asset.data.projected_gravity_b[:, 1]) / 0.1
    base_orient_error_pitch = torch.abs(asset.data.projected_gravity_b[:, 0]) / 0.85
    # 底盘高度误差归一化。
    base_height_error = torch.abs((base_height - base_height_target)) / 0.2
    # arm_middle_pos = torch.tensor([0.0, 1.57, -2.7, 0.0, 0.0, 0.0], device=asset.device)
    # 机械臂中位姿态 = 每个关节上下限中点。
    arm_middle_pos = (
        asset.data.default_joint_limits[:, env._arm_joint_ids, 1]
        + asset.data.default_joint_limits[:, env._arm_joint_ids, 0]
    ) / 2.0
    # 机械臂偏离中位的 L1 误差（再做缩放）。
    arm_middle_error = (
        torch.sum(
            torch.abs(
                torch.index_select(
                    asset.data.joint_pos,
                    1,
                    torch.tensor(env._arm_joint_ids, device=asset.device),
                )
                - arm_middle_pos
            ),
            dim=1,
        )
        / 9.0
    )

    # 速度相关误差（车轮速度、底盘线/角速度、EE 线/角速度）。
    wheel_vel_error = (torch.sum(torch.abs(asset.data.joint_vel[:, env._wheels_joint_ids]), dim=1) / 3.0).clip(max=4)
    base_lin_vel_error = torch.norm(asset.data.root_link_lin_vel_b, p=2, dim=1) / 0.5
    base_ang_vel_error = torch.norm(asset.data.root_link_ang_vel_b, p=2, dim=1) / 1.2
    Ee_lin_vel_error = torch.norm(asset.data.body_lin_vel_w[:, env._ee_link_idx].squeeze(1), p=2, dim=1) / 0.6
    Ee_ang_vel_error = torch.norm(asset.data.body_ang_vel_w[:, env._ee_link_idx].squeeze(1), p=2, dim=1) / 3.14

    # 操作向安全误差（manipulation safety）加权融合后归一化。
    normalized_mani_error = (
        foot_pos_error_b  # 2
        + wheel_vel_error  # 2
        + base_lin_vel_error  # 1
        + base_ang_vel_error  # 1
        + base_height_error * 0.5  # 1
        + base_orient_error_roll * 0.5  # 0.5
        + base_orient_error_pitch * 0.25  # 0.5
        + arm_middle_error  # 1
        + Ee_lin_vel_error  # 1
        + Ee_ang_vel_error  # 1
    ) / 13.0

    # 机动向安全误差（locomotion safety）更强调足端/底盘稳定性。
    normalized_loco_error = (
        foot_pos_error_b / 2.0  # 2
        + base_orient_error_pitch  # 0.5
        + base_orient_error_roll  # 0.5
        # + arm_middle_error * 1.5  # 1
        + base_height_error  # 1
    ) / 6.0

    # for debug
    # total = foot_pos_error.mean() + base_orient_error.mean() + arm_middle_error.mean() + base_height_error.mean()
    # foot_error_ratio = foot_pos_error.mean() / total
    # base_error_ratio = base_orient_error.mean() / total
    # arm_error_ratio = arm_middle_error.mean() / total
    # height_error_ratio = base_height_error.mean() / total
    # print(
    #     f"foot_error_ratio: {foot_error_ratio}, \n base_error_ratio: {base_error_ratio}, \n arm_error_ratio: {arm_error_ratio}, \n height_error_ratio: {height_error_ratio}"
    # )
    # sigmoid_scale = generate_sigmoid_scale(
    #     mu=0.8, decay_length=0.8, x=env.command_manager.get_term("EE_pose").se3_distance_ref
    # )
    # 误差 -> 安全分数，指数核映射到 (0,1]。
    mani_safety_scale = torch.exp(-normalized_mani_error / std**2)

    loco_safety_scale = torch.exp(-normalized_loco_error / std**2)

    # 缓存到 env，供其他 reward 项复用（例如 EE 跟踪项会乘上这些安全尺度）。
    env._mani_safety_scale = mani_safety_scale + 0.4

    env._loco_safety_scale = loco_safety_scale + 0.4

    # 按 _loco_mani_scale 混合manipulation安全和locomotion安全。
    return mani_safety_scale * (1 - env._loco_mani_scale) + loco_safety_scale * env._loco_mani_scale


def track_EE_reference_exp(
    env: ManagerBasedRLEnv,
    std: float,
    relese_delta: float = 0.5,
    init_value: float = 0.99,
    command_name: str = "EE_pose",
) -> torch.Tensor:
    """跟踪 EE 参考距离（SE3）奖励。

    参数:
        env: 环境对象。
        std: 指数核尺度，越小越严格。
        relese_delta: 释放阈值，小误差区间内不惩罚（相当于 dead-zone）。
        init_value: 旧版保留参数，目前函数主体未使用。
        command_name: 命令项名称，默认 EE_pose。

    返回:
        shape=(num_envs,) 的奖励值。
    """

    # sigmoid_scale = generate_sigmoid_scale(
    #     mu=0.8, decay_length=0.8, x=env.command_manager.get_term("EE_pose").se3_distance_ref
    # )
    # contact_time = env.command_manager.get_term(command_name).metrics["contact_time"]
    # init_value = torch.ones(env.num_envs, device=env.device) * init_value
    # scale = torch.pow(init_value, contact_time)

    # 位置/姿态误差来自 command term 的 metrics。
    EE_position_error = env.command_manager.get_term(command_name).metrics["position_error"]
    EE_orientation_error = env.command_manager.get_term(command_name).metrics["orientation_error"]
    # 当前参考 SE3 距离。
    se3_distance_ref = env.command_manager.get_term("EE_pose").se3_distance_ref

    # 构造跟踪误差（位置误差权重更高，乘 2）。
    track_error = torch.abs(se3_distance_ref - EE_orientation_error - 2 * EE_position_error) - relese_delta

    # 小于 dead-zone 的误差置 0。
    track_error = torch.clamp(track_error, min=0.0)

    # 乘上 loco/mani 混合尺度与安全尺度。
    return torch.exp(-track_error / std**2) * env._loco_mani_scale * env._loco_safety_scale


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """统计并惩罚非期望接触次数。

    参数:
        env: 环境对象。
        threshold: 接触力阈值，超过即视为违规接触。
        sensor_cfg: 接触传感器配置（包含 sensor 名和 body_ids）。
    """
    # 读取接触传感器。
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 取历史窗口的接触力，便于做“是否曾超过阈值”的判定。
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    # 先对 force 向量做范数，再对时间窗口取 max，最后按 body 统计超阈值数量。
    is_contact_num = torch.sum(torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > threshold, dim=-1)
    # if getattr(env, "_feet_ids", None) is None:
    #     env._feet_ids = contact_sensor.find_bodies("wheel_.*")[0]
    # feet_is_contact = torch.sum(
    #     torch.max(torch.norm(contact_sensor.data.net_forces_w_history[:, :, env._feet_ids, :2], dim=-1), dim=1)[0]
    #     > threshold,
    #     dim=-1,
    # )
    # 每个环境返回“超阈值 body 个数”，通常会在配置里乘负权重。
    return is_contact_num  # + feet_is_contact


def weighted_joint_torques_l2(
    env: ManagerBasedRLEnv,
    torque_weight: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """按关节加权的扭矩惩罚（L2 平方）。

    参数:
        env: 环境对象。
        torque_weight: 字典，键是关节名/正则可解析名，值是该关节扭矩惩罚权重。
        asset_cfg: 机器人资产配置，默认 "robot"。

    返回:
        每个环境一个标量，值越大表示扭矩消耗越大。
    """
    # 取出关节系统数据。
    asset: Articulation = env.scene[asset_cfg.name]

    # 初始化与 applied_torque 同形状的容器，未配置权重的关节保持 0。
    weighted_torque = torch.zeros_like(asset.data.applied_torque)

    # 逐关节写入 w * torque^2。
    for joint_name, w in torque_weight.items():
        joint_idx, _ = asset.find_joints(joint_name)
        weighted_torque[:, joint_idx] = torch.square(asset.data.applied_torque[:, joint_idx]) * w

    # 在关节维度求和，得到每个环境的总惩罚。
    return torch.sum(weighted_torque, dim=1)


def weighted_joint_power_l1(
    env: ManagerBasedRLEnv,
    power_weight: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """按关节加权的功率惩罚（L1）。

    功率近似为 torque * joint_vel，这里取绝对值后加权求和。

    参数:
        env: 环境对象。
        power_weight: 字典，键是关节名，值是功率惩罚权重。
        asset_cfg: 机器人资产配置。
    """
    # 取关节状态。
    asset: Articulation = env.scene[asset_cfg.name]

    # 初始化容器，未参与惩罚的关节保持 0。
    weighted_power = torch.zeros_like(asset.data.applied_torque)

    # 写入每个关节的 |torque * vel| * w。
    for joint_name, w in power_weight.items():
        joint_idx = asset.find_joints(joint_name)[0]
        weighted_power[:, joint_idx] = (
            torch.abs(asset.data.applied_torque[:, joint_idx] * asset.data.joint_vel[:, joint_idx]) * w
        )
    # if hasattr(env, "_wheels_joint_ids"):
    #     if torch.rand(1) < 0.1:
    #         print(
    #             (
    #                 torch.abs(
    #                     asset.data.applied_torque[:, env._wheels_joint_ids]
    #                     * asset.data.joint_vel[:, env._wheels_joint_ids]
    #                     * 2
    #                 ).sum(dim=-1)
    #                 / torch.sum(weighted_power, dim=1)
    #             ).mean()
    #         )

    # 关节维度求和，得到每个环境的总功率惩罚。
    return torch.sum(weighted_power, dim=1)


def track_EE_position_exp(
    env: ManagerBasedRLEnv,
    std: float,
    init_value: float = 0.99,
    command_name: str = "EE_pose",
) -> torch.Tensor:
    """EE 位置跟踪奖励（指数核）。

    参数:
        env: 环境对象。
        std: 指数核尺度。越小越严格，误差稍大 reward 就会明显下降。
        init_value: 历史兼容参数，当前主公式未启用。
        command_name: 命令项名称，默认 "EE_pose"。

    返回:
        shape=(num_envs,) 的位置跟踪奖励。
    """
    # compute the error
    # contact_time = env.command_manager.get_term(command_name).metrics["contact_time"]
    # init_value = torch.ones(env.num_envs, device=env.device) * init_value
    # scale = torch.pow(init_value, contact_time)

    # 从命令管理器读取 EE 位置误差。
    EE_position_error = env.command_manager.get_term(command_name).metrics["position_error"]

    # 常规指数奖励项。
    normal = torch.exp(-EE_position_error / std**2)

    # 微调增强项：在小误差区域给更强激励（更“追求精确”）。
    micro_enhancement = torch.exp(-5 * EE_position_error / std**2)

    # 只在操作阶段（1 - _loco_mani_scale）有效，并乘操作安全尺度。
    return (normal + micro_enhancement) * (1 - env._loco_mani_scale) * env._mani_safety_scale


def track_EE_orientation_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "EE_pose",
    init_value: float = 0.99,
) -> torch.Tensor:
    """EE 朝向跟踪奖励（指数核）。

    参数:
        env: 环境对象。
        std: 朝向误差的指数核尺度。
        command_name: 命令项名称。
        init_value: 历史兼容参数，当前主公式未启用。

    返回:
        shape=(num_envs,) 的朝向跟踪奖励。
    """

    # contact_time = env.command_manager.get_term(command_name).metrics["contact_time"]
    # init_value = torch.ones(env.num_envs, device=env.device) * init_value
    # scale = torch.pow(init_value, contact_time)

    # contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # over_contact_mask = (
    #     torch.max(torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    #     > threshold
    # ).any(dim=-1)

    # 位置优先策略：位置误差越大，朝向奖励应适当衰减。
    EE_position_error = env.command_manager.get_term(command_name).metrics["position_error"]

    position_scale = torch.exp(-EE_position_error / 0.5)

    # 读取朝向误差。
    EE_orientation_error = env.command_manager.get_term(command_name).metrics["orientation_error"]

    # 常规朝向指数项。
    normal = torch.exp(-EE_orientation_error / std**2)

    # 小误差强化项。
    micro_enhancement = torch.exp(-5 * EE_orientation_error / std**2)

    # 位置优先 * 操作阶段门控 * 安全尺度。
    return (normal + micro_enhancement) * position_scale * (1 - env._loco_mani_scale) * env._mani_safety_scale


def reach_EE_position(
    env: ManagerBasedRLEnv, std: float, duration: float = 3, command_name: str = "EE_pose", init_value: float = 0.99
) -> torch.Tensor:
    """EE 位置到达奖励（按剩余时间窗口激活）。

    参数:
        env: 环境对象。
        std: 倒数二次核尺度。
        duration: 激活窗口时长，通常表示命令结束前多久开始强调“到位”。
        command_name: 命令项名称。
        init_value: 历史兼容参数，当前主公式未启用。

    返回:
        shape=(num_envs,) 的到达奖励。
    """
    # compute the error
    # sigmoid_scale = generate_sigmoid_scale(
    #     mu=0.8, decay_length=0.8, x=env.command_manager.get_term("EE_pose").se3_distance_ref
    # )
    # 读取位置误差与命令剩余时间。
    EE_position_error = env.command_manager.get_term(command_name).metrics["position_error"]
    time_left = env.command_manager.get_term(command_name).time_left
    # contact_time = env.command_manager.get_term(command_name).metrics["contact_time"]
    # init_value = torch.ones(env.num_envs, device=env.device) * init_value
    # scale = torch.pow(init_value, contact_time)
    # 时长掩码：仅在指定时间窗内生效。
    mask = command_duration_mask(time_left, duration)
    # 倒数二次核（相比指数核，对中等误差更平滑）。
    normal = (1.0 / (1.0 + torch.square(EE_position_error / std**2))) * mask
    # 更陡的精确度奖励。
    micro_enhancement = (1.0 / (1.0 + torch.square(4 * EE_position_error / std**2))) * mask
    # 仅操作阶段有效，并乘安全尺度。
    return (normal + micro_enhancement) * env._mani_safety_scale * (1 - env._loco_mani_scale)


def reach_EE_orient(
    env: ManagerBasedRLEnv, std: float, duration: float = 3, command_name: str = "EE_pose", init_value: float = 0.99
) -> torch.Tensor:
    """EE 朝向到达奖励（按剩余时间窗口激活）。

    参数:
        env: 环境对象。
        std: 倒数二次核尺度。
        duration: 激活窗口时长。
        command_name: 命令项名称。
        init_value: 历史兼容参数，当前主公式未启用。

    返回:
        shape=(num_envs,) 的朝向到达奖励。
    """
    # compute the error
    # sigmoid_scale = generate_sigmoid_scale(
    #     mu=0.8, decay_length=0.8, x=env.command_manager.get_term("EE_pose").se3_distance_ref
    # )
    # contact_time = env.command_manager.get_term(command_name).metrics["contact_time"]
    # init_value = torch.ones(env.num_envs, device=env.device) * init_value
    # scale = torch.pow(init_value, contact_time)
    # 读取朝向误差与剩余时间。
    EE_orient_error = env.command_manager.get_term(command_name).metrics["orientation_error"]
    time_left = env.command_manager.get_term(command_name).time_left
    # 时长掩码。
    mask = command_duration_mask(time_left, duration)
    # 倒数二次核主项 + 精确度增强项。
    normal = (1.0 / (1.0 + torch.square(EE_orient_error / std**2))) * mask
    micro_enhancement = (1.0 / (1.0 + torch.square(4 * EE_orient_error / std**2))) * mask
    # 操作阶段门控与安全尺度。
    return (normal + micro_enhancement) * env._mani_safety_scale * (1 - env._loco_mani_scale)


def track_EE_pb(env: ManagerBasedRLEnv, command_name: str = "EE_pose") -> torch.Tensor:
    """基于优化进展（progress-based）的 EE 跟踪奖励。

    参数:
        env: 环境对象。
        command_name: 命令项名称。

    返回:
        shape=(num_envs,) 的进展奖励。
    """
    # 当前位置到优化目标的距离。
    optim_pos_distance = env.command_manager.get_term(command_name).optim_pos_distance
    position_scale = torch.exp(-optim_pos_distance / 0.5)
    # 当前朝向到优化目标的距离。
    optim_orient_distance = env.command_manager.get_term(command_name).optim_orient_distance
    orient_scale = torch.exp(-optim_orient_distance / 0.5)
    # 与上一时刻相比的“进步量”。
    pos_improve = env.command_manager.get_term(command_name).pos_improvement
    orient_improve = env.command_manager.get_term(command_name).orient_improvement
    # 远距离时保留完整的进度奖励；近目标时衰减到 25%，避免策略为了立即
    # 减小 EE 误差而忽略底盘姿态、用快速小碎步抢进度。精定位仍由
    # track_EE_position_exp / track_EE_orientation_exp 负责。
    near_target_progress_scale = 0.25 + 0.75 * _walking_gate(
        env, command_name=command_name
    )
    return (
        (2 * pos_improve * position_scale + orient_improve * orient_scale)
        * env._loco_safety_scale
        * near_target_progress_scale
    )


def body_ee_alignment(env: ManagerBasedRLEnv, joint_names: Sequence[str] = ("J1", "J4")) -> torch.Tensor:
    """机械臂体轴对齐惩罚：约束关键关节接近默认位姿。

    参数:
        env: 环境对象。
        joint_names: 需要约束的关节名列表。

    返回:
        shape=(num_envs,) 的 L1 偏差和。
    """
    asset: Articulation = env.scene["robot"]
    # 找到目标关节 id。
    joint_ids = asset.find_joints(joint_names)[0]
    # 当前关节角 - 默认关节角。
    diff = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    # 对目标关节做绝对值求和。
    return torch.sum(diff.abs(), dim=1)


def contact_ankle_deviation_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=("ankle_L_Joint", "ankle_R_Joint")
    ),
) -> torch.Tensor:
    """接触脚踝姿态惩罚：抑制脚接触地面时用脚尖/脚跟极端姿态蹭地。

    只惩罚正在接触地面的那只脚踝，不约束摆动脚，避免影响正常抬脚。
    该项不按远近门控：无论远近，支撑脚都应避免脚尖/脚跟极端承重。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    ankle_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    penalty = torch.square(ankle_error) * in_contact
    # 无论远近，只要脚在支撑就不应以脚尖/脚跟的极端踝姿态承重。
    return torch.sum(penalty, dim=1)


def foot_flat_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float = 5.0,
    command_name: str = "EE_pose",
) -> torch.Tensor:
    """近目标接触脚平整惩罚，抑制脚尖/脚跟承重。

    该项沿用学长版本的几何定义：将每只脚 link 的局部 z 轴旋转到世界系，
    用其 xy 分量平方和（等价于脚面倾角的 ``sin(theta)^2``）衡量不平整程度。
    相比原版本，本实现增加两个门控：

    1. 仅惩罚当前接触力超过 ``threshold`` 的脚，不约束摆动脚；
    2. 乘实时 ``_standing_gate``，只在 EE 接近目标时重点要求平脚站稳。

    ``asset_cfg`` 与 ``sensor_cfg`` 必须按相同顺序各解析出左右两只 ankle link。
    函数同时缓存左右脚倾角、接触状态和 standing gate，供 W&B 精度日志聚合；
    这些诊断缓存不参与 reward 计算。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    current_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    if foot_quat_w.shape[1] != 2 or current_forces.shape[1] != 2:
        raise ValueError(
            "foot_flat_l2 expects exactly two foot bodies in matching left/right order; "
            f"got {foot_quat_w.shape[1]} asset bodies and {current_forces.shape[1]} sensor bodies."
        )

    local_z = torch.zeros(
        foot_quat_w.shape[0],
        foot_quat_w.shape[1],
        3,
        device=asset.device,
        dtype=foot_quat_w.dtype,
    )
    local_z[..., 2] = 1.0
    foot_z_w = math_utils.quat_apply(
        foot_quat_w.reshape(-1, 4), local_z.reshape(-1, 3)
    ).reshape_as(local_z)

    per_foot_flatness_error = torch.sum(torch.square(foot_z_w[..., :2]), dim=-1)
    contact_mask = torch.norm(current_forces, dim=-1) > threshold
    standing_gate = _standing_gate(env, command_name=command_name)

    # 供 commands.py 在 episode reset 时做精确的加权聚合。显式 detach，避免
    # 诊断缓存持有无用的计算图；reward 本身仍使用上面的原始张量。
    foot_tilt_rad = torch.acos(torch.clamp(foot_z_w[..., 2], min=-1.0, max=1.0))
    env._foot_flat_diagnostics = {  # type: ignore[attr-defined]
        "tilt_rad": foot_tilt_rad.detach(),
        "contact_mask": contact_mask.detach(),
        "standing_gate": standing_gate.detach(),
    }

    return (
        torch.sum(per_foot_flatness_error * contact_mask.float(), dim=1)
        * standing_gate
    )


def foot_slip_l2(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_grace_period: float = 0.1,
) -> torch.Tensor:
    """Penalize horizontal foot velocity during contact and brief contact losses.

    The grace period prevents a policy from avoiding the penalty by rapidly
    alternating between contact and no-contact states.  Once a foot has been
    airborne longer than the grace period, it is treated as a normal swing foot
    and is no longer subject to the slip penalty.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    is_contact = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > threshold
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    recently_lost_contact = (current_air_time > 0.0) & (current_air_time <= contact_grace_period)
    contact_or_grace = is_contact | recently_lost_contact
    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(torch.square(torch.norm(foot_vel_xy, dim=-1)) * contact_or_grace, dim=1)


def _walking_gate(
    env: ManagerBasedRLEnv,
    command_name: str = "EE_pose",
    position_error_threshold: float = 0.3,
    position_error_transition: float = 0.25,
) -> torch.Tensor:
    """根据实时末端位置误差平滑切换到行走阶段。

    小于约 0.4 m 时接近 0（双脚站稳微调）；大于约 0.6 m 时接近 1
    （正常抬脚行走）。使用实时误差而不是随时间递减的 _loco_mani_scale。
    """
    position_error = env.command_manager.get_term(command_name).metrics["position_error"]
    return generate_sigmoid_scale(
        mu=position_error_threshold,
        decay_length=position_error_transition,
        x=position_error,
    )


def _standing_gate(env: ManagerBasedRLEnv, command_name: str = "EE_pose") -> torch.Tensor:
    """与 _walking_gate 互补：近目标时鼓励双脚稳定支撑。"""
    return 1.0 - _walking_gate(env, command_name=command_name)


def near_target_body_stability_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "EE_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """近目标底盘稳定惩罚，不包含 base height 约束。

    同时抑制底盘平面速度、竖直速度、roll/pitch 角速度与倾斜。
    不惩罚 yaw 角速度，保留近目标时转向斜前方目标的能力。
    该项只在实时 EE 位置误差较小时平滑生效，不限制远距离行走。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    planar_velocity = torch.sum(
        torch.square(asset.data.root_link_lin_vel_b[:, :2]), dim=1
    )
    vertical_velocity = torch.square(asset.data.root_link_lin_vel_b[:, 2])
    roll_pitch_angular_velocity = torch.sum(
        torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1
    )
    tilt = torch.sum(
        torch.square(asset.data.projected_gravity_b[:, :2]), dim=1
    )

    penalty = (
        planar_velocity
        + 0.25 * vertical_velocity
        + 0.25 * roll_pitch_angular_velocity
        + 2.0 * tilt
    )
    return penalty * _standing_gate(env, command_name=command_name)


def near_target_leg_joint_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "EE_pose",
) -> torch.Tensor:
    """近目标腿部关节速度惩罚，用于抑制站立时髋、膝、踝关节来回抖动。"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    # 用均值而非求和，使权重不依赖选中的关节数量。
    penalty = torch.mean(torch.square(joint_velocity), dim=1)
    return penalty * _standing_gate(env, command_name=command_name)


def legs_min_separation(
    env: ManagerBasedRLEnv,
    min_distance: float = 0.3,
    body_names: tuple[str, str] = ("ankle_L_Link", "ankle_R_Link"),
    axis: str = "y",
) -> torch.Tensor:
    """Penalty if the two legs (ankles/feet) get closer than a target distance along one base-frame axis."""
    asset: Articulation = env.scene["robot"]
    # cache leg link ids
    if not hasattr(env, "_leg_link_ids"):
        env._leg_link_ids = asset.find_bodies(list(body_names))[0]  # type: ignore
    ids = env._leg_link_ids
    axis_id = {"x": 0, "y": 1}[axis]
    foot_pos_w = asset.data.body_pos_w[:, ids, :]  # (N,2,3)
    base_quat_w = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    base_pos_w = asset.data.root_link_pos_w.unsqueeze(1).expand(-1, 2, -1)
    foot_pos_b = math_utils.quat_rotate_inverse(base_quat_w, foot_pos_w - base_pos_w)
    dist = torch.abs(foot_pos_b[:, 0, axis_id] - foot_pos_b[:, 1, axis_id])
    penalty = torch.clamp(min_distance - dist, min=0.0)
    return penalty


def pose_product_reward(
    env: ManagerBasedRLEnv,
    pos_sigma: float,
    orn_sigma: float,
    command_name: str = "EE_pose",
) -> torch.Tensor:
    """UMI 风格位姿乘积奖励：位置项 * 朝向项。

    参数:
        env: 环境对象。
        pos_sigma: 位置项尺度（越大越宽容）。
        orn_sigma: 朝向项尺度（越大越宽容）。
        command_name: 命令项名称。

    返回:
        shape=(num_envs,) 的位姿联合奖励。
    """
    term = env.command_manager.get_term(command_name)
    # 读取位置与朝向误差。
    pos_err = term.metrics["position_error"]
    orn_err = term.metrics["orientation_error"]
    # 分别计算两项奖励。
    pos_reward = torch.exp(-(pos_err ** 2) / pos_sigma)
    orn_reward = torch.exp(-orn_err / orn_sigma)
    # 乘积形式会强调“位置和朝向都要好”。
    return pos_reward * orn_reward


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """关节速度惩罚（L2 平方）。

    参数:
        env: 环境对象。
        asset_cfg: 资产配置，可通过 joint_ids/joint_names 选择参与惩罚的关节。

    返回:
        shape=(num_envs,) 的关节速度平方和。
    """
    # 取机器人关节系统。
    asset: Articulation = env.scene[asset_cfg.name]
    # 对选择关节做 velocity^2 后求和。
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def stay_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """存活奖励：每步返回常数 1。"""
    # 常用于给 agent 一个基础正激励，权重通常较小。
    return torch.ones(env.num_envs, device=env.device)


def base_height_rough_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """底盘高度惩罚（适配 rough/射线地形）。

    参数:
        env: 环境对象。
        target_height: 目标高度（米）。
        sensor_cfg: RayCaster 传感器配置（用于估计地面高度）。
        asset_cfg: 资产配置。

    返回:
        shape=(num_envs,) 的高度偏差平方。
    """
    # 读取机器人与地面射线传感器。
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # 机器人根部高度 - 射线命中高度 => 相对地面高度。
    height = asset.data.root_link_pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[:, :, 2]
    # sensor.data.ray_hits_w can be inf, so we clip it to avoid NaN
    height = torch.nan_to_num(height, nan=target_height, posinf=target_height, neginf=target_height)
    # 以多条射线平均高度与目标高度的偏差作为惩罚。
    return torch.square(height.mean(dim=1) - target_height)


def flat_orientation_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """底盘竖直朝向惩罚（L2）。

    参数:
        env: 环境对象。
        asset_cfg: 资产配置。

    返回:
        shape=(num_envs,) 的姿态惩罚。
    """
    # 读取机器人状态。
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # projected_gravity_b[:, 2] 理想值约为 -1（重力朝 base -z）。
    return torch.square(asset.data.projected_gravity_b[:, 2] + 1)


def dof_power_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """关节机械功率惩罚（L2）。

    参数:
        env: 环境对象。
        asset_cfg: 参与惩罚的关节配置。

    返回:
        shape=(num_envs,) 的 |torque*vel|^2 求和。
    """
    # 读取关节系统。
    asset: Articulation = env.scene[asset_cfg.name]
    # 机械功率近似 torque*velocity，对其平方后求和。
    return torch.sum(
        torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]),
        dim=1,
    )


def dof_power_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """关节机械功率惩罚（L1）。

    参数:
        env: 环境对象。
        asset_cfg: 参与惩罚的关节配置。

    返回:
        shape=(num_envs,) 的 |torque*vel| 绝对值求和。
    """
    # 读取关节系统。
    asset: Articulation = env.scene[asset_cfg.name]
    # 机械功率绝对值求和。
    return torch.sum(
        torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]),
        dim=1,
    )


def feet_regulation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    feet_radius: float,
    feet_height_target: float,
) -> torch.Tensor:
    """足端调节惩罚：抑制低高度时的高速摆动。

    参数:
        env: 环境对象。
        asset_cfg: 足端 body 配置。
        feet_radius: 足端半径，用于由 link 高度近似接触高度。
        feet_height_target: 高度尺度参数。

    返回:
        shape=(num_envs,) 的足端速度惩罚。
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # 足端高度（简单近似并裁剪到 [0,1] 以稳定数值）。
    feet_height = torch.clip(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - feet_radius, 0, 1
    )  # TODO: change to the height relative to the vertical projection of the terrain
    # 足端平面速度。
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]

    # 高度越低，exp(-h/scale) 越大，低空高速动作惩罚更强。
    reward = torch.sum(
        torch.exp(-feet_height / feet_height_target) * torch.square(torch.norm(feet_vel_xy, dim=-1)), dim=1
    )
    # 近距离保留 30% 的防拖脚/低空高速足端约束，允许慢速小步微调，
    # 但不鼓励贴地快速小碎步。
    return reward * (0.3 + 0.7 * _walking_gate(env))


def feet_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float,
    feet_radius: float,
    std: float,
) -> torch.Tensor:
    """足端离地高度奖励：鼓励摆动脚抬到合适高度。

    这项只在单脚支撑时，对另一只摆动脚生效，用于给“正常抬脚”一个更直接的正反馈。
    在 flat 地面上，足底高度用 body 世界 z 减 feet_radius 近似。
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    feet_height = torch.clip(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - feet_radius, 0.0, 1.0
    )
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    swing_foot = (~in_contact) & single_stance.unsqueeze(-1)

    height_error = torch.abs(feet_height - target_height)
    reward = torch.exp(-height_error / std) * swing_foot
    # 只有目标较远、确实需要行走时才奖励摆动脚离地。
    return torch.sum(reward, dim=1) * _walking_gate(env)


def feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_air_time: float,
    max_air_time: float,
) -> torch.Tensor:
    """鼓励足端腾空时间处于指定范围内。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    # 范围内为正，范围外为负，中间值奖励最大
    duration_reward = torch.minimum(
        last_air_time - min_air_time,
        max_air_time - last_air_time,
    )

    # 防止异常长腾空产生过大的负值
    duration_reward = torch.clamp(duration_reward, min=-0.25)

    reward = torch.sum(duration_reward * first_contact, dim=1)
    # 近目标微调不要求单脚腾空，避免为了拿奖励而掂脚。
    return reward * _walking_gate(env)
def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """在移动阶段鼓励稳定的单脚支撑。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)

    # 恰好一只脚接触地面
    single_stance = torch.sum(in_contact.int(), dim=1) == 1

    reward = torch.min(
        torch.where(
            single_stance.unsqueeze(-1),
            in_mode_time,
            0.0,
        ),
        dim=1,
    )[0]

    reward = torch.clamp(reward, max=threshold)

    # 只在目标较远时奖励单脚支撑；近目标应优先双脚平放微调。
    return reward * _walking_gate(env)


def feet_contacts_reg(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    stable_time: float = 0.2,
) -> torch.Tensor:
    """近目标双脚稳定接触奖励。

    不再使用历史窗口的最大接触力，避免左右脚快速交替点地时被误判为
    "双脚稳定接触"。只有当前两脚同时接触，且连续接触时间足够长时才给满奖励。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    current_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    current_contact = torch.norm(current_forces, dim=-1) > threshold
    both_feet_contact = torch.all(current_contact, dim=1)

    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    shortest_contact_time = torch.min(contact_time, dim=1)[0]
    stable_contact_scale = torch.clamp(
        shortest_contact_time / stable_time, min=0.0, max=1.0
    )
    reward_mani = both_feet_contact.float() * stable_contact_scale

    # 位置误差用于附加缩放（位置偏差大时降低该奖励影响）。
    EE_position_error = env.command_manager.get_term("EE_pose").metrics["position_error"]

    position_scale = torch.exp(-EE_position_error / 0.5)

    # sigmoid_scale = generate_sigmoid_scale(
    #     mu=0.8, decay_length=0.8, x=env.command_manager.get_term("EE_pose").se3_distance_ref
    # )

    # 近目标时鼓励双脚连续接触；使用实时位置误差，不受计时式 loco/mani scale 影响。
    return reward_mani * _standing_gate(env) * position_scale


def fly_penalty(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """离地惩罚标志：若所有脚都无接触则为 True。"""
    # 读取接触传感器。
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 当前时刻接触力。
    net_contact_forces = contact_sensor.data.net_forces_w

    # 超阈值接触脚数量。
    feet_contact_num = torch.sum(torch.norm(net_contact_forces, dim=-1) > threshold, dim=-1)
    # 完全离地。
    reward = feet_contact_num == 0

    return reward


def no_fly(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """非离地奖励：至少有一个脚接触地面。"""
    return 1 - fly_penalty(env, threshold, sensor_cfg).float()


class ActionSmoothnessPenaltyWrapper:
    """
    A wrapper class for calculating action smoothness penalty.

    The main purposes of this wrapper are:
    1. To maintain state across multiple calls (prev_action and prev_prev_action).
    2. To calculate a smoothness penalty based on the current, previous, and
       two-steps-ago actions.
    3. To provide a serializable interface compatible with IsaacLab's YAML
       configuration system.
    """

    def __init__(self):
        # t-2 时刻动作。
        self.prev_prev_action = None
        # t-1 时刻动作。
        self.prev_action = None
        # 供配置系统识别的可调用名字。
        self.__name__ = "action_smoothness_penalty"

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """动作平滑惩罚：抑制动作二阶差分（近似 jerk）。"""
        # 当前时刻动作。
        current_action = env.action_manager.action.clone()

        # 第一次调用：仅初始化历史，不惩罚。
        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        # 第二次调用：补齐两帧历史，不惩罚。
        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        # 二阶差分: a_t - 2*a_{t-1} + a_{t-2}，对动作维求和。
        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        # 更新历史缓存，供下一步使用。
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        # episode 起始前几步不计惩罚，避免初始条件导致的偏置。
        startup_env_musk = env.episode_length_buf < 3
        penalty[startup_env_musk] = 0

        return penalty


# 导出实例，供 RewardsCfg 里直接作为 func 引用。
action_smoothness_penalty = ActionSmoothnessPenaltyWrapper()

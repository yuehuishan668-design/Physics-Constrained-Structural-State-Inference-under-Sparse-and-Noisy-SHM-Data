"""
File location:
    opensees_models/run_dynamic_analysis.py

Purpose:
    Run transient dynamic analysis of the 2D frame created in frame_base.py.

Research role:
    This file generates one structural response sample:
        input candidate:
            floor acceleration time histories
        label:
            story-level stiffness degradation vector

Important OpenSees note:
    With UniformExcitation, nodal displacement/velocity/acceleration responses are relative responses.
    Therefore this script records both:
        1. relative floor acceleration from OpenSees,
        2. approximate absolute floor acceleration = relative floor acceleration + ground acceleration.

中文说明：
    文件位置：
        opensees_models/run_dynamic_analysis.py

    文件目的：
        对 frame_base.py 中创建的二维框架结构进行瞬态动力时程分析。

    研究作用：
        本文件用于生成一个结构响应样本：
            可作为模型输入的数据：
                各楼层加速度时程；
            可作为标签的数据：
                层级刚度退化向量。

    重要 OpenSees 说明：
        使用 UniformExcitation 时，OpenSees 返回的节点位移、速度、加速度通常是相对响应。
        因此本脚本同时记录：
            1. OpenSees 输出的楼层相对加速度；
            2. 近似绝对加速度 = 相对加速度 + 地震动输入加速度。
"""

from __future__ import annotations
# Postpone type-hint evaluation for cleaner annotations.
# 中文：延后类型注解的求值，使函数签名更简洁，也减少类型引用导致的兼容性问题。

from pathlib import Path
# `Path` provides object-oriented file path operations.
# It is safer and cleaner than raw string paths.
# 中文：`Path` 提供面向对象的文件路径操作方式，比直接使用字符串路径更安全、更清晰。

from typing import Dict, List
# Type hints for dictionaries and lists.
# 中文：`Dict` 和 `List` 是字典与列表的类型提示，用于增强代码可读性和静态检查能力。

import numpy as np
# `numpy` is used for arrays, ground motion generation, and saving .npz files.
# 中文：`numpy` 用于数组计算、地震动生成，以及保存 `.npz` 数据文件。

import matplotlib.pyplot as plt
# `matplotlib.pyplot` is used to plot response curves for sanity checking.
# 中文：`matplotlib.pyplot` 用于绘制响应曲线，帮助快速检查分析结果是否合理。

import openseespy.opensees as ops
# OpenSeesPy command module.
# 中文：OpenSeesPy 的核心命令模块，后续所有 OpenSees 分析命令都通过 `ops` 调用。

from opensees_models.frame_base import FrameConfig, build_2d_frame, compute_modal_periods
# Import the frame builder and configuration from the base model file.
# 中文：从基础模型文件中导入框架配置类、建模函数和模态周期计算函数。


def create_sine_ground_motion(
    duration: float = 20.0,
    dt: float = 0.01,
    amplitude_g: float = 0.10,
    frequency_hz: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Create a simple sine-wave ground acceleration for debugging.

    This is NOT the final earthquake dataset.
    It is only used to test whether the OpenSees model and transient analysis pipeline work.

    Parameters:
        duration:
            Total duration, in seconds.
        dt:
            Time step, in seconds.
        amplitude_g:
            Acceleration amplitude in g.
        frequency_hz:
            Sine frequency in Hz.

    Returns:
        Dictionary with:
            t: time vector
            accel_g: ground acceleration in g

    中文说明：
        创建一个简单的正弦波地面加速度，用于调试。

        这不是最终的地震动数据集。
        它只是用于测试 OpenSees 模型和瞬态分析流程是否可以正常运行。

        参数：
            duration：
                地震动总时长，单位是秒。
            dt：
                时间步长，单位是秒。
            amplitude_g：
                加速度幅值，单位是 g。
            frequency_hz：
                正弦波频率，单位是 Hz。

        返回：
            一个字典，包含：
                t：时间向量；
                accel_g：单位为 g 的地面加速度。
    """

    t = np.arange(0.0, duration, dt)
    # `np.arange(start, stop, step)` creates a time vector from 0 to duration with interval dt.
    # 中文：`np.arange(start, stop, step)` 会按照步长 `dt` 生成从 0 到 `duration` 的时间向量。

    accel_g = amplitude_g * np.sin(2.0 * np.pi * frequency_hz * t)
    # Sine ground acceleration in unit g.
    # `2*pi*f*t` is the phase angle of a sine wave.
    # 中文：这里生成单位为 g 的正弦地面加速度；`2*pi*f*t` 是正弦波的相位角。

    return {
        "t": t,
        "accel_g": accel_g,
    }


def add_rayleigh_damping(
    damping_ratio: float = 0.02,
    n_modes_for_damping: int = 2,
) -> Dict[str, float]:
    """
    Add Rayleigh damping to the current OpenSees model.

    Damping matrix form:
        C = alphaM * M + betaKinit * K_initial

    Parameters:
        damping_ratio:
            Target damping ratio, commonly 2% or 5% for initial testing.
        n_modes_for_damping:
            Use the first two modes to calculate Rayleigh coefficients.

    Returns:
        Dictionary containing alpha_m and beta_k_init.

    中文说明：
        为当前 OpenSees 模型添加 Rayleigh 阻尼。

        阻尼矩阵形式：
            C = alphaM * M + betaKinit * K_initial

        参数：
            damping_ratio：
                目标阻尼比，初步测试中常用 2% 或 5%。
            n_modes_for_damping：
                用于计算 Rayleigh 阻尼系数的模态阶数，这里通常使用前两阶。

        返回：
            一个字典，包含质量比例阻尼系数 alpha_m 和初始刚度比例阻尼系数 beta_k_init。
    """

    eigen_values = ops.eigen(n_modes_for_damping)
    # Compute eigenvalues of the current model.
    # We need the first two frequencies to calculate Rayleigh coefficients.
    # 中文：计算当前模型的特征值；后续需要利用前两阶频率计算 Rayleigh 阻尼系数。

    omega_1 = np.sqrt(eigen_values[0])
    # First circular frequency, rad/s.
    # 中文：第一阶圆频率，单位是 rad/s。

    omega_2 = np.sqrt(eigen_values[1])
    # Second circular frequency, rad/s.
    # 中文：第二阶圆频率，单位是 rad/s。

    alpha_m = 2.0 * damping_ratio * omega_1 * omega_2 / (omega_1 + omega_2)
    # Mass-proportional damping coefficient.
    # 中文：质量比例阻尼系数，对应 Rayleigh 阻尼中的 `alphaM`。

    beta_k_init = 2.0 * damping_ratio / (omega_1 + omega_2)
    # Initial-stiffness-proportional damping coefficient.
    # 中文：初始刚度比例阻尼系数，对应 Rayleigh 阻尼中的 `betaKinit`。

    ops.rayleigh(alpha_m, 0.0, beta_k_init, 0.0)
    # `ops.rayleigh(alphaM, betaK, betaKinit, betaKcomm)` assigns Rayleigh damping.
    # Here:
    #     alphaM = alpha_m,
    #     betaK = 0.0,
    #     betaKinit = beta_k_init,
    #     betaKcomm = 0.0.
    # 中文：`ops.rayleigh(alphaM, betaK, betaKinit, betaKcomm)` 用于给模型施加 Rayleigh 阻尼。
    # 中文：这里设置 `alphaM = alpha_m`，`betaK = 0.0`，`betaKinit = beta_k_init`，`betaKcomm = 0.0`。

    return {
        "alpha_m": float(alpha_m),
        "beta_k_init": float(beta_k_init),
    }


def setup_transient_analysis() -> None:
    """
    Define the numerical solver settings for transient dynamic analysis.

    中文说明：
        定义瞬态动力分析所需的数值求解器设置。
    """

    ops.constraints("Transformation")
    # `constraints("Transformation")` defines how constraints are enforced.
    # It is robust for fixed supports and possible future multi-point constraints.
    # 中文：`constraints("Transformation")` 定义约束处理方式；它对固定支座以及后续可能加入的多点约束较稳健。

    ops.numberer("RCM")
    # `numberer("RCM")` uses Reverse Cuthill-McKee numbering.
    # It reduces matrix bandwidth and can improve efficiency.
    # 中文：`numberer("RCM")` 使用 Reverse Cuthill-McKee 编号方法，可以减小矩阵带宽并提高求解效率。

    ops.system("BandGeneral")
    # `system("BandGeneral")` selects a general banded linear equation solver.
    # 中文：`system("BandGeneral")` 选择通用带状矩阵线性方程求解器。

    ops.test("NormDispIncr", 1.0e-8, 20)
    # `test` defines the convergence test.
    # `"NormDispIncr"` checks the norm of displacement increments.
    # `1.0e-8` is the convergence tolerance.
    # `20` is the maximum number of iterations per step.
    # 中文：`test` 定义收敛判据；`"NormDispIncr"` 表示检查位移增量范数；`1.0e-8` 是收敛容差；`20` 是每个时间步允许的最大迭代次数。

    ops.algorithm("Newton")
    # `algorithm("Newton")` uses Newton-Raphson iterations.
    # For an elastic model it should be stable.
    # 中文：`algorithm("Newton")` 使用 Newton-Raphson 迭代算法；对于当前线弹性模型通常应当稳定。

    ops.integrator("Newmark", 0.5, 0.25)
    # `integrator("Newmark", gamma, beta)` selects Newmark time integration.
    # gamma = 0.5 and beta = 0.25 are average-acceleration parameters.
    # This is unconditionally stable for linear systems.
    # 中文：`integrator("Newmark", gamma, beta)` 选择 Newmark 时间积分方法；`gamma = 0.5`、`beta = 0.25` 是平均加速度法参数；该组合对线性系统无条件稳定。

    ops.analysis("Transient")
    # `analysis("Transient")` tells OpenSees to perform time-history analysis.
    # 中文：`analysis("Transient")` 告诉 OpenSees 执行瞬态时程分析。


def run_time_history(
    damage: List[float],
    ground_accel_g: np.ndarray,
    dt: float,
    config: FrameConfig | None = None,
    damping_ratio: float = 0.02,
    g: float = 9.81,
) -> Dict[str, np.ndarray]:
    """
    Run time-history analysis for one damage case and one ground motion.

    Parameters:
        damage:
            Story-level stiffness degradation vector.
        ground_accel_g:
            Ground acceleration array in unit g.
        dt:
            Time step of the ground motion.
        config:
            Frame configuration.
        damping_ratio:
            Target Rayleigh damping ratio.
        g:
            Gravitational acceleration used to convert g to m/s^2.

    Returns:
        Dictionary containing response arrays and metadata.

    中文说明：
        针对一个损伤工况和一条地震动，执行一次结构时程分析。

        参数：
            damage：
                层级刚度退化向量。
            ground_accel_g：
                单位为 g 的地面加速度数组。
            dt：
                地震动时间步长。
            config：
                框架结构配置。
            damping_ratio：
                目标 Rayleigh 阻尼比。
            g：
                重力加速度，用于将 g 单位转换为 m/s^2。

        返回：
            一个字典，包含响应数组和元数据。
    """

    if config is None:
        # Use default frame if no config is provided.
        # 中文：如果未提供结构配置，则使用默认框架配置。
        config = FrameConfig()

    model_info = build_2d_frame(config=config, damage=damage)
    # Build the OpenSees model for this damage case.
    # 中文：针对当前损伤工况建立 OpenSees 模型。

    node_tags = model_info["node_tags"]
    # Extract node tag mapping, so we can record floor responses.
    # 中文：提取节点编号映射，后续才能根据楼层和跨号记录楼层响应。

    modal_periods = compute_modal_periods(config=config, damage=damage, n_modes=4)
    # Compute modal periods for sanity checking.
    # Note: compute_modal_periods rebuilds the model, so we rebuild again below before transient analysis.
    # 中文：计算模态周期，用于基本合理性检查。
    # 中文：注意：`compute_modal_periods` 会重新建立模型，因此在真正进行瞬态分析前，下面需要再次重建模型，避免分析状态被改变。

    model_info = build_2d_frame(config=config, damage=damage)
    # Rebuild the model after modal analysis.
    # This avoids any analysis state contamination.
    # 中文：在模态分析后重新建立模型，避免模态分析留下的状态污染后续瞬态分析。

    node_tags = model_info["node_tags"]
    # Re-extract node tags from the rebuilt model.
    # 中文：从重新建立的模型中再次提取节点编号映射。

    damping_info = add_rayleigh_damping(damping_ratio=damping_ratio)
    # Add Rayleigh damping to the current model.
    # 中文：为当前模型添加 Rayleigh 阻尼。

    ground_accel_mps2 = ground_accel_g * g
    # Convert ground acceleration from g to m/s^2.
    # 中文：将地面加速度从 g 单位转换为 m/s^2。

    ops.timeSeries(
        "Path",
        1,
        "-dt",
        dt,
        "-values",
        *ground_accel_mps2.tolist(),
    )
    # `timeSeries("Path", tag, "-dt", dt, "-values", values...)` defines a time-dependent input series.
    # `"Path"` means the input is a sequence of values over time.
    # `1` is the time-series tag.
    # `"-dt", dt` defines the time interval between consecutive values.
    # `"-values"` means the following numbers are the time-series values.
    # `*ground_accel_mps2.tolist()` expands the Python list into separate OpenSees arguments.
    # 中文：`timeSeries("Path", tag, "-dt", dt, "-values", values...)` 定义随时间变化的输入序列。
    # 中文：`"Path"` 表示输入是一串按时间排列的数值；`1` 是时间序列编号；`"-dt", dt` 定义相邻输入值之间的时间间隔；`"-values"` 后面跟随具体输入值；`*ground_accel_mps2.tolist()` 会把 Python 列表展开为 OpenSees 可接收的多个参数。

    ops.pattern("UniformExcitation", 1, 1, "-accel", 1)
    # `pattern("UniformExcitation", patternTag, dir, "-accel", tsTag)` applies uniform base excitation.
    # `1` after "UniformExcitation" is the pattern tag.
    # The next `1` is the excitation direction: DOF 1 = horizontal x direction.
    # `"-accel", 1` means acceleration time series with tag 1 is used.
    # 中文：`pattern("UniformExcitation", patternTag, dir, "-accel", tsTag)` 用于施加一致支座激励。
    # 中文：`"UniformExcitation"` 后第一个 `1` 是荷载模式编号；第二个 `1` 表示激励方向为 DOF 1，即水平 x 方向；`"-accel", 1` 表示使用编号为 1 的加速度时间序列。

    setup_transient_analysis()
    # Configure solver, convergence test, integrator, and analysis type.
    # 中文：配置求解器、收敛判据、时间积分方法和分析类型。

    n_steps = len(ground_accel_mps2)
    # Number of time steps in the input motion.
    # 中文：输入地震动的时间步总数。

    n_story = config.n_story
    # Number of stories, used to define output array width.
    # 中文：结构层数，用于确定输出数组的列数。

    rel_disp = np.zeros((n_steps, n_story), dtype=np.float64)
    # Relative floor horizontal displacement array.
    # Shape: [time_step, story_index].
    # 中文：楼层相对水平位移数组；数组形状为 `[时间步, 楼层索引]`。

    rel_accel = np.zeros((n_steps, n_story), dtype=np.float64)
    # Relative floor horizontal acceleration array from OpenSees.
    # 中文：OpenSees 输出的楼层相对水平加速度数组。

    abs_accel = np.zeros((n_steps, n_story), dtype=np.float64)
    # Approximate absolute floor acceleration array.
    # For uniform support excitation:
    # absolute acceleration ≈ relative acceleration + ground acceleration.
    # 中文：近似楼层绝对加速度数组。
    # 中文：对于一致支座激励，绝对加速度约等于相对加速度加上地面输入加速度。

    story_drift = np.zeros((n_steps, n_story), dtype=np.float64)
    # Story drift ratio array.
    # story_drift[:, i] = interstory displacement / story height.
    # 中文：层间位移角数组；`story_drift[:, i]` 等于第 i 层的层间位移除以层高。

    mid_bay = config.n_bay // 2
    # Use the middle column line as the representative floor response location.
    # For a 2-bay frame, mid_bay = 1.
    # 中文：选择中间柱线作为代表性楼层响应位置；对于 2 跨框架，`mid_bay = 1`。

    analysis_failed_at = -1
    # If analysis succeeds, keep -1.
    # If it fails, store the failed step index.
    # 中文：如果分析成功，则保持为 `-1`；如果分析失败，则记录失败发生的时间步索引。

    for step in range(n_steps):
        # Loop through each time step.
        # 中文：逐个时间步进行动力分析。

        ok = ops.analyze(1, dt)
        # `ops.analyze(1, dt)` advances the transient analysis by one time step of size dt.
        # `ok = 0` means success; nonzero means failure.
        # 中文：`ops.analyze(1, dt)` 表示让瞬态分析向前推进 1 个时间步，时间步长为 `dt`；`ok = 0` 表示成功，非零表示失败。

        if ok != 0:
            # Stop if numerical analysis fails.
            # 中文：如果数值分析失败，则停止循环，避免继续记录无效结果。
            analysis_failed_at = step
            break

        previous_floor_disp = 0.0
        # Base displacement is zero in the relative coordinate system.
        # 中文：在相对坐标体系中，基础位移视为 0。

        for story in range(1, n_story + 1):
            # Loop over stories 1 to n_story.
            # 中文：从第 1 层到第 `n_story` 层逐层提取响应。

            node = node_tags[(story, mid_bay)]
            # Select the middle node at this floor.
            # 中文：选择当前楼层中间柱线处的节点作为代表节点。

            ux = ops.nodeDisp(node, 1)
            # `ops.nodeDisp(node, 1)` returns horizontal displacement at DOF 1.
            # 中文：`ops.nodeDisp(node, 1)` 返回该节点 DOF 1 方向的水平位移。

            ax_rel = ops.nodeAccel(node, 1)
            # `ops.nodeAccel(node, 1)` returns relative horizontal acceleration at DOF 1.
            # 中文：`ops.nodeAccel(node, 1)` 返回该节点 DOF 1 方向的相对水平加速度。

            rel_disp[step, story - 1] = ux
            # Store relative displacement for this story.
            # 中文：将当前楼层的相对位移存入数组。

            rel_accel[step, story - 1] = ax_rel
            # Store relative acceleration for this story.
            # 中文：将当前楼层的相对加速度存入数组。

            abs_accel[step, story - 1] = ax_rel + ground_accel_mps2[step]
            # Approximate absolute acceleration by adding ground acceleration.
            # 中文：通过“相对加速度 + 当前时间步地面加速度”近似得到楼层绝对加速度。

            interstory_disp = ux - previous_floor_disp
            # Interstory displacement = current floor displacement - lower floor displacement.
            # 中文：层间位移等于当前楼层位移减去下层楼层位移。

            story_drift[step, story - 1] = interstory_disp / config.story_height
            # Story drift ratio = interstory displacement / story height.
            # 中文：层间位移角等于层间位移除以层高。

            previous_floor_disp = ux
            # Update lower-floor displacement for the next story.
            # 中文：更新“下层楼层位移”，供下一层计算层间位移时使用。

    if analysis_failed_at >= 0:
        # If the analysis failed, truncate arrays to successful steps only.
        # 中文：如果分析中途失败，则只保留失败前已经成功完成的时间步数据。
        rel_disp = rel_disp[:analysis_failed_at]
        rel_accel = rel_accel[:analysis_failed_at]
        abs_accel = abs_accel[:analysis_failed_at]
        story_drift = story_drift[:analysis_failed_at]
        ground_accel_mps2 = ground_accel_mps2[:analysis_failed_at]

    return {
        "damage": np.array(damage, dtype=np.float64),
        "ground_accel_mps2": ground_accel_mps2,
        "rel_disp": rel_disp,
        "rel_accel": rel_accel,
        "abs_accel": abs_accel,
        "story_drift": story_drift,
        "dt": np.array(dt, dtype=np.float64),
        "modal_periods": np.array(modal_periods, dtype=np.float64),
        "damping_alpha_m": np.array(damping_info["alpha_m"], dtype=np.float64),
        "damping_beta_k_init": np.array(damping_info["beta_k_init"], dtype=np.float64),
        "analysis_failed_at": np.array(analysis_failed_at, dtype=np.int64),
    }


def save_response_npz(response: Dict[str, np.ndarray], output_path: Path) -> None:
    """
    Save one response sample as a compressed .npz file.

    .npz is convenient because it stores multiple NumPy arrays in one file.

    中文说明：
        将一个响应样本保存为压缩的 `.npz` 文件。

        `.npz` 很适合本项目，因为它可以在一个文件中同时存储多个 NumPy 数组。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Create the parent folder if it does not exist.
    # 中文：如果输出文件所在的父文件夹不存在，则自动创建。

    np.savez_compressed(output_path, **response)
    # `np.savez_compressed(path, **response)` saves dictionary values into a compressed .npz file.
    # Each dictionary key becomes one array name inside the file.
    # 中文：`np.savez_compressed(path, **response)` 会把字典中的各个数组保存到压缩 `.npz` 文件中；字典的每个键会成为文件内部的数组名称。


def plot_sanity_check(response: Dict[str, np.ndarray], output_dir: Path) -> None:
    """
    Plot response curves to verify whether the model output is reasonable.

    中文说明：
        绘制响应曲线，用于快速判断模型输出是否合理。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    # Create figure output folder if needed.
    # 中文：如果图片输出文件夹不存在，则自动创建。

    dt = float(response["dt"])
    # Convert NumPy scalar dt to Python float.
    # 中文：将 NumPy 标量形式的 `dt` 转换为普通 Python 浮点数，方便后续计算。

    n_steps = response["abs_accel"].shape[0]
    # Number of successful analysis steps.
    # 中文：成功完成分析的时间步数量。

    t = np.arange(n_steps) * dt
    # Recreate time vector.
    # 中文：根据时间步数量和时间步长重新生成时间向量。

    roof_index = response["abs_accel"].shape[1] - 1
    # Last column corresponds to roof floor.
    # 中文：数组最后一列对应结构顶层，也就是屋面楼层。

    plt.figure(figsize=(9, 4))
    # Create a new figure.
    # 中文：创建一张新的绘图窗口，尺寸为 9×4 英寸。

    plt.plot(t, response["ground_accel_mps2"], label="Ground acceleration")
    # Plot ground acceleration.
    # 中文：绘制输入地面加速度时程曲线。

    plt.xlabel("Time [s]")
    # X-axis label.
    # 中文：设置 x 轴标签为时间，单位是秒。

    plt.ylabel("Acceleration [m/s²]")
    # Y-axis label.
    # 中文：设置 y 轴标签为加速度，单位是 m/s²。

    plt.title("Input ground acceleration")
    # Figure title.
    # 中文：设置图题为输入地面加速度。

    plt.legend()
    # Show legend.
    # 中文：显示图例。

    plt.tight_layout()
    # Adjust layout to prevent label clipping.
    # 中文：自动调整图像布局，避免坐标轴标签或标题被裁剪。

    plt.savefig(output_dir / "ground_acceleration.png", dpi=200)
    # Save the figure as a PNG file.
    # 中文：将图像保存为 PNG 文件，分辨率为 200 dpi。

    plt.close()
    # Close figure to release memory.
    # 中文：关闭当前图像，释放内存，避免后续绘图互相干扰。

    plt.figure(figsize=(9, 4))
    # Create another figure.
    # 中文：创建另一张新的绘图窗口。

    plt.plot(t, response["abs_accel"][:, roof_index], label="Roof absolute acceleration")
    # Plot roof absolute acceleration.
    # 中文：绘制屋面楼层的绝对加速度时程曲线。

    plt.xlabel("Time [s]")
    # X-axis label.
    # 中文：设置 x 轴标签为时间，单位是秒。

    plt.ylabel("Acceleration [m/s²]")
    # Y-axis label.
    # 中文：设置 y 轴标签为加速度，单位是 m/s²。

    plt.title("Roof absolute acceleration")
    # Figure title.
    # 中文：设置图题为屋面绝对加速度。

    plt.legend()
    # Show legend.
    # 中文：显示图例。

    plt.tight_layout()
    # Adjust layout.
    # 中文：自动调整图像布局。

    plt.savefig(output_dir / "roof_absolute_acceleration.png", dpi=200)
    # Save the figure.
    # 中文：保存屋面绝对加速度图像。

    plt.close()
    # Close the figure.
    # 中文：关闭当前图像窗口，释放绘图资源。


def demo_one_case() -> None:
    """
    Run one complete demo:
        1. create artificial ground motion,
        2. run damaged-frame time-history analysis,
        3. save .npz response,
        4. save sanity-check figures.

    中文说明：
        运行一个完整的示例流程：
            1. 创建人工地震动；
            2. 对损伤框架进行时程分析；
            3. 保存 `.npz` 响应数据；
            4. 保存用于合理性检查的图像。
    """

    config = FrameConfig()
    # Use the default 4-story 2-bay frame.
    # 中文：使用默认的 4 层 2 跨框架配置。

    damage = [0.0, 0.15, 0.0, 0.25]
    # Example damage vector.
    # This is the label for this simulated sample.
    # 中文：示例损伤向量；它就是该仿真样本对应的标签。

    motion = create_sine_ground_motion(
        duration=20.0,
        dt=0.01,
        amplitude_g=0.10,
        frequency_hz=1.0,
    )
    # Create artificial sine ground acceleration for debugging.
    # 中文：创建用于调试的人工正弦地面加速度。

    response = run_time_history(
        damage=damage,
        ground_accel_g=motion["accel_g"],
        dt=0.01,
        config=config,
        damping_ratio=0.02,
    )
    # Run transient dynamic analysis.
    # 中文：执行瞬态动力时程分析。

    output_path = Path("data_raw/opensees_outputs/demo_damage_case.npz")
    # Define where to save the response data.
    # 中文：定义响应数据的保存位置。

    save_response_npz(response, output_path)
    # Save response to disk.
    # 中文：将响应数据保存到磁盘。

    plot_sanity_check(response, Path("results/figures"))
    # Save response plots.
    # 中文：保存响应曲线图，用于结果检查。

    print("Saved response to:", output_path)
    # Print output data path.
    # 中文：打印输出数据文件路径。

    print("Damage label:", response["damage"])
    # Print damage vector.
    # 中文：打印损伤标签向量。

    print("Modal periods [s]:", response["modal_periods"])
    # Print modal periods.
    # 中文：打印模态周期。

    print("Absolute acceleration shape:", response["abs_accel"].shape)
    # Expected shape: [number_of_time_steps, number_of_stories].
    # For 20 seconds and dt=0.01, expected time steps ≈ 2000, stories = 4.
    # 中文：打印绝对加速度数组的形状；预期格式为 `[时间步数量, 结构层数]`。
    # 中文：对于 20 秒、`dt = 0.01` 的输入，理论时间步约为 2000，结构层数为 4。

    print("Analysis failed at step:", int(response["analysis_failed_at"]))
    # -1 means no failure.
    # 中文：打印分析失败发生的时间步；`-1` 表示没有失败。


if __name__ == "__main__":
    # Execute demo only when this file is run as a module/script.
    # 中文：只有当本文件被直接作为模块或脚本运行时，才执行下面的示例分析。

    demo_one_case()
    # Run one dynamic analysis sample.
    # 中文：运行一个动力分析样本。

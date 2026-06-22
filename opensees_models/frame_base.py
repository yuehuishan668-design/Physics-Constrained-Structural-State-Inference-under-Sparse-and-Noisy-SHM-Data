"""
File location:
    opensees_models/frame_base.py

Purpose:
    Build a 2D multi-story, multi-bay elastic frame in OpenSeesPy.
    This is the first finite element model for generating structural response data.

Research role:
    This file creates the controllable structure used later for:
    1. healthy response generation,
    2. stiffness-degradation damage simulation,
    3. sparse/noisy monitoring data generation,
    4. physics-constrained state inference experiments.

Important modeling decision:
    Damage is represented by story-level stiffness degradation:
        E_i = (1 - d_i) * E
    where d_i is the damage ratio of story i.
    Example:
        d_i = 0.00 means no stiffness degradation.
        d_i = 0.20 means 20% stiffness degradation.

中文说明：
    文件位置：
        opensees_models/frame_base.py

    文件目的：
        在 OpenSeesPy 中建立一个二维、多层、多跨的弹性框架结构模型。
        这是后续生成结构响应数据的第一个有限元基础模型。

    研究作用：
        本文件创建一个可控的结构模型，后续可以用于：
        1. 生成健康结构响应；
        2. 模拟刚度退化形式的损伤；
        3. 生成稀疏和含噪的监测数据；
        4. 开展物理约束状态识别实验。

    重要建模决定：
        损伤采用“层级刚度退化”表示：
            E_i = (1 - d_i) * E
        其中 d_i 表示第 i 层的损伤比例。
        例如：
            d_i = 0.00 表示没有刚度退化；
            d_i = 0.20 表示该层刚度退化 20%。
"""

from __future__ import annotations
# `from __future__ import annotations` postpones type-hint evaluation.
# It makes type hints more flexible and avoids some circular-reference problems.
# 中文：这句会延后类型注解的求值，使类型提示更灵活，并减少循环引用导致的问题。

import math
# `math` provides mathematical constants and functions.
# Here it is used for pi and square-root-related calculations.
# 中文：`math` 是 Python 的数学模块，这里主要用于圆周率 `pi` 和平方根等计算。

from dataclasses import dataclass
# `dataclass` is used to define a clean configuration object.
# It automatically creates an __init__ method for the class.
# 中文：`dataclass` 用于简洁地定义配置类，会自动生成类的初始化方法 `__init__`。

from typing import Dict, List, Tuple
# `Dict`, `List`, and `Tuple` are type hints.
# They do not change runtime behavior but make the code easier to read and check.
# 中文：`Dict`、`List`、`Tuple` 是类型提示，不改变运行结果，但能让代码结构更清晰，也方便编辑器或类型检查工具发现问题。

import openseespy.opensees as ops
# `openseespy.opensees` is the OpenSeesPy command module.
# `as ops` gives it a shorter alias, so we can write `ops.node(...)` instead of the full module name.
# 中文：`openseespy.opensees` 是 OpenSeesPy 的核心命令模块；`as ops` 是给它起一个短别名，后续可以写 `ops.node(...)`，不用每次写完整模块名。


@dataclass
class FrameConfig:
    """
    Configuration class for the 2D frame.

    Each attribute controls one part of the finite element model.
    Units:
        length: m
        mass: kg
        force: N
        stress/modulus: Pa = N/m^2

    中文说明：
        二维框架结构的配置类。

        每一个属性都控制有限元模型中的一部分参数。
        单位：
            长度：m
            质量：kg
            力：N
            应力/弹性模量：Pa = N/m^2
    """

    n_story: int = 4
    # `n_story` is the number of stories/floors above the base.
    # 中文：`n_story` 表示基础以上的结构层数，也就是楼层数量。

    n_bay: int = 2
    # `n_bay` is the number of horizontal spans.
    # 中文：`n_bay` 表示水平方向的跨数。

    story_height: float = 3.2
    # `story_height` is the vertical distance between two adjacent floors, in meters.
    # 中文：`story_height` 表示相邻两层之间的竖向距离，单位是米。

    bay_width: float = 6.0
    # `bay_width` is the horizontal span length, in meters.
    # 中文：`bay_width` 表示每一跨的水平长度，单位是米。

    E: float = 2.0e11
    # `E` is Young's modulus, in Pa.
    # 2.0e11 Pa is a typical order of magnitude for steel.
    # 中文：`E` 是材料的杨氏模量，单位是 Pa；`2.0e11 Pa` 是钢材弹性模量的常见数量级。

    A_col: float = 0.09
    # `A_col` is the cross-sectional area of columns, in m^2.
    # 中文：`A_col` 是柱截面面积，单位是 m^2。

    I_col: float = 6.75e-4
    # `I_col` is the second moment of area of columns, in m^4.
    # In a 2D frame, this controls bending stiffness E*I.
    # 中文：`I_col` 是柱截面的惯性矩，单位是 m^4；在二维框架中，它与 `E` 相乘形成弯曲刚度 `E*I`。

    A_beam: float = 0.075
    # `A_beam` is the cross-sectional area of beams, in m^2.
    # 中文：`A_beam` 是梁截面面积，单位是 m^2。

    I_beam: float = 4.80e-4
    # `I_beam` is the second moment of area of beams, in m^4.
    # 中文：`I_beam` 是梁截面的惯性矩，单位是 m^4，用于控制梁的弯曲刚度。

    floor_mass: float = 2.0e5
    # `floor_mass` is the total lumped mass assigned to each floor, in kg.
    # 中文：`floor_mass` 表示每一层楼面分配的总集中质量，单位是 kg。

    tiny_mass_ratio_y: float = 1.0e-6
    # `tiny_mass_ratio_y` gives a very small vertical mass to avoid numerical singularity in modal analysis.
    # It has negligible influence on lateral response.
    # 中文：`tiny_mass_ratio_y` 用于给竖向自由度分配一个极小质量，避免模态分析中的数值奇异；它对水平地震响应的影响可以忽略。

    tiny_mass_ratio_rot: float = 1.0e-9
    # `tiny_mass_ratio_rot` gives a very small rotational mass to avoid numerical singularity.
    # It is not meant to represent a calibrated physical rotational inertia.
    # 中文：`tiny_mass_ratio_rot` 用于给转动自由度分配一个极小质量以提升数值稳定性；它不是经过物理标定的真实转动惯量。


def validate_damage(damage: List[float], n_story: int) -> List[float]:
    """
    Validate the story-level damage vector.

    Parameters:
        damage:
            List of stiffness degradation ratios.
            Its length must equal the number of stories.
        n_story:
            Number of stories in the structure.

    Returns:
        The validated damage list.

    Raises:
        ValueError if damage is invalid.

    中文说明：
        检查层级损伤向量是否合法。

        参数：
            damage：
                刚度退化比例列表，长度必须等于结构层数。
            n_story：
                结构总层数。

        返回：
            通过检查后的损伤列表。

        异常：
            如果损伤向量长度或数值范围不合理，则抛出 ValueError。
    """

    if len(damage) != n_story:
        # `len(damage)` counts how many entries are in the damage vector.
        # It must match `n_story`; otherwise, one story has no damage value or too many values exist.
        # 中文：`len(damage)` 统计损伤向量中有多少个数值；它必须与 `n_story` 相等，否则会出现某层缺少损伤值或损伤值数量过多的问题。
        raise ValueError(f"damage must have length {n_story}, but got {len(damage)}.")

    for idx, value in enumerate(damage):
        # `enumerate(damage)` gives both index and value.
        # `idx` starts at 0, so story number is idx + 1.
        # 中文：`enumerate(damage)` 会同时给出索引和值；由于 Python 索引从 0 开始，所以实际楼层编号是 `idx + 1`。
        if value < 0.0 or value >= 0.95:
            # Damage must be within a physically reasonable range.
            # We use value < 0.95 instead of value <= 1.0 because near-zero stiffness can cause instability.
            # 中文：损伤值必须处于物理上合理的范围内；这里要求小于 0.95 而不是小于等于 1.0，是因为接近零刚度会导致结构模型数值不稳定。
            raise ValueError(
                f"damage[{idx}]={value} is invalid. Use 0.0 <= damage < 0.95."
            )

    return damage


def make_node_tag(floor: int, bay: int, n_bay: int) -> int:
    """
    Convert a floor-bay coordinate into a unique OpenSees node tag.

    OpenSees identifies nodes by integer tags.
    This helper creates deterministic tags.

    Example for n_bay = 2:
        floor 0: nodes 1, 2, 3
        floor 1: nodes 4, 5, 6
        floor 2: nodes 7, 8, 9

    中文说明：
        将“楼层-跨号”坐标转换为唯一的 OpenSees 节点编号。

        OpenSees 使用整数编号识别节点。
        这个辅助函数用于生成稳定、可重复的节点编号。

        当 n_bay = 2 时：
            第 0 层：节点 1、2、3
            第 1 层：节点 4、5、6
            第 2 层：节点 7、8、9
    """

    return floor * (n_bay + 1) + bay + 1
    # `n_bay + 1` is the number of column lines/nodes per floor.
    # `+ 1` makes the tag start from 1, because OpenSees tags are conventionally positive integers.
    # 中文：`n_bay + 1` 是每层的柱线数量，也就是每层节点数；最后的 `+ 1` 让节点编号从 1 开始，因为 OpenSees 中的编号通常使用正整数。


def build_2d_frame(
    config: FrameConfig | None = None,
    damage: List[float] | None = None,
) -> Dict[str, object]:
    """
    Build the 2D frame model in OpenSeesPy.

    Parameters:
        config:
            FrameConfig object containing geometry, section, material, and mass parameters.
        damage:
            Story-level stiffness degradation vector.
            Example:
                [0.0, 0.1, 0.0, 0.2]
            means:
                story 1: healthy,
                story 2: 10% stiffness degradation,
                story 3: healthy,
                story 4: 20% stiffness degradation.

    Returns:
        A dictionary containing node tags and element tags for later response extraction.

    中文说明：
        在 OpenSeesPy 中建立二维框架模型。

        参数：
            config：
                FrameConfig 配置对象，包含几何尺寸、截面、材料和质量参数。
            damage：
                层级刚度退化向量。
                例如：
                    [0.0, 0.1, 0.0, 0.2]
                表示：
                    第 1 层：健康；
                    第 2 层：刚度退化 10%；
                    第 3 层：健康；
                    第 4 层：刚度退化 20%。

        返回：
            一个字典，包含节点编号和单元编号，方便后续提取结构响应。
    """

    if config is None:
        # If the user does not pass a config, use the default 4-story 2-bay frame.
        # 中文：如果调用函数时没有传入 `config`，就使用默认的 4 层 2 跨框架配置。
        config = FrameConfig()

    if damage is None:
        # If no damage vector is given, build a healthy structure.
        # 中文：如果没有提供损伤向量，就默认建立一个完全健康的结构。
        damage = [0.0] * config.n_story

    damage = validate_damage(damage, config.n_story)
    # Validate damage length and value range before building the model.
    # 中文：在建立模型之前，先检查损伤向量长度和数值范围是否合法。

    ops.wipe()
    # `ops.wipe()` clears the existing OpenSees model from memory.
    # This is essential before building a new model, otherwise old nodes/elements may remain.
    # 中文：`ops.wipe()` 会清空内存中已有的 OpenSees 模型；建立新模型前必须执行，否则旧节点或旧单元可能残留并影响结果。

    ops.model("basic", "-ndm", 2, "-ndf", 3)
    # `ops.model` starts a new OpenSees model.
    # `"basic"` means the standard OpenSees basic model builder.
    # `"-ndm", 2` means number of spatial dimensions = 2.
    # `"-ndf", 3` means number of degrees of freedom per node = 3.
    # For a 2D frame, the three nodal DOFs are:
    #     1 = horizontal translation ux,
    #     2 = vertical translation uy,
    #     3 = in-plane rotation rz.
    # 中文：`ops.model` 用于启动一个新的 OpenSees 模型；`"basic"` 表示使用标准基础模型构建器；`"-ndm", 2` 表示二维空间模型；`"-ndf", 3` 表示每个节点有 3 个自由度。
    # 中文：对于二维框架，这 3 个节点自由度分别是：1 = 水平平动 ux，2 = 竖向平动 uy，3 = 平面内转动 rz。

    node_tags: Dict[Tuple[int, int], int] = {}
    # `node_tags` maps (floor, bay) coordinates to OpenSees node tags.
    # Example: node_tags[(2, 1)] gives the middle node on floor 2.
    # 中文：`node_tags` 用于把 `(楼层, 跨号)` 坐标映射到 OpenSees 节点编号；例如 `node_tags[(2, 1)]` 表示第 2 层中间柱线处的节点编号。

    for floor in range(config.n_story + 1):
        # `range(config.n_story + 1)` includes floor 0, which is the base level.
        # For 4 stories, floors are 0, 1, 2, 3, 4.
        # 中文：`range(config.n_story + 1)` 包含第 0 层，也就是基础层；如果结构为 4 层，则循环的楼层编号为 0、1、2、3、4。

        y = floor * config.story_height
        # `y` is the vertical coordinate of the current floor.
        # 中文：`y` 是当前楼层的竖向坐标。

        for bay in range(config.n_bay + 1):
            # For 2 bays, there are 3 vertical column lines: bay index 0, 1, 2.
            # 中文：对于 2 跨框架，会有 3 条竖向柱线，对应跨号索引 0、1、2。

            x = bay * config.bay_width
            # `x` is the horizontal coordinate of the current column line.
            # 中文：`x` 是当前柱线的水平坐标。

            tag = make_node_tag(floor, bay, config.n_bay)
            # `tag` is the unique integer node ID used by OpenSees.
            # 中文：`tag` 是 OpenSees 用来识别该节点的唯一整数编号。

            ops.node(tag, x, y)
            # `ops.node(tag, x, y)` creates a node at coordinate (x, y).
            # `tag` identifies this node in later commands.
            # 中文：`ops.node(tag, x, y)` 在坐标 `(x, y)` 处创建一个节点；后续命令会通过 `tag` 调用这个节点。

            node_tags[(floor, bay)] = tag
            # Store the tag so later code can create elements and extract responses.
            # 中文：把节点编号存入字典，方便后续创建单元和提取结构响应。

    for bay in range(config.n_bay + 1):
        # Loop over all base nodes.
        # 中文：循环遍历基础层上的所有节点。

        base_node = node_tags[(0, bay)]
        # `floor = 0` means the base level.
        # 中文：`floor = 0` 表示基础层。

        ops.fix(base_node, 1, 1, 1)
        # `ops.fix(nodeTag, 1, 1, 1)` fixes all 3 DOFs of the base node.
        # 1 means constrained/fixed; 0 would mean free.
        # Here:
        #     ux fixed,
        #     uy fixed,
        #     rz fixed.
        # 中文：`ops.fix(nodeTag, 1, 1, 1)` 表示约束基础节点的 3 个自由度；`1` 表示固定，`0` 表示自由。
        # 中文：这里固定了水平位移 ux、竖向位移 uy 和转角 rz，因此基础节点为完全固结支座。

    mass_per_node = config.floor_mass / (config.n_bay + 1)
    # The total floor mass is evenly distributed to all nodes on the same floor.
    # 中文：将每层总质量平均分配到该层的所有节点上。

    for floor in range(1, config.n_story + 1):
        # Only above-ground floors receive mass.
        # The base floor is fixed and does not need dynamic mass.
        # 中文：只给基础以上的楼层分配质量；基础层已经固定，不需要参与动力响应质量分配。

        for bay in range(config.n_bay + 1):
            # Loop over all nodes on this floor.
            # 中文：遍历当前楼层上的所有节点。

            node = node_tags[(floor, bay)]
            # Get OpenSees node tag.
            # 中文：获取当前节点对应的 OpenSees 节点编号。

            mx = mass_per_node
            # `mx` is the horizontal translational mass.
            # It dominates lateral seismic response.
            # 中文：`mx` 是水平方向平动质量，它对结构水平地震响应起主要作用。

            my = mass_per_node * config.tiny_mass_ratio_y
            # `my` is a tiny vertical mass for numerical stability.
            # It is intentionally much smaller than `mx`.
            # 中文：`my` 是一个极小的竖向质量，主要用于提高数值稳定性；它被故意设置得远小于 `mx`。

            mr = mass_per_node * config.tiny_mass_ratio_rot
            # `mr` is a tiny rotational mass for numerical stability.
            # 中文：`mr` 是一个极小的转动质量，同样用于避免数值奇异或不稳定。

            ops.mass(node, mx, my, mr)
            # `ops.mass(nodeTag, m1, m2, m3)` assigns nodal mass to the 3 DOFs.
            # DOF 1: horizontal translation, DOF 2: vertical translation, DOF 3: rotation.
            # 中文：`ops.mass(nodeTag, m1, m2, m3)` 为节点的 3 个自由度分配质量；DOF 1 是水平平动，DOF 2 是竖向平动，DOF 3 是转动。

    transf_tag = 1
    # `transf_tag` is the ID of the geometric transformation object.
    # 中文：`transf_tag` 是几何变换对象的编号，后续单元需要通过这个编号调用对应的几何变换。

    ops.geomTransf("Linear", transf_tag)
    # `ops.geomTransf("Linear", transfTag)` defines a linear coordinate transformation.
    # It maps local element coordinates to global coordinates.
    # `"Linear"` assumes small displacement geometry, suitable for the first elastic model.
    # 中文：`ops.geomTransf("Linear", transfTag)` 定义线性坐标变换，用于把单元局部坐标转换到整体坐标；`"Linear"` 假设小变形几何，适合当前第一版弹性框架模型。

    element_tags: Dict[str, List[int]] = {
        "columns": [],
        "beams": [],
    }
    # `element_tags` stores element IDs separately for columns and beams.
    # 中文：`element_tags` 分别存储柱单元和梁单元的编号，方便后续分类调用。

    ele_tag = 1
    # `ele_tag` is the unique integer tag for each OpenSees element.
    # 中文：`ele_tag` 是 OpenSees 中每个单元的唯一整数编号。

    for story in range(1, config.n_story + 1):
        # Story 1 connects floor 0 to floor 1.
        # Story 2 connects floor 1 to floor 2, etc.
        # 中文：第 1 层连接第 0 层和第 1 层；第 2 层连接第 1 层和第 2 层，以此类推。

        d_story = damage[story - 1]
        # Python list index starts at 0.
        # Therefore story 1 uses damage[0].
        # 中文：Python 列表索引从 0 开始，因此第 1 层对应 `damage[0]`。

        E_story = (1.0 - d_story) * config.E
        # Story stiffness degradation is implemented by reducing Young's modulus E.
        # This is equivalent to reducing flexural stiffness E*I for all columns in that story.
        # 中文：层级刚度退化通过降低该层柱子的杨氏模量 `E` 实现；这等价于降低该层柱子的弯曲刚度 `E*I`。

        for bay in range(config.n_bay + 1):
            # Create one column element on each column line for this story.
            # 中文：在当前层的每一条柱线上创建一个柱单元。

            node_i = node_tags[(story - 1, bay)]
            # Bottom node of the column.
            # 中文：柱单元的底部节点。

            node_j = node_tags[(story, bay)]
            # Top node of the column.
            # 中文：柱单元的顶部节点。

            ops.element(
                "elasticBeamColumn",
                ele_tag,
                node_i,
                node_j,
                config.A_col,
                E_story,
                config.I_col,
                transf_tag,
            )
            # `ops.element` creates a finite element.
            # `"elasticBeamColumn"` means a linear elastic frame element.
            # `ele_tag` is the element ID.
            # `node_i`, `node_j` are the two end nodes.
            # `config.A_col` is column area A.
            # `E_story` is degraded Young's modulus E for the current story.
            # `config.I_col` is column second moment of area I.
            # `transf_tag` links the element to the geometric transformation.
            # 中文：`ops.element` 用于创建有限元单元；`"elasticBeamColumn"` 表示线弹性梁柱单元。
            # 中文：`ele_tag` 是单元编号；`node_i` 和 `node_j` 是单元两端节点；`config.A_col` 是柱截面面积；`E_story` 是当前层退化后的杨氏模量；`config.I_col` 是柱截面惯性矩；`transf_tag` 将该单元与几何变换对象关联起来。

            element_tags["columns"].append(ele_tag)
            # Save the column element tag.
            # 中文：保存当前柱单元编号。

            ele_tag += 1
            # Increment element tag to keep each element ID unique.
            # 中文：单元编号加 1，确保每个单元都有唯一编号。

    for floor in range(1, config.n_story + 1):
        # Beam elements are placed at each floor level above the base.
        # 中文：梁单元布置在基础以上的每一个楼层平面上。

        for bay in range(config.n_bay):
            # For n_bay = 2, beams connect:
            # bay 0 -> bay 1, and bay 1 -> bay 2.
            # 中文：当 `n_bay = 2` 时，梁单元分别连接 bay 0 到 bay 1，以及 bay 1 到 bay 2。

            node_i = node_tags[(floor, bay)]
            # Left node of the beam.
            # 中文：梁单元左端节点。

            node_j = node_tags[(floor, bay + 1)]
            # Right node of the beam.
            # 中文：梁单元右端节点。

            ops.element(
                "elasticBeamColumn",
                ele_tag,
                node_i,
                node_j,
                config.A_beam,
                config.E,
                config.I_beam,
                transf_tag,
            )
            # Beam stiffness is not degraded in the first model.
            # Only story-level column stiffness degradation is used as the damage label.
            # 中文：第一版模型中梁刚度不进行退化；损伤标签只通过层级柱刚度退化体现。

            element_tags["beams"].append(ele_tag)
            # Save the beam element tag.
            # 中文：保存当前梁单元编号。

            ele_tag += 1
            # Increment element tag.
            # 中文：单元编号加 1，为下一个单元准备新的唯一编号。

    return {
        "config": config,
        "damage": damage,
        "node_tags": node_tags,
        "element_tags": element_tags,
    }
    # Return model metadata.
    # This does not return the OpenSees model itself, because OpenSees stores it internally.
    # 中文：返回模型的元数据，包括配置、损伤向量、节点编号和单元编号。
    # 中文：这里不直接返回 OpenSees 模型对象，因为 OpenSees 会在其内部状态中保存当前模型。



def compute_modal_periods(
    config: FrameConfig | None = None,
    damage: List[float] | None = None,
    n_modes: int = 4,
) -> List[float]:
    """
    Build a frame and compute its first modal periods.

    Modal period:
        T = 2*pi / omega
    where:
        lambda = omega^2
        lambda is the eigenvalue returned by OpenSees.

    中文说明：
        建立一个框架模型，并计算其前若干阶模态周期。

        模态周期：
            T = 2*pi / omega
        其中：
            lambda = omega^2
            lambda 是 OpenSees 返回的特征值。
    """

    build_2d_frame(config=config, damage=damage)
    # Build the OpenSees finite element model.
    # 中文：建立 OpenSees 有限元模型。

    eigen_values = ops.eigen(n_modes)
    # `ops.eigen(n_modes)` solves for the first `n_modes` eigenvalues.
    # In structural dynamics, eigenvalue lambda = omega^2.
    # 中文：`ops.eigen(n_modes)` 求解前 `n_modes` 阶特征值；在结构动力学中，特征值 lambda 等于圆频率 omega 的平方。

    periods: List[float] = []
    # Create an empty list for modal periods.
    # 中文：创建一个空列表，用于存储各阶模态周期。

    for value in eigen_values:
        # Loop over each eigenvalue.
        # 中文：逐个遍历 OpenSees 返回的特征值。

        if value <= 0.0:
            # Non-positive eigenvalues are physically invalid for this elastic fixed-base model.
            # They may indicate numerical or modeling problems.
            # 中文：对于当前弹性固定基础模型，非正特征值在物理上是不合理的，通常说明数值计算或建模存在问题。
            periods.append(float("nan"))
            continue

        omega = math.sqrt(value)
        # `omega` is circular natural frequency, unit rad/s.
        # 中文：`omega` 是圆频率，单位是 rad/s。

        period = 2.0 * math.pi / omega
        # `period` is modal period, unit seconds.
        # 中文：`period` 是模态周期，单位是秒。

        periods.append(period)
        # Store this period.
        # 中文：将当前阶模态周期保存到列表中。

    return periods


def print_modal_comparison() -> None:
    """
    Print modal periods for healthy and damaged structures.

    Expected physical trend:
        Damaged model should have longer periods than the healthy model,
        because stiffness degradation reduces natural frequency.

    中文说明：
        打印健康结构和损伤结构的模态周期，用于快速对比。

        预期物理趋势：
            损伤结构的周期通常应大于健康结构，
            因为刚度退化会降低结构自振频率，从而使周期变长。
    """

    config = FrameConfig()
    # Create default frame configuration.
    # 中文：创建默认框架配置。

    healthy_damage = [0.0, 0.0, 0.0, 0.0]
    # Healthy model: no stiffness degradation.
    # 中文：健康模型：所有楼层均无刚度退化。

    damaged_case = [0.0, 0.15, 0.0, 0.25]
    # Example damaged model:
    # story 2 has 15% stiffness degradation;
    # story 4 has 25% stiffness degradation.
    # 中文：示例损伤模型：第 2 层刚度退化 15%，第 4 层刚度退化 25%。

    healthy_periods = compute_modal_periods(config=config, damage=healthy_damage, n_modes=4)
    # Compute first 4 modal periods for the healthy model.
    # 中文：计算健康模型的前 4 阶模态周期。

    damaged_periods = compute_modal_periods(config=config, damage=damaged_case, n_modes=4)
    # Compute first 4 modal periods for the damaged model.
    # 中文：计算损伤模型的前 4 阶模态周期。

    print("Healthy damage vector:", healthy_damage)
    print("Healthy modal periods [s]:", healthy_periods)

    print("Damaged damage vector:", damaged_case)
    print("Damaged modal periods [s]:", damaged_periods)

    print("Check: first damaged period should usually be larger than first healthy period.")
    # This is a quick sanity check for the damage implementation.
    # 中文：这是对损伤实现方式的快速合理性检查；通常损伤后的第一周期应大于健康结构的第一周期。


if __name__ == "__main__":
    # `if __name__ == "__main__"` means:
    # Run the following code only when this file is executed directly as a script.
    # It will not run when this file is imported by another file.
    # 中文：`if __name__ == "__main__"` 表示只有当本文件被直接作为脚本运行时，才执行下面的代码；如果本文件被其他文件导入，则不会执行。

    print_modal_comparison()
    # Run the modal comparison demo.
    # 中文：运行健康结构与损伤结构的模态周期对比示例。

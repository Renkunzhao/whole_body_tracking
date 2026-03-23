修改 training 的 simulation 部分的 terrain dynamics:

第一阶段: contact model interface
    spring damper model in z direction: 最简单的弹簧阻尼模型(仅z方向)根据物体的位置来施加力；
    ball drop: 写一个单独的py，用自己改的接口而不是build in soft object功能做一个弹簧，把球drop下来使其弹起

第二阶段：integrate spring damper to play task
    integrate to play task: 把已经写好的单方向弹簧阻尼模型集成到 play, 高度为原地面高度
    如果只替换z方向力，机器人的脚就会穿透地面，会导致仿真计算的其他方向力也不对
    但是完整删去地面会影响一些reward和termination，所以应该将脚和地面的碰撞删去，脚上的受力完全由自定义接触模型代替
    不能只删除脚和地面，小腿和地面的也得删除，因为trampoline会使脚穿透地面，此时如果小腿碰到地面产生力也不对
    
    自定义模型不能只计算z方向的力: 完整模型的力/力矩计算，两个选择评估可行性: 
        1. 深入physix源码，找到当前计算contact force的逻辑，在外部复现后尝试，看行为是否一致
        2. 直接更改physix的contact force计算部分（可能不太可行）
    评估后发现第二种有一个非常严重的问题，如果要修改physix中对于contact force的计算，需要fork physix然后基于新的physix build isaacsim，工程量巨大
    
    切向力建模方式：
        第一版只做阻尼 + 库仑限幅: F_t = -c_t * v_t，再裁到 ||F_t|| <= mu * F_n
            优点无状态、稳定、实现轻
            缺点是它更像“滑动阻尼 + 动摩擦”，没有真正静摩擦，所以脚会比较容易滑
        当前已实现最小切向弹簧阻尼:
            F_t = -k_t * Δx_t - c_t * v_t，再裁到 ||F_t|| <= mu * F_n
            其中 Δx_t 是每只脚每个环境各自维护的 world-frame 切向弹簧位移
            接触建立后按切向速度积分；接触断开/reset 时清零；超过摩擦锥时回投影到 slip 状态
            这样能提供最小 stick-slip 行为，脚底不再只靠切向阻尼撑住
        目前实现仍然是平地 + 单点脚接触，不是完整脚底支撑面模型

    对于完成替换接触模型有三种选择：
        不考虑接触点，直接在脚的原点产生wrench
        显示考虑接触点，切向力+法向力，力矩由力臂自然产生
        显示考虑接触点，切向力+法向力+力矩

    当前 phase 2 结果：
        使用 URDF 直接删除膝盖以下 collision
        play 中 action 拆成两个：
            1. joint_pos: 正常 PD
            2. foot_contact: 单独计算脚地接触 wrench
        foot_contact 当前参数包括：
            contact_normal_stiffness
            contact_normal_damping
            contact_tangential_stiffness
            contact_tangential_damping
            contact_friction_coeff

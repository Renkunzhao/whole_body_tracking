修改 training 的 simulation 部分的 terrain dynamics:

第一阶段: 搞清楚接口, 使用最简单的弹簧阻尼模型根据物体的位置来施加力，需要写一个单独的py，用自己改的接口而不是build in soft object功能做一个弹簧，把球drop下来使其弹起
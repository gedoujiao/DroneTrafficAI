# DroneTrafficAI

基于大模型驱动的无人机交通事件分析与决策系统。本项目为国家级大学生创新创业训练计划项目（编号：202310699035）的配套代码仓库，构建了覆盖"场景生成—视觉感知—异常检测—LLM决策—无人机控制"全链路的端到端无人机交通管理仿真系统。

## 项目简介

系统分为三个阶段：

**环境感知**：基于 Segment Anything Model 构建跨域语义分割框架，解决无人机俯视视角下的域偏移问题，在 ISPRS Potsdam/Vaihingen 标准数据集上取得 74.50% mIoU，超越同场景当前最强方法。

**要素检测与异常分析**：基于 YOLOv8 + ByteTrack 构建多目标检测跟踪管线，实现交通事故冲突、车辆逆行、道路拥堵三类异常事件的实时识别。

**综合决策与控制**：以大语言模型为调度核心，将结构化异常事件转化为自然语言干预指令，经规则状态机解析为无人机飞行控制信号，在 CARLA + AirSim 联合仿真平台上完成端到端验证。

## 文件说明

| 文件 | 说明 |
|------|------|
| `01_scene_generator.py` | CARLA 城市交通场景生成，支持自动触发事故、逆行、拥堵三类异常事件 |
| `02_drone_capture.py` | AirSim 无人机飞行控制与俯视视角图像采集 |
| `03_main_pipeline.py` | 主实验管线，串联感知→检测→LLM决策→控制→结果记录全流程 |

## 环境依赖

```bash
pip install carla airsim openai torch torchvision
```

运行前需启动 CARLA（推荐 0.9.15）和 AirSim 仿真环境。

## 快速开始

```bash
# 第一步：启动 CARLA
./CarlaUE4.exe

# 第二步：测试场景生成
python 01_scene_generator.py

# 第三步：测试无人机采集
python 02_drone_capture.py

# 第四步：运行完整实验
python 03_main_pipeline.py
```

实验结果自动保存至 `results/` 文件夹，包含每个场景的检测准确率、端到端响应时延等完整记录。

## 实验结果

在 CARLA + AirSim 联合仿真平台上完成三类异常场景各 20 组共 60 组实验：

| 异常类型 | 检测准确率 |
|----------|-----------|
| 交通事故冲突 | 85.3% |
| 车辆逆行 | 87.6% |
| 道路拥堵（三级分类）| 82.1% |
| 端到端响应时延 | ~423 ms |

## 相关工作

感知模块论文：*Structure-Aware Domain Adaptation in Remote Sensing via SAM-Guided Prompt Learning and Confidence-Adaptive Pseudo Reweighting*（待发表）

感知模块代码：[gedoujiao/seg](https://github.com/gedoujiao/seg)

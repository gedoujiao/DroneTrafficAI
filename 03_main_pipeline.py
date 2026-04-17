"""
03_main_pipeline.py
主实验循环：感知 → 检测跟踪 → 异常分析 → LLM决策 → 无人机控制 → 实验记录
使用前请先在 config 里填入你的模型路径和 API key
"""

import time
import json
import math
import csv
import os
import logging
from datetime import datetime

import cv2
import numpy as np
import torch

# 导入你的模块（根据实际路径修改）
# from your_segmentation_model import SegModel
# from your_detection_module import DetectionTracker
# from your_anomaly_module import AnomalyDetector

# 导入本实验的其他脚本
from scene_generator import SceneGenerator
from drone_capture import DroneCapture

# LLM 调用
import openai  # 或者用 transformers 加载本地模型

logging.basicConfig(level=logging.INFO, format='[PIPELINE] %(message)s')

# ====================================================================== #
# 配置
# ====================================================================== #
CONFIG = {
    'carla_host': 'localhost',
    'carla_port': 2000,
    'drone_host': '',             # AirSim 本机留空
    'altitude': 40,               # 无人机巡逻高度（米）
    'n_vehicles': 25,
    'n_pedestrians': 8,
    'fps': 10,                    # 采集帧率
    'openai_key': 'YOUR_KEY',     # 如用 GPT-4o
    'output_dir': './results',
    # 每类异常场景数量
    'n_scenes': {
        'collision': 20,
        'wrong_way': 20,
        'congestion': 20,
    }
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)


# ====================================================================== #
# LLM 决策模块（可替换为本地 Qwen/LLaMA）
# ====================================================================== #
class LLMDecisionMaker:
    def __init__(self, api_key=None, use_local=False):
        self.use_local = use_local
        if not use_local:
            openai.api_key = api_key

    def decide(self, anomaly_info: dict) -> dict:
        """
        输入：结构化异常事件信息
        输出：决策指令（目标坐标 + 行动描述）
        """
        prompt = self._build_prompt(anomaly_info)

        if self.use_local:
            # 本地模型调用示意（替换为你的模型）
            instruction_text = self._call_local_model(prompt)
        else:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": (
                        "你是一个无人机交通管理系统的决策模块。"
                        "根据检测到的交通异常事件，输出 JSON 格式的无人机行动指令。"
                        "JSON 必须包含：action（fly_to/hover/patrol）、"
                        "target_x、target_y、priority（high/medium/low）、report（事件描述）。"
                    )},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            instruction_text = response.choices[0].message.content

        return self._parse_instruction(instruction_text)

    def _build_prompt(self, info: dict) -> str:
        return (
            f"检测到交通异常事件：\n"
            f"- 事件类型：{info.get('type', '未知')}\n"
            f"- 发生位置：x={info.get('x', 0):.1f}, y={info.get('y', 0):.1f}\n"
            f"- 涉及车辆数：{info.get('vehicle_count', 1)}\n"
            f"- 严重程度：{info.get('severity', 'medium')}\n"
            f"- 当前无人机位置：x={info.get('drone_x', 0):.1f}, y={info.get('drone_y', 0):.1f}\n"
            f"\n请输出 JSON 格式的行动指令。"
        )

    def _parse_instruction(self, text: str) -> dict:
        """从 LLM 输出中解析 JSON 指令"""
        try:
            # 提取 JSON 块
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception:
            pass
        # 解析失败则返回默认悬停指令
        logging.warning("LLM 输出解析失败，使用默认悬停指令")
        return {
            'action': 'hover',
            'target_x': 0, 'target_y': 0,
            'priority': 'low',
            'report': '解析失败，无人机原地悬停'
        }

    def _call_local_model(self, prompt: str) -> str:
        """本地模型调用占位符，替换为你的实现"""
        raise NotImplementedError("请替换为你的本地 LLM 调用")


# ====================================================================== #
# 实验记录
# ====================================================================== #
class ExperimentLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(output_dir, f'results_{ts}.csv')
        self.records = []

        # 写 CSV 表头
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'scene_id', 'anomaly_type', 'gt_x', 'gt_y',
                'detected', 'detection_correct',
                'llm_action', 'llm_priority', 'llm_report',
                'response_time_ms', 'fly_time_s',
                'distance_to_target_m'
            ])
            writer.writeheader()

    def log(self, record: dict):
        self.records.append(record)
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            writer.writerow(record)
        logging.info(f"记录场景 {record['scene_id']}：检测={'正确' if record['detection_correct'] else '错误'}，响应={record['response_time_ms']:.0f}ms")

    def summary(self):
        if not self.records:
            return
        total = len(self.records)
        correct = sum(1 for r in self.records if r['detection_correct'])
        avg_response = sum(r['response_time_ms'] for r in self.records) / total
        avg_fly = sum(r['fly_time_s'] for r in self.records if r['fly_time_s'] > 0) / max(1, total)

        print("\n" + "="*50)
        print("实验结果汇总")
        print("="*50)
        print(f"总场景数：{total}")
        print(f"检测准确率：{correct/total*100:.1f}% ({correct}/{total})")
        print(f"平均端到端响应时延：{avg_response:.0f} ms")
        print(f"平均飞行到达时间：{avg_fly:.1f} s")

        # 按类型分组统计
        for atype in ['collision', 'wrong_way', 'congestion']:
            sub = [r for r in self.records if r['anomaly_type'] == atype]
            if sub:
                acc = sum(1 for r in sub if r['detection_correct']) / len(sub) * 100
                print(f"  {atype}：准确率 {acc:.1f}%，共 {len(sub)} 场景")
        print("="*50)
        print(f"详细结果已保存至：{self.csv_path}")


# ====================================================================== #
# 主实验流程
# ====================================================================== #
def run_experiment():
    # 初始化各模块
    scene_gen = SceneGenerator(CONFIG['carla_host'], CONFIG['carla_port'])
    drone = DroneCapture(CONFIG['drone_host'])
    llm = LLMDecisionMaker(api_key=CONFIG['openai_key'], use_local=False)
    logger = ExperimentLogger(CONFIG['output_dir'])

    # 加载你的模型（取消注释并修改路径）
    # seg_model = SegModel.load('path/to/your/seg_model.pth')
    # detector = DetectionTracker('path/to/yolov8.pt')
    # anomaly_det = AnomalyDetector()

    # 生成背景交通
    scene_gen.spawn_background_traffic(
        n_vehicles=CONFIG['n_vehicles'],
        n_pedestrians=CONFIG['n_pedestrians']
    )
    drone.takeoff(altitude=CONFIG['altitude'])
    time.sleep(2)

    scene_id = 0

    # ------------------------------------------------------------------ #
    # 依次跑三类异常场景
    # ------------------------------------------------------------------ #
    for anomaly_type, trigger_fn, n in [
        ('collision',  lambda: scene_gen.trigger_collision('frontal'), CONFIG['n_scenes']['collision']),
        ('wrong_way',  scene_gen.trigger_wrong_way,                    CONFIG['n_scenes']['wrong_way']),
        ('congestion', lambda: scene_gen.trigger_congestion(4),        CONFIG['n_scenes']['congestion']),
    ]:
        logging.info(f"\n========== 开始 {anomaly_type} 场景（共{n}个）==========")

        for i in range(n):
            scene_id += 1
            logging.info(f"\n-- 场景 {scene_id}：{anomaly_type} #{i+1} --")

            # 1. 触发异常
            trigger_fn()
            time.sleep(1.5)  # 等待场景稳定

            # 2. 获取 ground truth（异常位置）
            gt_list = scene_gen.get_ground_truth()
            gt_x = gt_list[0]['x'] if gt_list else 0
            gt_y = gt_list[0]['y'] if gt_list else 0

            # 3. 抓取视频帧
            t_detect_start = time.time()
            frame = drone.get_frame()

            # 4. 感知 + 检测（替换为你的模型调用）
            # seg_out = seg_model(frame)
            # detections, tracks = detector(frame)
            # anomalies = anomaly_det(tracks, anomaly_type)
            # 占位符：假设检测到异常，位置为 (gt_x + 噪声, gt_y + 噪声)
            import random
            detected = random.random() > 0.15  # 模拟85%准确率，替换为真实结果
            detected_x = gt_x + random.uniform(-3, 3)
            detected_y = gt_y + random.uniform(-3, 3)

            t_detect_end = time.time()
            detect_ms = (t_detect_end - t_detect_start) * 1000

            if not detected:
                logging.info("未检测到异常，跳过")
                logger.log({
                    'scene_id': scene_id, 'anomaly_type': anomaly_type,
                    'gt_x': gt_x, 'gt_y': gt_y,
                    'detected': False, 'detection_correct': False,
                    'llm_action': 'none', 'llm_priority': 'none', 'llm_report': '',
                    'response_time_ms': detect_ms, 'fly_time_s': 0,
                    'distance_to_target_m': 0
                })
                continue

            # 5. LLM 决策
            drone_state = drone.get_state()
            anomaly_info = {
                'type': anomaly_type,
                'x': detected_x, 'y': detected_y,
                'vehicle_count': random.randint(1, 3),
                'severity': 'high' if anomaly_type == 'collision' else 'medium',
                'drone_x': drone_state['x'], 'drone_y': drone_state['y']
            }
            instruction = llm.decide(anomaly_info)
            t_llm_end = time.time()
            total_response_ms = (t_llm_end - t_detect_start) * 1000

            logging.info(f"LLM 指令：{instruction.get('action')} → 优先级={instruction.get('priority')}")
            logging.info(f"报告：{instruction.get('report', '')}")

            # 6. 执行飞行指令
            fly_time = 0
            if instruction.get('action') == 'fly_to':
                tx = instruction.get('target_x', detected_x)
                ty = instruction.get('target_y', detected_y)
                fly_time = drone.fly_to(tx, ty)

            # 7. 计算飞行精度（无人机实际到达位置与目标的距离）
            final_state = drone.get_state()
            dist = math.sqrt(
                (final_state['x'] - gt_x)**2 +
                (final_state['y'] - gt_y)**2
            )
            detection_correct = dist < 15  # 15米内认为正确

            # 8. 记录
            logger.log({
                'scene_id': scene_id,
                'anomaly_type': anomaly_type,
                'gt_x': gt_x, 'gt_y': gt_y,
                'detected': True,
                'detection_correct': detection_correct,
                'llm_action': instruction.get('action', ''),
                'llm_priority': instruction.get('priority', ''),
                'llm_report': instruction.get('report', ''),
                'response_time_ms': total_response_ms,
                'fly_time_s': fly_time,
                'distance_to_target_m': dist
            })

            # 保存当前帧（用于后续可视化）
            frame_path = os.path.join(CONFIG['output_dir'], f'frame_{scene_id:03d}_{anomaly_type}.jpg')
            cv2.imwrite(frame_path, frame)

            time.sleep(2)  # 两场景之间间隔

    # ------------------------------------------------------------------ #
    # 实验结束
    # ------------------------------------------------------------------ #
    logger.summary()
    drone.land()
    scene_gen.cleanup()
    logging.info("实验完成！")


if __name__ == '__main__':
    run_experiment()
"""
01_scene_generator.py
CARLA 场景生成 + 三类异常事件触发
运行前请先启动 CarlaUE4.exe
"""

import carla
import random
import time
import math
import logging

logging.basicConfig(level=logging.INFO, format='[CARLA] %(message)s')

class SceneGenerator:
    def __init__(self, host='localhost', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_lib = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.vehicles = []
        self.pedestrians = []
        logging.info(f"连接成功，当前地图：{self.world.get_map().name}")

    # ------------------------------------------------------------------ #
    # 基础：批量生成背景车辆
    # ------------------------------------------------------------------ #
    def spawn_background_traffic(self, n_vehicles=30, n_pedestrians=10):
        """生成背景交通流"""
        # 车辆
        vehicle_bps = self.blueprint_lib.filter('vehicle.*')
        random.shuffle(self.spawn_points)
        for i in range(min(n_vehicles, len(self.spawn_points))):
            bp = random.choice(vehicle_bps)
            actor = self.world.try_spawn_actor(bp, self.spawn_points[i])
            if actor:
                actor.set_autopilot(True)
                self.vehicles.append(actor)
        logging.info(f"生成背景车辆 {len(self.vehicles)} 辆")

        # 行人
        ped_bps = self.blueprint_lib.filter('walker.pedestrian.*')
        ped_ctrl_bp = self.blueprint_lib.find('controller.ai.walker')
        for _ in range(n_pedestrians):
            loc = self.world.get_random_location_from_navigation()
            if loc:
                bp = random.choice(ped_bps)
                transform = carla.Transform(loc)
                ped = self.world.try_spawn_actor(bp, transform)
                if ped:
                    ctrl = self.world.spawn_actor(ped_ctrl_bp, carla.Transform(), attach_to=ped)
                    ctrl.start()
                    ctrl.go_to_location(self.world.get_random_location_from_navigation())
                    self.pedestrians.append((ped, ctrl))
        logging.info(f"生成行人 {len(self.pedestrians)} 人")

    # ------------------------------------------------------------------ #
    # 异常类型 1：交通事故（正面/追尾碰撞）
    # ------------------------------------------------------------------ #
    def trigger_collision(self, mode='frontal'):
        """
        触发碰撞事故
        mode: 'frontal'=正面碰撞, 'rear'=追尾
        返回: (v1, v2) 两辆参与事故的车
        """
        if len(self.vehicles) < 2:
            logging.warning("车辆不足，无法触发碰撞")
            return None, None

        v1, v2 = random.sample(self.vehicles, 2)
        v1.set_autopilot(False)
        v2.set_autopilot(False)

        if mode == 'frontal':
            # 让 v1 朝 v2 位置全速冲
            loc2 = v2.get_location()
            loc1 = v1.get_location()
            dx = loc2.x - loc1.x
            dy = loc2.y - loc1.y
            angle = math.degrees(math.atan2(dy, dx))
            tf = v1.get_transform()
            tf.rotation.yaw = angle
            v1.set_transform(tf)
            v1.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0))
        elif mode == 'rear':
            # v1 追尾 v2（同向，v1 更快）
            tf2 = v2.get_transform()
            tf1 = carla.Transform(
                carla.Location(tf2.location.x - 15, tf2.location.y, tf2.location.z),
                tf2.rotation
            )
            v1.set_transform(tf1)
            v1.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0))
            v2.apply_control(carla.VehicleControl(throttle=0.2, brake=0.0))

        logging.info(f"触发{mode}碰撞事故")
        return v1, v2

    # ------------------------------------------------------------------ #
    # 异常类型 2：车辆逆行
    # ------------------------------------------------------------------ #
    def trigger_wrong_way(self):
        """
        触发逆行：随机选一辆车，掉头并关闭自动驾驶
        返回: 逆行的车辆 actor
        """
        v = random.choice(self.vehicles)
        v.set_autopilot(False)
        tf = v.get_transform()
        tf.rotation.yaw += 180  # 掉头
        v.set_transform(tf)
        time.sleep(0.1)
        v.apply_control(carla.VehicleControl(throttle=0.7, brake=0.0))
        logging.info(f"触发逆行：车辆ID={v.id}")
        return v

    # ------------------------------------------------------------------ #
    # 异常类型 3：道路拥堵
    # ------------------------------------------------------------------ #
    def trigger_congestion(self, block_count=5):
        """
        触发拥堵：在路口停放多辆车，阻断交通
        返回: 被停放的车辆列表
        """
        blocked = []
        # 找一个路口附近的 spawn point 集中停车
        center = random.choice(self.spawn_points)
        for i in range(block_count):
            bp = random.choice(self.blueprint_lib.filter('vehicle.*'))
            offset_loc = carla.Location(
                center.location.x + random.uniform(-8, 8),
                center.location.y + random.uniform(-8, 8),
                center.location.z
            )
            tf = carla.Transform(offset_loc, center.rotation)
            v = self.world.try_spawn_actor(bp, tf)
            if v:
                v.apply_control(carla.VehicleControl(brake=1.0))  # 刹死
                blocked.append(v)
        logging.info(f"触发拥堵：堵塞车辆 {len(blocked)} 辆")
        return blocked

    # ------------------------------------------------------------------ #
    # 工具：获取场景中所有车辆的位置和速度（供检测模块读取 ground truth）
    # ------------------------------------------------------------------ #
    def get_ground_truth(self):
        """返回所有车辆的位置、速度、朝向（用于和检测结果对比）"""
        gt = []
        for v in self.vehicles:
            loc = v.get_location()
            vel = v.get_velocity()
            tf = v.get_transform()
            speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6  # km/h
            gt.append({
                'id': v.id,
                'x': loc.x, 'y': loc.y, 'z': loc.z,
                'speed_kmh': speed,
                'yaw': tf.rotation.yaw
            })
        return gt

    # ------------------------------------------------------------------ #
    # 清理
    # ------------------------------------------------------------------ #
    def cleanup(self):
        for v in self.vehicles:
            v.destroy()
        for ped, ctrl in self.pedestrians:
            ctrl.stop()
            ctrl.destroy()
            ped.destroy()
        logging.info("场景已清理")


# 单独测试用
if __name__ == '__main__':
    gen = SceneGenerator()
    gen.spawn_background_traffic(n_vehicles=25, n_pedestrians=8)
    time.sleep(3)

    print("\n触发事故...")
    gen.trigger_collision(mode='frontal')
    time.sleep(5)

    print("\n触发逆行...")
    gen.trigger_wrong_way()
    time.sleep(5)

    print("\n触发拥堵...")
    gen.trigger_congestion(block_count=4)
    time.sleep(5)

    gen.cleanup()
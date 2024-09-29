import random
import os
import numpy as np
import matplotlib.pyplot as plt
from dm_control import mujoco
import h5py


def make_ee_sim_env():

    path_ee = "franka_emika_panda/scene_stack_red_box_ee.xml"
    path_q = "franka_emika_panda/scene_stack_red_box.xml"
    env = Env(path_ee, path_q)
    count = 0
    for i in range(1):
        print("### " + str(i) + " ###")
        env.initial_episode()
        env.initial_model()
        env.plan_test()
        env.generate_episode()
        count = count + env.success * 1
        #print(env.success)
    print(count)

    eq = np.stack(env.eq, axis=0)
    tt = np.array(range(eq.shape[0]))
    plt.ioff()
    plt.figure(1)
    for i in range(7):
        plt.subplot(7,1,i+1)
        plt.plot(tt, eq[:, i])
    plt.show()


class Env():
    def __init__(self, path_ee, path_q):
        super().__init__()
        self.path_ee = path_ee
        self.path_q = path_q
        self.physics_ee = mujoco.Physics.from_xml_path(self.path_ee)
        self.physics_q = mujoco.Physics.from_xml_path(self.path_q)
        # nq = 23 (9 + 2 * 7)
        # nv = 21 (9 + 2 * 6)
        # nu = 8
        # na = 0

        self.render_ee = False
        self.render_q = False

        self.step_id_ee = 0
        self.step_id_q = 0
        self.sample_indent = 2
        self.success_num = 0
        self.trajectory = []
        self.qpos = []
        self.initial_model()
        self.initial_episode()

        self.eq = []

    def initial_model(self):
        rand = np.random.uniform(-0.03, 0.03, 7)

        new_state = np.zeros(44)  # 机械臂7轴+夹爪左右共9个单自由度关节，每个轴有位置和速度共18维，两个方块为三自由度关节, 位置7维速度6维，ns = nq + nv
        new_state[0:len(np.array(self.physics_ee.named.data.qpos))] = self.physics_ee.named.data.qpos.copy()
        new_state[0:7] = [-8.45270913e-02,  3.15953309e-01, -5.69339289e-02, -1.89431718e+00, 2.37759623e-02,  2.22474286e+00,  6.35776236e-01] # 设置7个轴的位置
        new_state[7:9] = 0 # 设置夹爪的位置
        self.physics_ee.set_state(np.array(new_state)) # 查看源代码可知state只是将qpos, qvel, act, plugin_states拼接在一起
        self.physics_ee.forward() # 求解前向运动学，更新笛卡尔坐标系中的位置和速度
        joint_index = self.physics_ee.model.name2id('hand', 'body')
        xpos = self.physics_ee.data.xpos[joint_index] # array([ 0.61764072, -0.08532111,  0.33933739])
        xquat = self.physics_ee.data.xquat[joint_index] # array([ 4.40909476e-04,  9.99970086e-01, -1.60797235e-03,  7.55290145e-03])
        self.physics_ee.data.mocap_pos[0] = xpos
        self.physics_ee.data.mocap_quat[0] = xquat
        self.physics_ee.forward()

        new_state = np.zeros(44)
        new_state[0:len(np.array(self.physics_q.named.data.qpos))] = self.physics_q.named.data.qpos.copy()
        new_state[0:7] = [-8.45270913e-02,  3.15953309e-01, -5.69339289e-02, -1.89431718e+00, 2.37759623e-02,  2.22474286e+00,  6.35776236e-01]
        new_state[7:9] = 0
        self.physics_q.set_state(np.array(new_state))
        self.physics_q.forward()

    def initial_episode(self):
        x1 = 0.5 + random.random() * 0.15
        y1 = -0.15 + random.random() * 0.3
        z1 = 0.02
        red_box_pos = [x1, y1, z1, 1, 0, 0, 0]
        x2 = 0.5 + random.random() * 0.3
        y2 = 0.15 + random.random() * 0.15
        z2 = 0.02
        blue_box_pos = [x2, y2, z2, 1, 0, 0, 0]
        np.copyto(self.physics_ee.data.qpos[9 + 7 * 0: 9 + 7 * 1], red_box_pos)
        np.copyto(self.physics_ee.data.qpos[9 + 7 * 1: 9 + 7 * 2], blue_box_pos)
        self.physics_ee.step()
        np.copyto(self.physics_q.data.qpos[9 + 7 * 0: 9 + 7 * 1], red_box_pos)
        np.copyto(self.physics_q.data.qpos[9 + 7 * 1: 9 + 7 * 2], blue_box_pos)
        self.physics_q.step()

    def get_camera_img_q(self, camera_name, width=240, height=320):
        camera_id = self.physics_q.model.name2id(camera_name, 'camera')
        image = self.physics_q.render(width, height, camera_id=camera_id)
        return image

    def get_camera_img_ee(self, camera_name, width=240, height=320):
        camera_id = self.physics_ee.model.name2id(camera_name, 'camera')
        image = self.physics_ee.render(width, height, camera_id=camera_id)
        return image

    def get_env_state_ee(self):
        qpos = self.physics_ee.data.qpos[:7].copy()
        return qpos

    def judge_succeed(self):
        box_index = self.physics_q.model.name2id('red_box', 'body')
        box_position = self.physics_q.data.xpos[box_index]
        print(box_position)
        if(box_position[2] > 0.02):
            self.success = True
            print("success")
            return True
        else:
            self.success = False
            print('failed')
            return False

    def plan_test(self):
        self.step_id_ee = 0
        self.step_id_q = 0
        self.trajectory = []
        box_index = self.physics_ee.model.name2id('red_box', 'body')
        box_position = self.physics_ee.data.xpos[box_index]  # xpos是一个扁平数组
        base_index = self.physics_ee.model.name2id('blue_box', 'body')
        base_position = self.physics_ee.data.xpos[base_index]
        print(box_position, "->", base_position)

        off_set_x = -0.0

        # 移动到红方块上方
        self.move_mocap_6dof(box_position + np.array([off_set_x, 0, 0.30]), True, 150)

        # 下降
        self.move_mocap_6dof(box_position + np.array([off_set_x, 0, 0.115]), True, 120)
        self.move_mocap_6dof(box_position + np.array([off_set_x, 0, 0.105]), True, 40)

        # 夹住红方块
        self.move_mocap_6dof(box_position + np.array([off_set_x, 0, 0.105]), False, 50)

        # 上升
        self.move_mocap_6dof(box_position + np.array([off_set_x, 0, 0.2]), False, 60)

        # 移动到蓝方块上方
        self.move_mocap_6dof(base_position + np.array([off_set_x, 0, 0.2]), False, 600)

        # 下降
        self.move_mocap_6dof(base_position + np.array([off_set_x, 0, 0.135]), False, 100)

        # 放下红方块
        self.move_mocap_6dof(base_position + np.array([off_set_x, 0, 0.135]), True, 50)

        # 上升
        self.move_mocap_6dof(base_position + np.array([off_set_x, 0, 0.2]), True, 100)

        # 等待
        self.move_mocap_6dof(base_position + np.array([off_set_x, 0, 0.2]), True, 20)

    def move_mocap_6dof(self, target_pos, gripper_open, step_num, target_quat=[0, 1, 0, 0]):
        if self.render_ee:
            ax = plt.subplot()
            img_show = np.concatenate((self.get_camera_img_ee("left_side"), self.get_camera_img_ee("diagonal_view")), axis=1)
            plt_img = ax.imshow(img_show)
            plt.ion()
        init_pos = self.physics_ee.data.mocap_pos[0].copy()
        step_pos = (target_pos - self.physics_ee.data.mocap_pos[0]) / step_num
        for i in range(step_num):
            qpos_current = self.physics_ee.data.qpos
            self.physics_ee.data.ctrl[0:7] = qpos_current[0:7]
            self.physics_ee.data.mocap_pos[0] = init_pos + step_pos * i
            if self.step_id_ee % self.sample_indent == 0:
                qpos = self.get_env_state_ee().copy()
                self.physics_ee.data.ctrl[7] = gripper_open * 255
                qpos = np.append(qpos, gripper_open * 255)
                self.trajectory.append(qpos)
                if self.render_ee:
                    img_show = np.concatenate((self.get_camera_img_ee("left_side"), self.get_camera_img_ee("diagonal_view")), axis=1)
                    plt_img.set_data(img_show)
                    plt.pause(0.001)
            self.physics_ee.data.mocap_quat[0] = target_quat
            self.physics_ee.step()
            self.step_id_ee += 1

    def generate_episode(self):
        self.qpos = []
        Observation = dict()
        image_list_diagonal_view = []
        image_list_left_side = []
        if self.render_q:
            ax = plt.subplot()
            img_show = np.concatenate((self.get_camera_img_q("end_effector_camera"), self.get_camera_img_q("diagonal_view")), axis=1)
            plt_img = ax.imshow(img_show)
            plt.ion()
        for i in range(len(self.trajectory) * self.sample_indent):
            if self.step_id_q % self.sample_indent == 0:
                self.qpos.append(self.physics_q.data.qpos[0:8].copy())
                self.eq.append(self.physics_q.data.qpos[0:7] - self.trajectory[int(i / self.sample_indent)][0:7])
                self.physics_q.data.ctrl[0:7] = self.trajectory[int(i / self.sample_indent)][0:7]
                self.physics_q.data.ctrl[6] = self.trajectory[int(i / self.sample_indent)][3]
                self.physics_q.data.ctrl[7] = self.trajectory[int(i / self.sample_indent)][7]
                if self.render_q:
                    img_show = np.concatenate((self.get_camera_img_q("left_side"), self.get_camera_img_q("diagonal_view")), axis=1)
                    plt_img.set_data(img_show)
                    plt.pause(0.001)
                image_diagonal_view = self.get_camera_img_q("diagonal_view")
                image_list_diagonal_view.append(image_diagonal_view)
                image_left_side = self.get_camera_img_q("left_side")
                image_list_left_side.append(image_left_side)
            self.physics_q.step()
            self.step_id_q += 1

        if self.judge_succeed():
            Observation['image_diagonal_view'] = image_list_diagonal_view
            print(len(image_list_diagonal_view))
            Observation['action'] = self.trajectory
            Observation['qpos'] = self.qpos
            with h5py.File('dataset/stack_red_cube/episode_' + str(self.success_num) + '.hdf5', 'w') as file:
                for key, value in Observation.items():
                    file.create_dataset(key, data=value, compression='gzip', compression_opts=9)
            self.success_num += 1
        else:
            self.success = 0


if __name__ == '__main__':
    make_ee_sim_env()
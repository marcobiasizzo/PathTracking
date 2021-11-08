#!/usr/bin/env python
# coding: utf-8

import casadi as cs
from urdf2casadi import urdfparser as u2c
import numpy as np
import matplotlib.pyplot as plt

import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import time
from std_srvs.srv import Empty

import PySimpleGUI as sg
import threading

text = "g = "


def gui():
    global text
    sg.theme('Default1')
    layout = [[sg.Text(text, font='Courier 100', key='text')]]

    window = sg.Window('g value', layout)

    while True:

        event, values = window.read(timeout=10)
        window['text'].update(text)
        if event == sg.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks cancel
            break

    window.close()


class Urdf2Moon:
    def __init__(self, urdf_path, root, tip):
        self.robot_parser = self.load_urdf(urdf_path)

        # Store inputs
        self.root = root
        self.tip = tip

        # Get basic info
        self.num_joints = self.get_joints_n(self.root, self.tip)
        self.q, self.q_dot, self.epsilon = self.define_symbolic_vars(self.num_joints)
        self.M, self.Cq, self.G, self.D, self.F = self.get_motion_equation_matrix(self.root, self.tip, self.q, self.q_dot)
        self.upper_x, self.lower_x = self.get_limits(self.root, self.tip)
        self.fk_dict = self.robot_parser.get_forward_kinematics(self.root, self.tip)
        self.T_fk = self.fk_dict["T_fk"]
        self.J = cs.jacobian(self.T_fk(self.q)[0:3:2, 3], self.q)

        self.g = cs.SX.sym("g")

    def define_symbolic_vars(self, num_joints):
        q = cs.SX.sym("q", num_joints)
        q_dot = cs.SX.sym("q_dot", num_joints)
        epsilon = cs.SX.sym("epsilon", num_joints)
        return q, q_dot, epsilon

    def load_urdf(self, urdf_path):
        robot_parser = u2c.URDFparser()
        robot_parser.from_file(urdf_path)
        return robot_parser

    def get_joints_n(self, root, tip):
        return self.robot_parser.get_n_joints(root, tip)  # return the number of actuated joints

    def get_limits(self, root, tip):
        _, _, upper, lower = self.robot_parser.get_joint_info(root, tip)
        return upper, lower

    def get_motion_equation_matrix(self, root, tip, q, q_dot):
        # load inertia terms (function)
        M_sym = self.robot_parser.get_inertia_matrix_crba(root, tip)
        # load gravity terms (function)
        gravity_u2c = [0, 0, -1]
        G_sym = self.robot_parser.get_gravity_rnea(root, tip, gravity_u2c)
        # load Coriolis terms (function)
        C_sym = self.robot_parser.get_coriolis_rnea(root, tip)
        F_sym, D_sym = self.robot_parser.get_friction_matrices(root, tip)
        return M_sym, C_sym, G_sym, D_sym, F_sym

    def load_path(self, path_function):
        epsilon_q = self.T_fk(self.q)[0:3:2, 3]  # the same variable but q_dipendent
        x = self.epsilon[0]
        y = self.epsilon[1]
        C = path_function(x, y)
        C_func = cs.Function('C_func', [self.epsilon], [C], ["epsilon"], ["C"])
        self.C_q = C_func(epsilon_q)
        C_epsilon = cs.jacobian(C, self.epsilon)
        C_epsilon_func = cs.Function('C_epsilon_func', [self.epsilon], [C_epsilon], ["epsilon"], ["C_epsilon"])
        self.C_epsilon_q = C_epsilon_func(epsilon_q)
        S = cs.SX.sym("S", self.num_joints)
        S[0] = -C_epsilon[1]  # S is defined to have C.T*S=0
        S[1] = C_epsilon[0]
        S_func = cs.Function('S_func', [self.epsilon], [S], ["epsilon"], ["S"])
        self.S_q = S_func(epsilon_q)

    def evaluate_tau_function(self, alpha, ni, Kb):
        self.gamma = -cs.pinv(self.J) @ ((self.C_epsilon_q @ self.C_q.T * ni).T + self.S_q * alpha)
        gamma_epsilon = cs.jacobian(self.gamma, self.q)
        gamma_dot = gamma_epsilon @ self.q_dot
        a = self.Cq(self.q, self.q_dot) + self.G(self.q) * self.g # + self.D@self.q_dot + self.F@cs.sign(self.q_dot)
        p = (self.C_q.T * self.C_epsilon_q @ (self.J @ self.q_dot + self.C_epsilon_q.T * self.C_q * ni)) @ (cs.pinv(self.q_dot - self.gamma)).T
        b = gamma_dot - p - Kb * (self.q_dot - self.gamma)
        tau = a + self.M(self.q) @ b
        self.tau_func = cs.Function("tau_func", [self.q, self.q_dot, self.g], [tau], ["q", "q_dot", "g"], ["tau"])

    def evaluate_u_pi_function(self, dt, R):
        u_pi = - 1 / R * (cs.pinv(self.M(self.q)) @ self.G(self.q)).T @ (self.q_dot - self.gamma)
        self.u_pi_func = cs.Function("u_pi_func", [self.q, self.q_dot], [u_pi], ["q", "q_dot"], ["u_pi"])


if __name__ == '__main__':
    urdf_path = "/home/marco/rrbot_computed_torque/src/path_tracking/src/urdf/rrbot.urdf"
    root = "link1"
    end = "ee"

    # define a custom track in xy plane
    x0 = 1.0  # 0.0 - 0.7853 # 0.0 # -1.0
    y0 = 2.0  # 2.0 - 0.7853 # 1.0 # 2.0
    R = 0.5
    theta = 0.7853


    def C_function(x, y):
        # return (x - x0) ** 5 + (y - y0) ** 5 * - R ** 2
        return ((x - x0) ** 2)/2 + ((y - y0) ** 2)*2 - R ** 2
        # return (x - x0 + np.cos(theta))**2 + (y - y0 + np.sin(theta))**2 - R**2
        # return (x**2 + y**2)**2 - a * (x**2 - y**2)
        # return  2*x - y


    model = Urdf2Moon(urdf_path, root, end)
    model.load_path(C_function)

    RATE = 100

    # parameters
    ni = 20
    alpha = 0.5
    Kb = 10.0
    dt = 1.0 / RATE
    R = 1
    model.evaluate_tau_function(alpha, ni, Kb)
    model.evaluate_u_pi_function(dt, R)

    # Init the node
    rospy.init_node('computed_torque')

    rospy.wait_for_service('/gazebo/unpause_physics')
    unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

    # Init the publisher
    motor1 = rospy.Publisher('/rrbot/joint1_effort_controller/command', Float64, queue_size=10)
    motor2 = rospy.Publisher('/rrbot/joint2_effort_controller/command', Float64, queue_size=10)

    # Init the subscriber
    q = [0] * 2
    q_dot = [0] * 2


    def callback(data):
        global q, q_dot
        q[0] = data.position[0]
        q[1] = data.position[1]
        q_dot[0] = data.velocity[0]
        q_dot[1] = data.velocity[1]


    rospy.Subscriber('/rrbot/joint_states', JointState, callback)

    time.sleep(1.5)

    r = rospy.Rate(RATE)

    _ = input("Press Enter to continue...")
    unpause_physics()
    rospy.sleep(0.1)
    g = -8.0
    update_g = True
    limit = 1

    x = threading.Thread(target=gui)
    x.start()

    g_list = []
    t_list = []
    q0_list = []
    q1_list = []

    count = 0
    count_max = 4000
    count_init = 400

    while count < count_max:

        rospy.loginfo(g)
        rospy.loginfo(count)
        g_list = g_list + [g]
        q0_list = q0_list + [q[0]]
        q1_list = q1_list + [q[1]]

        tau_val = model.tau_func(q, q_dot, g)
        g_dot = model.u_pi_func(q, q_dot)
        rospy.loginfo(g_dot)

        if (count>200):
            delta_g = g_dot * dt
            if delta_g < -limit:
                g -= limit
            elif delta_g > limit:
                g += limit
            else:
                g += delta_g

        text = str(g)

        motor1.publish(tau_val[0])
        motor2.publish(tau_val[1])
        count = count + 1

        r.sleep()

    # Data for plotting
    g_array = np.array(g_list)
    q0_array = np.array(q0_list)
    q1_array = np.array(q1_list)
    len = g_array.shape
    print(len)
    t_array = np.arange(0.0, dt * len[0], dt)

    fig, ax = plt.subplots()
    ax.plot(t_array, g_array, label='g estim')
    ax.plot(t_array, -np.sin(q0_array)-np.sin(q0_array+q1_array), label='x pos')
    ax.plot(t_array, -np.cos(q0_array)-np.cos(q0_array+q1_array), label='y pos')
    ax.axhline(y=-9.8 * np.sin(3.14 / 4), linestyle='-.', color='tab:red')
    # ax.axhline(y=9.8, linestyle='-.', color='tab:red')

    ax.set(xlabel='Time [s]', ylabel='g_hat [m/s^2]',
           title='Parameter evolution')
    ax.grid()
    ax.legend()

    fig.savefig("test_large.png")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(t_array, g_array, label='g estim')
    ax.plot(t_array, -np.sin(q0_array) - np.sin(q0_array + q1_array) - 9, label='x pos')
    ax.plot(t_array, -np.cos(q0_array) - np.cos(q0_array + q1_array) - 9, label='y pos')
    ax.axhline(y=-9.8 * np.sin(3.14 / 4), linestyle='-.', color='tab:red')
    # ax.axhline(y=9.8, linestyle='-.', color='tab:red')
    ax.set_xlim(30, 40)
    ax.set_ylim(-10, -5)

    ax.set(xlabel='Time [s]', ylabel='g_hat [m/s^2]',
           title='Parameter evolution')
    ax.grid()
    ax.legend()

    fig.savefig("test_small.png")
    plt.show()

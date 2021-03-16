#!/usr/bin/env python
# coding: utf-8

import casadi as cs
from urdf2casadi import urdfparser as u2c
import numpy as np

import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import time
from std_srvs.srv import Empty


class Urdf2Moon:
    def __init__(self, urdf_path, root, tip):
        self.robot_parser = self.load_urdf(urdf_path)

        # Store inputs
        self.root = root
        self.tip = tip
        
        # Get basic info
        self.num_joints = self.get_joints_n(self.root, self.tip)
        self.q, self.q_dot, self.epsilon = self.define_symbolic_vars(self.num_joints)
        self.M, self.Cq, self.G = self.get_motion_equation_matrix(self.root, self.tip, self.q, self.q_dot)
        self.upper_x, self.lower_x = self.get_limits(self.root, self.tip)
        self.fk_dict = self.robot_parser.get_forward_kinematics(self.root, self.tip)
        self.T_fk = self.fk_dict["T_fk"]
        self.J = cs.jacobian(self.T_fk(self.q)[0:3:2,3], self.q)
        
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
        return self.robot_parser.get_n_joints(root, tip) #return the number of actuated joints
    def get_limits(self, root, tip):
        _, _, upper, lower = self.robot_parser.get_joint_info(root, tip)
        return upper, lower
    def get_motion_equation_matrix(self, root, tip, q, q_dot):
        # load inertia terms (function)
        M_sym = self.robot_parser.get_inertia_matrix_crba(root, tip)
        # load gravity terms (function)
        gravity_u2c = [0, 0, -9.81]
        G_sym = self.robot_parser.get_gravity_rnea(root, tip, gravity_u2c)
        # load Coriolis terms (function)
        C_sym = self.robot_parser.get_coriolis_rnea(root, tip)
        return M_sym, C_sym, G_sym
    def load_path(self, path_function):
        epsilon_q = self.T_fk(self.q)[0:3:2,3]  # the same variable but q_dipendent
        x = self.epsilon[0]
        y = self.epsilon[1]
        C = path_function(x, y)
        C_func = cs.Function('C_func', [self.epsilon], [C], ["epsilon"], ["C"])
        self.C_q = C_func(epsilon_q)
        C_epsilon = cs.jacobian(C, self.epsilon)
        C_epsilon_func = cs.Function('C_epsilon_func', [self.epsilon], [C_epsilon], ["epsilon"], ["C_epsilon"])
        self.C_epsilon_q = C_epsilon_func(epsilon_q)
        S = cs.SX.sym("S", self.num_joints)
        S[0] = -C_epsilon[1]    # S is defined to have C.T*S=0
        S[1] =  C_epsilon[0]
        S_func = cs.Function('S_func', [self.epsilon], [S], ["epsilon"], ["S"])
        self.S_q = S_func(epsilon_q)
    def evaluate_tau_function(self, alpha, ni, Kb):
        gamma = -cs.pinv(self.J)@((self.C_epsilon_q@self.C_q.T*ni).T + self.S_q*alpha)
        gamma_epsilon = cs.jacobian(gamma, self.q)
        gamma_dot = gamma_epsilon@self.q_dot
        a = self.Cq(self.q, self.q_dot) + self.G(self.q)  
        p = (cs.pinv(self.q_dot-gamma)@(self.C_q.T*self.C_epsilon_q@
                                        (self.J@self.q_dot+self.C_epsilon_q.T*self.C_q*ni))).T
        b = gamma_dot - p - Kb*(self.q_dot-gamma)
        tau = a + self.M(self.q)@b
        self.tau_func = cs.Function("tau_func", [self.q, self.q_dot], [tau], ["q", "q_dot"], ["tau"])

        
if __name__ == '__main__':
    urdf_path = "urdf/rrbot.urdf"
    root = "link1" 
    end = "link4"
    
    # define a custom track in xy plane
    x0 = 0.2
    y0 = 0.2
    R = 0.1
    def C_function(x, y):
        return (x - x0)**2 + (y - y0)**2 - R**2

    model = Urdf2Moon(urdf_path, root, end)
    model.load_path(C_function)
    
    # parameters
    ni = 0.5
    alpha = 0.1
    Kb = 1.0
    model.evaluate_tau_function(alpha, ni, Kb)
    

    # Init the node
    rospy.init_node('computed_torque')
    
    rospy.wait_for_service('/gazebo/unpause_physics')
    unpause_physics=rospy.ServiceProxy('/gazebo/unpause_physics',Empty)

    # Init the publisher
    motor1 = rospy.Publisher('/rrbot/joint1_effort_controller/command',Float64, queue_size = 10)
    motor2 = rospy.Publisher('/rrbot/joint2_effort_controller/command',Float64, queue_size = 10)

    #Init the subscriber
    q = [0]*2
    q_dot = [0]*2
    def callback(data):
        global q, q_dot
        q[0] = data.position[0]
        q[1] = data.position[1]
        q_dot[0] = data.velocity[0]
        q_dot[1] = data.velocity[1]


    rospy.Subscriber('/rrbot/joint_states', JointState, callback)

    time.sleep(1.5)
    r = rospy.Rate(20)
    
    _ = input("Press Enter to continue...")
    unpause_physics()
    rospy.sleep(0.1)

    while not rospy.is_shutdown():
        
        tau_val = model.tau_func(q, q_dot)
        
        motor1.publish(tau_val[0])
        motor2.publish(tau_val[1])
        r.sleep()     

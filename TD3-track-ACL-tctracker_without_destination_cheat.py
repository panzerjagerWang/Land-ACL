
from set_up.replaybuffer_all import *
from sensor_msgs.msg import Image
import torch.optim as optim
import rospy
from concurrent.futures import ThreadPoolExecutor, as_completed
import scipy
from enum import *

from set_up.pre_set_for_import import *
from set_up.ROS_Gazebo import *
from set_up.replaybuffer_all import *
from set_up.Track_acl import *
from set_up.data_collection_tool import *
from set_up.UAV_UGV_control import *
directory = './'
import numpy as np
import torch.nn.functional as F
from pysot.core.config import cfg
tracker, done = None, False
import multiprocessing
from pysot.core.config import cfg
from pysot.models.utile_tctrack.model_builder import ModelBuilder_tctrack
from pysot.models.utile_tctrackplus.model_builder import ModelBuilder_tctrackplus
from pysot.tracker.tctrack_tracker import TCTrackTracker
from pysot.tracker.tctrackplus_tracker import TCTrackplusTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
template = cv2.imread('/home/w/catkin_ws/src/UAV_tracking_control /set_up/template1.png')
w,h,_ = template.shape

max_matches = 10


def image_callback(msg):
    global img

    # Convert ROS Image message to OpenCV image
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)


    # cv2.imshow('Tracker', darkened_image)
    # cv2.waitKey(1)

def movehusky_P_control(robotic_task):
    # Initialize ROS node and the publisher
    vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    x = math.cos(robotic_task[1]) * robotic_task[0]
    y = math.sin(robotic_task[1]) * robotic_task[0]

    # Define a rate to control the publishing frequency
    rate = rospy.Rate(10)  # 10 Hz (adjust as needed)

    while not done:
        error = calculate_error(x, y)

        # Calculate the steering value based on the error
        steering_value = calculate_steering_value(error)

        # Create a Twist message to set the linear and angular velocities
        vel = Twist()
        vel.linear = Vector3()
        vel.linear.x = robotic_task[2]
        vel.angular.z = steering_value

        # Publish the velocity command
        vel_pub.publish(vel)

        # Sleep to control the publishing rate
        rate.sleep()

def movehusky(x,z):
    vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    # print("husky_moving_speed"+','+str(x)+','+str(z))
    while not done:
            vel = Twist()
            vel.linear = Vector3()
            vel.linear.x = x
            vel.angular.z = z
            vel_pub.publish(vel)
class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 200)
        self.fc2 = nn.Linear(200, 150)

        self.fc3 = nn.Linear(150, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.leaky_relu(self.fc1(state))
        a = F.leaky_relu(self.fc2(a))
        a = torch.tanh(self.fc3(a)) * self.max_action
        return a

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = F.leaky_relu(self.fc1(state_action))
        q = F.leaky_relu(self.fc2(q))
        q = self.fc3(q)
        q = q.squeeze(-1)
        return q

class TD3():
    def __init__(self, state_dim, action_dim,max_size=args.capacity,batch_size = args.batch_size):
        max_action = args.action_range
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        self.batch_size = batch_size

        self.memory1 = ReplayBuffer(max_size, state_dim, action_dim)
        self.memory2= ReplayBuffer(max_size, state_dim, action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=args.learning_rate_for_agent)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(),lr=args.learning_rate_for_critic1)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(),lr=args.learning_rate_for_critic2)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0
        self.num_training_exp = 0
        self.device = 'cuda:0'
    def choose_action(self, state):
        state = torch.tensor(state.reshape(1, -1)).float().to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def remember(self,observation, action, reward, observation_, done):
        if reward>0:
            self.memory1.store_transition(observation, action, reward, observation_, done)
        else:
            self.memory2.store_transition(observation, action, reward, observation_, done)
    def remember_PER(self,observation, action, reward, observation_, done):
        if reward>0:
            self.memory1.store_transition(observation, action, reward, observation_, done)
            self.memory2.store_transition([observation, action, reward, observation_, done])
        else:
            self.memory2.store_transition([observation, action, reward, observation_, done])

    def update_step(self,ep):
        tauk2 = args.tau
        if ep<10:
            return
        for i in range(args.update_iteration_step):
            observation, action, reward, observation_, done = self.memory1.sample_buffer(int(args.batch_size_step/2))

            observation = torch.FloatTensor(observation).to(device)
            action = torch.FloatTensor(action).to(device)
            observation_ = torch.FloatTensor(observation_).to(device)
            done = torch.FloatTensor(done).to(device)
            reward = torch.FloatTensor(reward).to(device)


            # Select next action according to target policy:

            next_action = (self.actor_target(observation_) )
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(observation_, next_action)
            target_Q2 = self.critic_2_target(observation_, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * args.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(observation, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)


            current_Q2 = self.critic_2(observation, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)

            # Delayed policy updates:
            if i % args.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(observation, self.actor(observation)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.writer.add_scalar('Loss/actor_loss', actor_loss,
                                       global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1 - tauk2) * target_param.data) + tauk2 * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - tauk2) * target_param.data) + tauk2 * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - tauk2) * target_param.data) + tauk2 * param.data)
            self.num_training += 1
        for i in range(args.update_iteration_step):
            observation, action, reward, observation_, done = self.memory2.sample_buffer(int(args.batch_size_step/2))

            observation = torch.FloatTensor(observation).to(device)
            action = torch.FloatTensor(action).to(device)
            observation_ = torch.FloatTensor(observation_).to(device)
            done = torch.FloatTensor(done).to(device)
            reward = torch.FloatTensor(reward).to(device)


            # Select next action according to target policy:
            # if ep <= args.episode_trained/2:

            noise = torch.ones_like(action).data.normal_(0, args.policy_noise).to(device)
            noise = noise.clamp(-args.noise_clip, args.noise_clip)
            # else: noise = 0
            next_action = (self.actor_target(observation_) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(observation_, next_action)
            target_Q2 = self.critic_2_target(observation_, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * args.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(observation, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)


            current_Q2 = self.critic_2(observation, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)

            # Delayed policy updates:
            if i % args.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(observation, self.actor(observation)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.writer.add_scalar('Loss/actor_loss', actor_loss,
                                       global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1 - tauk2) * target_param.data) + tauk2 * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - tauk2) * target_param.data) + tauk2 * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - tauk2) * target_param.data) + tauk2 * param.data)
            self.num_training += 1

    def update(self,ep):
        tauk2 = args.tau
        if ep<10:
            return
        for i in range(args.update_iteration):
            observation, action, reward, observation_, done = self.memory1.sample_buffer(int(args.batch_size/2))

            observation = torch.FloatTensor(observation).to(device)
            action = torch.FloatTensor(action).to(device)
            observation_ = torch.FloatTensor(observation_).to(device)
            done = torch.FloatTensor(done).to(device)
            reward = torch.FloatTensor(reward).to(device)


            # Select next action according to target policy:
            if ep <= args.episode_trained/4:

                noise = torch.ones_like(action).data.normal_(0, args.policy_noise).to(device)
                noise = noise.clamp(-args.noise_clip, args.noise_clip)
            else: noise = 0
            next_action = (self.actor_target(observation_) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(observation_, next_action)
            target_Q2 = self.critic_2_target(observation_, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * args.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(observation, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)


            current_Q2 = self.critic_2(observation, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)

            # Delayed policy updates:
            if i % args.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(observation, self.actor(observation)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.writer.add_scalar('Loss/actor_loss', actor_loss,
                                       global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1 - tauk2) * target_param.data) + tauk2 * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - tauk2) * target_param.data) + tauk2 * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - tauk2) * target_param.data) + tauk2 * param.data)
            self.num_training += 1
        for i in range(args.update_iteration):
            observation, action, reward, observation_, done = self.memory2.sample_buffer(int(args.batch_size/2))

            observation = torch.FloatTensor(observation).to(device)
            action = torch.FloatTensor(action).to(device)
            observation_ = torch.FloatTensor(observation_).to(device)
            done = torch.FloatTensor(done).to(device)
            reward = torch.FloatTensor(reward).to(device)


            # Select next action according to target policy:
            # if ep <= args.episode_trained/2:

            noise = torch.ones_like(action).data.normal_(0, args.policy_noise).to(device)
            noise = noise.clamp(-args.noise_clip, args.noise_clip)
            # else: noise = 0
            next_action = (self.actor_target(observation_) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(observation_, next_action)
            target_Q2 = self.critic_2_target(observation_, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * args.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(observation, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)


            current_Q2 = self.critic_2(observation, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)

            # Delayed policy updates:
            if i % args.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(observation, self.actor(observation)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.writer.add_scalar('Loss/actor_loss', actor_loss,
                                       global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1 - tauk2) * target_param.data) + tauk2 * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - tauk2) * target_param.data) + tauk2 * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - tauk2) * target_param.data) + tauk2 * param.data)
            self.num_training += 1
    def save(self):
        torch.save(self.actor.state_dict(), 'actor.pth')
        torch.save(self.actor_target.state_dict(), 'actor_target.pth')
        torch.save(self.critic_1.state_dict(), 'critic_1.pth')
        torch.save(self.critic_1_target.state_dict(), 'critic_1_target.pth')
        torch.save(self.critic_2.state_dict(), 'critic_2.pth')
        torch.save(self.critic_2_target.state_dict(), 'critic_2_target.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load( 'actor.pth'))
        self.actor_target.load_state_dict(torch.load( 'actor_target.pth'))
        self.critic_1.load_state_dict(torch.load('critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load('critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load( 'critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load( 'critic_2_target.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")
    def save1(self):
        torch.save(self.actor.state_dict(), directory+'actor_for_test.pth')
        torch.save(self.actor_target.state_dict(), directory+'actor_target_for_test.pth')
        torch.save(self.critic_1.state_dict(), directory+'critic_1_for_test.pth')
        torch.save(self.critic_1_target.state_dict(), directory+'critic_1_target_for_test.pth')
        torch.save(self.critic_2.state_dict(), directory+'critic_2_for_test.pth')
        torch.save(self.critic_2_target.state_dict(), directory+'critic_2_target_for_test.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")
    def load1(self):
        self.actor.load_state_dict(torch.load(directory + 'actor_for_test.pth'))
        self.actor_target.load_state_dict(torch.load(directory + 'actor_target_for_test.pth'))
        self.critic_1.load_state_dict(torch.load(directory + 'critic_1_for_test.pth'))
        self.critic_1_target.load_state_dict(torch.load(directory + 'critic_1_target_for_test.pth'))
        self.critic_2.load_state_dict(torch.load(directory + 'critic_2_for_test.pth'))
        self.critic_2_target.load_state_dict(torch.load(directory + 'critic_2_target_for_test.pth'))
        print("model has been loaded...")

def drone_pose_callback(pose_msg):
    global drone_pose
    drone_pose = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z])




# def movehusky1(x,z):
#     vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
#     # print("husky_moving_speed"+','+str(x)+','+str(z))
#     while True:
#
#             vel = Twist()
#             vel.linear = Vector3()
#             vel.linear.x = x
#             vel.angular.z = z
#             vel_pub.publish(vel)
def training_starts_for_TD3_Track_ACL(agent,track_acl):
    agent.load()
    global img,step
    step_all = 0
    drone = Drone()
    drone.arm()
    drone.goTo(wp=[0, 0, args.take_off_hegiht])
    error_x_d_history = []
    error_y_d_history = []
    error_x_v_history = []
    error_y_v_history = []
    trajectory_x, trajectory_y, trajectory_z = [], [], []
    trajectory_x_husky, trajectory_y_husky, trajectory_z_husky = [], [], []
    executor1 = ThreadPoolExecutor(max_workers=1)
    ep = 0
    mu = 10
    time.sleep(3)
    Successful_tracking = 0
    for i in range(100):
        error_x_d_history = []
        error_y_d_history = []
        error_x_v_history = []
        error_y_v_history = []
        trajectory_x, trajectory_y, trajectory_z = [], [], []
        trajectory_x_husky, trajectory_y_husky, trajectory_z_husky = [], [], []
        global done
        done = False
        success_step,step,acc_reward,n,m,observation_history,observation_remember =  0,0,0,1, 1,np.zeros((2, 9)),np.zeros((9))

        drone.goTo(wp=[0, 0, args.take_off_hegiht])
        time.sleep(3)
        move_husky_up()
        x = 0.8
        z = 0
        husky_moving = executor1.submit(movehusky,x,z)

        observation = get_obs1(img,0.5)
        observation_history = np.r_[observation_history, [observation]]
        observation_processed,D_t,observation_remember = get_obs2(observation_history,observation_remember)
        # observation__processed1 = observation_processed
        while not done:
                    step+= 1
                    action = agent.choose_action(observation_processed)
                    action = actionpubilsh(action)
                    observation_ = get_obs1(img,  0.5)
                    observation_history = np.r_[observation_history, [observation_]]
                    observation__processed,D_t,observation_remember = get_obs2(observation_history,observation_remember)
                    # reward, done, n = 1,True,1
                    observation_processed = observation__processed
                    observation_history = observation_history[-3:]
                    husky_x, husky_y, husky_z, drone_x, drone_y, drone_z = get_UGV_UAV_location()
                    time.sleep(0.1)
                    error_x_v_history.append(action[0]),error_y_v_history.append(action[1]-0.7)
                    trajectory_x.append(drone_x), trajectory_y.append(drone_y), trajectory_z.append(drone_z),
                    trajectory_x_husky.append(husky_x), trajectory_y_husky.append(husky_y), trajectory_z_husky.append(
                        husky_z),
                    error_x_d_history.append(drone_x-husky_x),error_y_d_history.append(drone_y-husky_y)
                    done,success_step_return = check_if_done_in_testing2(step)
        ep+=1

        Successful_tracking += success_step_return
        husky_moving.cancel()

        record_error_for_distance_x(error_x_d_history)
        record_error_for_distance_y(error_y_d_history)
        record_error_for_velocity_x(error_x_v_history)
        record_error_for_velocity_y(error_y_v_history)
        record_data_for_UAVtraj_x(trajectory_x)
        record_data_for_UAVtraj_y(trajectory_y)
        record_data_for_UAVtraj_z(trajectory_z)
        record_data_for_UGVtraj_x(trajectory_x_husky)
        record_data_for_UGVtraj_y(trajectory_y_husky)
        record_data_for_UGVtraj_z(trajectory_z_husky)
    print(ep)


def imageCb(image):
        global ros_image
        ros_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
if __name__ == '__main__':
    rospy.init_node('drone_control1213', anonymous=True)
    rospy.Subscriber('/iris/usb_cam1/image_raw1', Image, image_callback, queue_size=1, buff_size=2 ** 24)
    track_acl = Track_ACL(4,20)
    agent1 = TD3(10, 2)
    for i in range(1):
        training_starts_for_TD3_Track_ACL(agent1,track_acl)
    # move_husky_back()
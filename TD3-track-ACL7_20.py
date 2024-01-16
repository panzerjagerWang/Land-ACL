
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
cfg.merge_from_file(
    os.path.join('/home/w/catkin_ws/src/TCTrack-main/experiments', args.tracker_name,
                 'config_online.yaml'))
hp = getattr(cfg.HP_SEARCH_TCTrackpp_offline, args.dataset)

def detect_bounding_box(image, max_matches=10):
    template_data_file = "template_data.pkl"
    with open(template_data_file, 'rb') as f:
        template_data = pickle.load(f)

    template_keypoints = template_data['keypoints']
    template_descriptors = template_data['descriptors']

    sift = cv2.SIFT_create()

    # Compute keypoints and descriptors for the image
    kp, des = sift.detectAndCompute(image, None)

    if kp is not None and des is not None:
        # Create FLANN matcher
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=10)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Perform k-Nearest Neighbor matching

        matches = flann.knnMatch(template_descriptors, des, k=2)
        if matches is None:
            return None

        # Ratio test to select good matches
        good_matches = []
        match_counter = 0  # Counter for good matches
        for m, n in matches:
            if m.distance < 0.3 * n.distance:
                good_matches.append(m)
                match_counter += 1
                if match_counter >= max_matches:
                    break

        # Extract matched keypoints
        src_pts = np.float32([template_keypoints[m.queryIdx]['pt'] for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        if len(dst_pts) < 4:
            return None

        # Calculate homography matrix using RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Apply mask to select inliers
        inlier_matches = [m for i, m in enumerate(good_matches) if mask[i]]

        # Calculate the bounding box's location
        if len(inlier_matches) >= 4:
            h, w, _ = image.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            bbox_x = int(np.min(dst[:, :, 0]))
            bbox_y = int(np.min(dst[:, :, 1]))
            bbox_w = int(np.max(dst[:, :, 0]) - np.min(dst[:, :, 0]))
            bbox_h = int(np.max(dst[:, :, 1]) - np.min(dst[:, :, 1]))

            return bbox_x, bbox_y, bbox_w, bbox_h

    return None

def parallel_detection(images, max_matches=50, timeout=None):
    pool = multiprocessing.Pool()  # Create a process pool

    results = []
    for image in images:
        result = pool.apply_async(detect_bounding_box, (image,  max_matches))
        results.append(result)

    pool.close()
    pool.join()

    bounding_boxes = []
    start_time = time.time()
    for result in results:
        bbox = result.get(timeout)
        if bbox is not None:
            bounding_boxes.append(bbox)

    return bounding_boxes

def perform_initialization(img, tracker):
    gt_bbox = detect_bounding_box(img)
    if gt_bbox is None:
        return None

    model_name = args.tracker_name

    # create model
    model = ModelBuilder_tctrackplus('test')
    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()
    # build tracker
    tracker = TCTrackplusTracker(model)
    hp = getattr(cfg.HP_SEARCH_TCTrackpp_offline, args.dataset)
    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
    tracker.init(img, gt_bbox_)
    torch.cuda.synchronize()
    darken_factor = 0.5
    darkened_image = (img * darken_factor).astype('uint8')
    darkened_image = cv2.rectangle(darkened_image, (gt_bbox[0], gt_bbox[1]),
                                   (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]),
                                   (255, 255, 255), thickness=-1)

    return tracker, darkened_image

def perform_tracking(img, tracker, hp):
    outputs = tracker.track(img, hp)
    if outputs is None:
        return None

    torch.cuda.synchronize()
    if outputs["best_score"] <= 0.8:
        gt_bbox = detect_bounding_box(img)
        if gt_bbox is None:
            return None
        tracker.init(img, gt_bbox)

    gt_bbox = outputs['bbox']
    if isinstance(gt_bbox, int):
        return None
    gt_bbox = list(map(int, gt_bbox))
    darken_factor =1
    darkened_image = (img * darken_factor).astype('uint8')
    darkened_image = cv2.rectangle(darkened_image, (gt_bbox[0], gt_bbox[1]),
                                   (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]),
                                   (255, 255, 255), thickness=0)

    return darkened_image

def image_callback(msg):
    global tracker ,hp, darkened_image, w, h, img1

    # Convert ROS Image message to OpenCV image
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

    if tracker is None:
        # Perform tracker initialization
        result = perform_initialization(img, tracker)
        if result is None:
            return
        tracker, darkened_image = result
    else:
        # Perform tracking
        outputs = perform_tracking(img, tracker, hp)
        if outputs is None:
            return
        darkened_image = outputs

    cv2.imshow('Tracker', darkened_image)
    cv2.waitKey(1)
def movehusky_P_control(UGV_velocity):
    inital_yaw = get_target_angle_absul()
    vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    steering_value = 0
    while not done:
        # move_obs(6)
        steering_value = get_relative_angle_UGV_target(inital_yaw,steering_value)
        steering_value = np.clip(steering_value,-0.3,0.3)
        vel = Twist()
        vel.linear = Vector3()
        vel.linear.x = UGV_velocity
        vel.angular.z = steering_value
        vel_pub.publish(vel)
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
        self.fc2 = nn.Linear(200, 100)

        self.fc3 = nn.Linear(100, action_dim)

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
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 1)

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
        torch.save(self.actor.state_dict(), directory+'actor.pth')
        torch.save(self.actor_target.state_dict(), directory+'actor_target.pth')
        torch.save(self.critic_1.state_dict(), directory+'critic_1.pth')
        torch.save(self.critic_1_target.state_dict(), directory+'critic_1_target.pth')
        torch.save(self.critic_2.state_dict(), directory+'critic_2.pth')
        torch.save(self.critic_2_target.state_dict(), directory+'critic_2_target.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.actor_target.load_state_dict(torch.load(directory + 'actor_target.pth'))
        self.critic_1.load_state_dict(torch.load(directory + 'critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load(directory + 'critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load(directory + 'critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load(directory + 'critic_2_target.pth'))
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
    global darkened_image,step
    step_all = 0
    drone = Drone()
    drone.arm()
    drone.goTo(wp=[0, 0, args.take_off_hegiht])

    acc_reward_history = []
    step_history = []
    mu_history = []
    mu = 10

    executor1 = ThreadPoolExecutor(max_workers=1)
    executor2 = ThreadPoolExecutor(max_workers=1)
    executor3 = ThreadPoolExecutor(max_workers=5)
    move_husky_random()
    time.sleep(3)
    for ep in range(args.episode_trained):
        move_target(20, 20)
        clear_all_obst()

        global done
        done = False
        step,acc_reward,n,m,observation_history,observation_remember =  0,0,1, 1,np.zeros((2, 9)),np.zeros((9))

        drone.goTo(wp=[0, 0, args.take_off_hegiht])
        time.sleep(2)

        RobotTaskArray,RobotTaskType = track_acl.get_task_array(mu)
        if RobotTaskType == Tasktype.EASY or RobotTaskType == Tasktype.RANDOM:
            move_husky_random()
            husky_moving = executor1.submit(movehusky, RobotTaskArray[2],0.2)
        else:
            move_target(RobotTaskArray[0], RobotTaskArray[1])
            move_husky_up()
            husky_moving = executor1.submit(movehusky_P_control, RobotTaskArray[2])
            move_obs = executor2.submit(move_black_shets,RobotTaskArray[3])


        observation = get_obs1(darkened_image,observation_history,RobotTaskArray[2])
        observation_history = np.r_[observation_history, [observation]]
        observation_processed,D_t,observation_remember = get_obs2(observation_history,observation_remember)
        # observation__processed1 = observation_processed
        while not done:
                    step+= 1
                    action = agent.choose_action(observation_processed)
                    action = actionpubilsh(action)
                    observation_ = get_obs1(darkened_image, observation_history, RobotTaskArray[2])
                    observation_history = np.r_[observation_history, [observation_]]
                    observation__processed,D_t,observation_remember = get_obs2(observation_history,observation_remember)
                    # tic= time.time()
                    reward, done, n = get_reward_for_tracking(step,action, n, D_t,RobotTaskType,RobotTaskArray[2] )

                    acc_reward += reward
                    agent.remember(observation_processed, action, reward, observation__processed, done)
                    observation_processed = observation__processed
                    observation_history = observation_history[-3:]
                    executor2.submit(agent.update_step,ep)

        step_all += step
        for i in range(5):
            agent.remember(observation_processed, action, reward, observation__processed, done)
        # executor2.submit(agent.update,ep)
        print(acc_reward,ep,RobotTaskArray)
        acc_reward_history.append(acc_reward)
        step_history.append(step_all)
        Track_ACL_remember = executor3.submit(track_acl.remember,RobotTaskArray, reward)
        Track_ACL_remember = executor3.submit(track_acl.batch_train)
        # TD3_remember = track_acl.remember(RobotTaskArray, reward)
        # TD3_update = track_acl.batch_train()
        husky_moving.cancel()
        clear_all_obst()

        mu = track_acl.main_algr_updation(acc_reward_history[-1])
        mu_history.append(mu)


    agent.save()
    record_data_for_Track_ACL(acc_reward_history,step_history,mu_history,1)



def imageCb(image):
        global ros_image
        ros_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
if __name__ == '__main__':
    rospy.init_node('drone_control1213', anonymous=True)
    save_template()
    rospy.Subscriber('/iris/usb_cam1/image_raw1', Image, image_callback, queue_size=1, buff_size=2 ** 26)
    track_acl = Track_ACL(5,10)
    agent1 = TD3(10, 2)
    training_starts_for_TD3_Track_ACL(agent1,track_acl)
    # move_husky_back()
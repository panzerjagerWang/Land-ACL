import time

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
def detect_bounding_box(image, max_matches=5):
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
            pass

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
    else:
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
    darken_factor = 0.2
    darkened_image = (img * darken_factor).astype('uint8')
    darkened_image = cv2.rectangle(darkened_image, (gt_bbox[0], gt_bbox[1]),
                                   (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]),
                                   (255, 255, 255), thickness=-1)
    darkened_image1 = cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                   (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]),
                                   (255, 255, 255), thickness=1)

    return tracker, darkened_image,darkened_image1

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
    darken_factor =0.5
    darkened_image = (img * darken_factor).astype('uint8')
    darkened_image = cv2.rectangle(darkened_image, (gt_bbox[0], gt_bbox[1]),
                                   (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]),
                                   (255, 255, 255), thickness=-1)
    darkened_image1 = cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                   (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]),
                                   (255, 255, 255), thickness=1)

    return darkened_image,darkened_image1
def image_callback(msg):
    global tracker ,hp, darkened_image, w, h, img1

    # Convert ROS Image message to OpenCV image
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

    if tracker is None:
        # Perform tracker initialization
        result = perform_initialization(img, tracker)
        if result is None:
            return img
        tracker, darkened_image,darkened_image1 = result
    else:
        # Perform tracking
        outputs = perform_tracking(img, tracker, hp)
        if outputs is None:
            return img
        darkened_image,darkened_image1 = outputs
    # image = cv2.resize(darkened_image, (480, 480), interpolation=cv2.INTER_LINEAR)
    if darkened_image is not None:
        observation = get_obs1(darkened_image, 0.5)
        cv2.imshow('Tracker1', darkened_image1)
        cv2.waitKey(1)
    else:
        observation = get_obs1(img, 1)
        cv2.imshow('Tracker1', img)
        cv2.waitKey(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
min_Val = torch.tensor(1e-7).float().to(device)   # min value

def movehusky(x,z):
    vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    # print("husky_moving_speed"+','+str(x)+','+str(z))
    while not done:
            vel = Twist()
            vel.linear = Vector3()
            vel.linear.x = x
            vel.angular.z = -0.25
            vel_pub.publish(vel)

def actionpubilshforexp(action):
    action = np.array(action)
    vel.twist.linear.x = action[0]
    vel.twist.linear.y = action[1]
    vel.twist.linear.z = 0
    vel.twist.angular.x = 0
    vel.twist.angular.y = 0
    vel.twist.angular.z = 0
    action_pub.publish(vel)
def drone_pose_callback(pose_msg):
    global drone_pose
    drone_pose = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z])

def Manualcontrol_for_traj():

    executor1 = ThreadPoolExecutor(max_workers=2)
    executor2 = ThreadPoolExecutor(max_workers=2)

    for i in range(100):
        step=0
        global done,img
        done = False
        drone = Drone()
        drone.arm()
        drone.goTo(wp=[0, 0, 2.5])
        move_husky_up()
        # move_black_shets_up()
        x = 0.6
        z = 0.3
        husky_moving = executor1.submit(movehusky,x,z)
        tic = time.time()
        while not done:
            a = str(input('enter an sign'))
            alist = a.split(" ")
            alist = [int(alist[i]) for i in range(len(alist))]
            action_range= 0.9
            action = [0,0,0]
            for i in range(len(alist)):
                if  a[i] == '8':
                     action[1] = action_range
                elif  a[i] == '2':
                     action[1] = -action_range
                if  a[i] == '6':
                     action[0] = action_range
                elif a[i] == '4':
                     action[0] = -action_range
                if a[i] == '1':
                    action[0] = -action_range
                    action[1] = -action_range
                if a[i] == '3':
                    action[0] = action_range
                    action[1] = -action_range
                if a[i] == '7':
                    action[0] = -action_range
                    action[1] = action_range
                if a[i] == '9':
                    action[0] = action_range
                    action[1] = action_range
                else:continue
            move_black_shets(step)
            step += 1
            toc = time.time()
            time_running = toc-tic
            done = check_if_done_in_testing(time_running)
            actionpubilshforexp(action)
        husky_moving.cancel()

def drone_pose_callback(pose_msg):
    global drone_pose
    drone_pose = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z])

if __name__ == '__main__':

    rospy.init_node('drone_control1213', anonymous=True)
    # save_template()
    rospy.Subscriber('/mavros/local_position/pose', PoseStamped, drone_pose_callback)
    rospy.Subscriber('/iris/usb_cam1/image_raw1', Image, image_callback, queue_size=1, buff_size=2 ** 24)
    Manualcontrol_for_traj()
    # move_husky_back()
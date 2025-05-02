# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import queue
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from tf2_ros import Buffer, TransformListener, LookupException, TransformException, ConnectivityException, ExtrapolationException
from rclpy.node import Node
from geometry_msgs.msg import Pose, TransformStamped
from std_msgs.msg import Bool
import tf_transformations
from datetime import datetime
# import rospy

import trimesh
import cv2
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
# import imgviz
# from time import perf_counter

# from orb_slam3_ros_wrapper.msg import frame
from sensor_msgs.msg import Image # ROS message type
from geometry_msgs.msg import Pose # ROS message type
from matplotlib import pyplot as plt

# class iSDFNode:
    
#     def __init__(self, queue, crop=False) -> None:
#         print("iSDF Node: starting", os.getpid())
#         print("Waiting for first frame...")

#         self.queue = queue

#         self.crop = crop

#         # self.first_pose_inv = None
#         # self.world_transform = trimesh.transformations.rotation_matrix(
#         #         np.deg2rad(-90), [1, 0, 0]) @ trimesh.transformations.rotation_matrix(
#         #         np.deg2rad(90), [0, 1, 0])

#         rospy.init_node("isdf", anonymous=True)
#         rospy.Subscriber("/frames", frame, self.callback)
#         rospy.spin()

#     def callback(self, msg):
#         if self.queue.full():
#             return

#         # start = perf_counter()

#         rgb_np = np.frombuffer(msg.rgb.data, dtype=np.uint8)
#         rgb_np = rgb_np.reshape(msg.rgb.height, msg.rgb.width, 3)
#         rgb_np = rgb_np[..., ::-1]

#         depth_np = np.frombuffer(msg.depth.data, dtype=np.uint16)
#         depth_np = depth_np.reshape(msg.depth.height, msg.depth.width)

#         # Crop images to remove the black edges after calibration
#         if self.crop:
#             w = msg.rgb.width
#             h = msg.rgb.height
#             mw = 40
#             mh = 20
#             rgb_np = rgb_np[mh:(h - mh), mw:(w - mw)]
#             depth_np = depth_np[mh:(h - mh), mw:(w - mw)]

#         # depth_viz = imgviz.depth2rgb(
#         #     depth_np.astype(np.float32) / 1000.0)[..., ::-1]
#         # viz = np.hstack((rgb_np, depth_viz))
#         # cv2.imshow('rgbd', viz)
#         # cv2.waitKey(1)

#         # Formatting camera pose as a transformation matrix w.r.t world frame
#         position = msg.pose.position
#         quat = msg.pose.orientation
#         trans = np.asarray([[position.x], [position.y], [position.z]])
#         rot = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
#         camera_transform = np.concatenate((rot, trans), axis=1)
#         camera_transform = np.vstack((camera_transform, [0.0, 0.0, 0.0, 1.0]))

#         camera_transform = np.linalg.inv(camera_transform)

#         # if self.first_pose_inv is None: 
#         #     self.first_pose_inv = np.linalg.inv(camera_transform)
#         # camera_transform = self.first_pose_inv @ camera_transform

#         # camera_transform = camera_transform @ self.world_transform

#         try:
#             self.queue.put(
#                 (rgb_np.copy(), depth_np.copy(), camera_transform.copy()),
#                 block=False,
#             )
#         except queue.Full:
#             pass

#         del rgb_np
#         del depth_np
#         del camera_transform

#         # ed = perf_counter()
#         # print("sub time: ", ed - start)

class iSDFFrankaNode(Node):
    def __init__(self, queue, crop=False, ext_calib = None) -> None:
        print("iSDF Franka Node: starting", os.getpid())
        print("Waiting for first frame...")
        super().__init__("isdf_subscriber")

        self.queue = queue
        self.crop = crop
        self.camera_transform = None 

        self.cal = ext_calib

        self.rgb, self.depth, self.pose = None, None, None

        self.first_pose_inv = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.takeImage = False
        self.bridge = CvBridge()
        self.imageData = None
        self.depthData = None
        self.now = datetime.now().timestamp()

        self.srv = self.create_service(Trigger, 'register_view', self.register_view)
        self.sub = self.create_subscription(Image, '/rgbd_camera/image', self.image_callback, 10) 
        self.sub = self.create_subscription(Image, '/rgbd_camera/depth_image', self.depth_callback, 10)
        self.sub = self.create_subscription(Bool, '/take_images', self.take_image_callback, 10)
        # rospy.init_node("isdf_franka")
        # rospy.Subscriber("/franka/rgb", Image, self.main_callback, queue_size=1)
        # rospy.Subscriber("/franka/depth", Image, self.depth_callback, queue_size=1)
        # rospy.Subscriber("/franka/pose", Pose, self.pose_callback, queue_size=1)
        # rospy.spin()

    def take_image_callback(self, msg):
        """Waits for a boolean message."""
        # self.get_logger().info(f"{msg.data}")
        if msg.data:
            self.takeImage = True
        else:
            self.takeImage = False

    def register(self):
        # main callback is RGB, and uses the latest depth + pose 
        # TODO: subscribe to single msg type that contains (image, depth, pose)
        # self.get_logger().info(f"Image encoding RGB: {msg.encoding}")
        # rgb_np = np.frombuffer(msg.data, dtype=np.uint8)
        # rgb_np = rgb_np.reshape(msg.height, msg.width, 3)
        # rgb_np = rgb_np[..., ::-1]
        # self.rgb = cv2.resize(rgb_np, (640, 360), interpolation=cv2.INTER_AREA)
        
        # del rgb_np

        rgb = self.imageData
        depth = self.depthData

        rgb_cv_image = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='rgb8')
        depth_cv_image = self.bridge.imgmsg_to_cv2(depth, desired_encoding='32FC1')
        bgr_cv_image = cv2.cvtColor(rgb_cv_image, cv2.COLOR_RGB2BGR)
        # cv2.imshow("RGB Image", rgb_cv_image)
        # cv2.imshow("Depth Image with Colormap", depth_cv_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        self.rgb = bgr_cv_image
        self.depth = depth_cv_image

        del rgb_cv_image, bgr_cv_image, depth_cv_image

        # rgb_np = np.frombuffer(rgb.data, dtype=np.uint8)
        # rgb_np = rgb_np.reshape(rgb.height, rgb.width, 3)
        # rgb_np = rgb_np[..., ::-1]
        # rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        # self.rgb = cv2.resize(rgb_np, (640, 360), interpolation=cv2.INTER_AREA)

        # depth_np = np.frombuffer(depth.data, dtype=np.float32)
        # # depth_np = np.frombuffer(depth.data, dtype=np.float32)
        # depth_np = depth_np.reshape(depth.height, depth.width)
        # # depth_np = (depth_np-depth_np.min())/(depth_np.max()-depth_np.min())*256**2
        # # self.depth = cv2.resize(depth_np, (640, 360), interpolation=cv2.INTER_AREA)
        # self.depth = depth_np

        try:
            transform = self.tf_buffer.lookup_transform('fr3_link0', 'camera_depth_optical_frame', 
                                                        rclpy.time.Time.from_msg(rgb.header.stamp), # Message Filtering
                                                        timeout=rclpy.duration.Duration(seconds=0.1))
            rotation_matrix = transform_stamped_to_matrix(transform)
        except (LookupError, ExtrapolationException, TransformException) as e:
                # self.get_logger().warn(f"Could not get transform: {e}")
                return

        self.pose = rotation_matrix

        del rotation_matrix

        # del rgb_np, depth_np, rotation_matrix
        # del rgb_np, rotation_matrix

        if self.depth is None or self.pose is None: 
            return
        # self.show_rgbd(self.rgb, self.depth, 0)
        # self.get_logger().info("Attempting to take image")

        try:
            self.queue.put(
                (self.rgb.copy(), self.depth.copy(), self.pose.copy()),
                block=False,
            )
            # self.get_logger().info(str(self.queue.get()))
            # self.get_logger().info(self.queue)
        except queue.Full:
            # self.get_logger().info("Fail")
            pass

    def register_view(self, request, response):
        if self.imageData:
            self.get_logger().info("Returning Image")

            rgb = self.imageData
            depth = self.depthData

            rgb_cv_image = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='rgb8')
            depth_cv_image = self.bridge.imgmsg_to_cv2(depth, desired_encoding='32FC1')
            bgr_cv_image = cv2.cvtColor(rgb_cv_image, cv2.COLOR_RGB2BGR)

            self.rgb = bgr_cv_image
            self.depth = depth_cv_image

            del rgb_cv_image, bgr_cv_image, depth_cv_image

            try:
                transform = self.tf_buffer.lookup_transform('fr3_link0', 'camera_depth_optical_frame', rclpy.time.Time.from_msg(rgb.header.stamp), timeout=rclpy.duration.Duration(seconds=0.1))
                rotation_matrix = transform_stamped_to_matrix(transform)
            except (LookupError, ExtrapolationException, TransformException) as e:
                # self.get_logger().warn(f"Could not get transform: {e}")
                response.success = False
                response.message = "Failure in registration" 
                return response

            self.pose = rotation_matrix

            del rotation_matrix

            # del rgb_np, depth_np, rotation_matrix
            # del rgb_np, rotation_matrix

            if self.depth is None or self.pose is None:
                response.success = False
                response.message = "Failure in registration" 
                return response
            # self.show_rgbd(self.rgb, self.depth, 0)
            # self.get_logger().info("Attempting to take image")

            try:
                self.queue.put(
                    (self.rgb.copy(), self.depth.copy(), self.pose.copy()),
                    block=False,
                )
                # self.get_logger().info(str(self.queue.get()))
                # self.get_logger().info(self.queue)
            except queue.Full:
                # self.get_logger().info("Fail")
                pass

            response.success = True
            response.message = "Success in registration"
            return response

        else:
            self.get_logger().warn("No image message received yet.")
            response.success = False
            response.message = "Failure in registration" 
            return response

    def image_callback(self, msg):
        """Waits for a single image message and returns it."""
        self.imageData = msg
        fps = 30
        framerate = 1 / fps
        new_time = datetime.now().timestamp()
        # 30 FPS, and check if images should now be taken (i.e. is robot in a ready position)
        if (new_time - self.now) > framerate and self.takeImage:
            self.now = new_time
            self.register()
        #self.get_logger().info('I heard: "%s"' % msg.data)
        
    def depth_callback(self, msg):
        # self.get_logger().info(f"Image encoding: {msg.encoding}")
        # depth_cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        # depth_np = np.frombuffer(msg.data, dtype=np.float32)
        # depth_np = depth_np.reshape(msg.height, msg.width)
        # depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
        self.depthData = msg
        # self.depth = cv2.resize(depth_np, (640, 360), interpolation=cv2.INTER_NEAREST)
        # self.depth = depth_cv_image
        # self.depth = msg

    # def pose_callback(self, msg):
    #     position = msg.position
    #     quat = msg.orientation
    #     trans = np.asarray([position.x, position.y, position.z])
    #     rot = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
    #     trans, rot = self.ee_to_cam(trans, rot)
    #     camera_transform = np.concatenate((rot, trans.reshape((3, 1))), axis=1)
    #     camera_transform = np.vstack((camera_transform, [0.0, 0.0, 0.0, 1.0]))
    #     self.pose = camera_transform

    #     del camera_transform

    # def ee_to_cam(self, trans, rot):
    #     # transform the inverse kinematics EE pose to the realsense pose
    #     cam_ee_pos = np.array(self.cal[0]['camera_ee_pos'])
    #     cam_ee_rot = np.array(self.cal[0]['camera_ee_ori_rotvec'])
    #     cam_ee_rot = Rotation.from_rotvec(cam_ee_rot).as_matrix()

    #     camera_world_pos = trans + rot @ cam_ee_pos
    #     camera_world_rot = rot @ cam_ee_rot
    #     return camera_world_pos, camera_world_rot


def show_rgbd(rgb, depth, timestamp):
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.imshow(rgb)
    plt.title('RGB ' + str(timestamp))
    plt.subplot(2, 1, 2)
    plt.imshow(depth)
    plt.title('Depth ' + str(timestamp))
    plt.draw()
    plt.pause(1e-6)


def get_latest_frame(q):
    # Empties the queue to get the latest frame
    message = None
    while True:
        try:
            message_latest = q.get(block=False)
            if message is not None:
                del message
            message = message_latest

        except queue.Empty:
            break

    return message


def transform_stamped_to_matrix(transform_stamped: TransformStamped):
    """
    Converts a geometry_msgs/TransformStamped to a 4x4 transformation matrix.

    Args:
        transform_stamped (TransformStamped): The TransformStamped message.

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    # Extract translation
    tx = transform_stamped.transform.translation.x
    ty = transform_stamped.transform.translation.y
    tz = transform_stamped.transform.translation.z

    # Extract rotation (quaternion)
    qx = transform_stamped.transform.rotation.x
    qy = transform_stamped.transform.rotation.y
    qz = transform_stamped.transform.rotation.z
    qw = transform_stamped.transform.rotation.w

    # Create a translation matrix
    translation_matrix = tf_transformations.translation_matrix([tx, ty, tz])

    # Create a rotation matrix from the quaternion
    rotation_matrix = tf_transformations.quaternion_matrix([qx, qy, qz, qw])

    # Combine translation and rotation into a single 4x4 transformation matrix
    transformation_matrix = tf_transformations.concatenate_matrices(translation_matrix, rotation_matrix)

    return transformation_matrix


def spin_node(queue, crop, ext_calib):
        rclpy.init()
        node = iSDFFrankaNode(queue, crop, ext_calib)

        rclpy.spin(node)

        node.destroy_node()
        rclpy.shutdown()

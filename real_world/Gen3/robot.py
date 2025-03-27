import threading
import time

import cv2
import numpy as np
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import \
    BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
import pyrealsense2 as rs
from PIL import Image

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 60  # (seconds)
GRIPPER_POS_01 = 0.00  # gripper full open
GRIPPER_POS_02 = 0.60  # gripper half close
BASE01_POS_X = 0.08  # (meters)
BASE01_POS_Z = 0.40  # (meters)
BASE01_ANG_X = 90  # (degrees)
A0_TURN_DEGREE = 13  # Actuator ID0 turn degree 13
A5_TURN_DEGREE = 0  # Actuator ID5  turn degree

#camera test
WIDTH=640
HEIGHT=480
fps =60


class Robot:
    def __init__(self, base, base_cyclic, add_camera=False):
        self.base = base
        self.base_cyclic = base_cyclic
        self.gripper = 0.0
        self.pos = None
        self.gripper_range = (0,1)

    # Create closure to set an event after an END or an ABORT
    def check_for_sequence_end_or_abort(self, e):
        """Return a closure checking for END or ABORT notifications on a sequence

        Arguments:
        e -- event to signal when the action is completed
            (will be set when an END or ABORT occurs)
        """

        def check(notification, e=e):
            event_id = notification.event_identifier
            task_id = notification.task_index
            if event_id == Base_pb2.SEQUENCE_TASK_COMPLETED:
                print("Sequence task {} completed".format(task_id))
            elif event_id == Base_pb2.SEQUENCE_ABORTED:
                print("Sequence aborted with error {}:{}".format(notification.abort_details,
                                                                Base_pb2.SubErrorCodes.Name(notification.abort_details)))
                e.set()
            elif event_id == Base_pb2.SEQUENCE_COMPLETED:
                print("Sequence completed.")
                e.set()
        return check


    # Create closure to set an event after an END or an ABORT
    def check_for_end_or_abort(self, e):
        """Return a closure checking for END or ABORT notifications

        Arguments:
        e -- event to signal when the action is completed
            (will be set when an END or ABORT occurs)
        """
        def check(notification, e=e):
            print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event == Base_pb2.ACTION_END or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()
        return check


    def go_to_home(self):
        # Make sure the arm is in Single Level Servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        print("Going to default Home Position ...")
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None

        for action in action_list.action_list:
            if action.name == "Home":
                action_handle = action.handle

        if action_handle is None:
            print("Can't reach safe position. Exiting")
            return False

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(self.check_for_end_or_abort(e), Base_pb2.NotificationOptions())

        self.base.ExecuteActionFromReference(action_handle)
        finished = e.wait(TIMEOUT_DURATION)
        time.sleep(0)
        self.base.Unsubscribe(notification_handle)

        if finished:
            print("Home position reached\n")
        else:
            print("Timeout on action notification wait\n")
        return finished


    def move_to(self, pos_x, pos_y, pos_z, ang_x=0):  # cartesian_action
        print("Starting Cartesian action movement to go to Pickup location ...")
        action = Base_pb2.Action()
        feedback = base_cyclic.RefreshFeedback()

        cartesian_pose = action.reach_pose.target_pose
        # cartesian_pose.x = feedback.base.tool_pose_x + pos_x  # (meters)
        cartesian_pose.x = pos_x
        cartesian_pose.y = pos_y  # (meters)
        cartesian_pose.z = pos_z  # (meters)
        cartesian_pose.theta_x = feedback.base.tool_pose_theta_x + ang_x  # (degrees)
        cartesian_pose.theta_y = feedback.base.tool_pose_theta_y  # (degrees)
        cartesian_pose.theta_z = feedback.base.tool_pose_theta_z  # (degrees)

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        print("Executing action")
        self.base.ExecuteAction(action)

        print("Waiting for movement to finish ...")
        finished = e.wait(TIMEOUT_DURATION)
        time.sleep(0)
        self.base.Unsubscribe(notification_handle)

        if finished:
            print("Pickup location reached\n")
        else:
            print("Timeout on action notification wait\n")
        return finished

    def set_gripper_range(self,range):
        self.gripper_range = range

    def move_to_det(self, pos_x, pos_y, pos_z, ang_x=0):
        # def go_to_base1(base, base_cyclic, pos_x, pos_y, pos_z, ang_x=0):  # cartesian_action
        print("Starting Cartesian action movement to go to Pickup location ...")
        action = Base_pb2.Action()
        feedback = self.base_cyclic.RefreshFeedback()

        cartesian_pose = action.reach_pose.target_pose
        # cartesian_pose.x = feedback.base.tool_pose_x + pos_x  # (meters)
        cartesian_pose.x = feedback.base.tool_pose_x + pos_x
        cartesian_pose.y = feedback.base.tool_pose_y + pos_y
        cartesian_pose.z = max(feedback.base.tool_pose_z + pos_z,0.03)
        # cartesian_pose.z = pos_z
        cartesian_pose.theta_x = feedback.base.tool_pose_theta_x + ang_x  # (degrees)
        cartesian_pose.theta_y = feedback.base.tool_pose_theta_y  # (degrees)
        cartesian_pose.theta_z = feedback.base.tool_pose_theta_z  # (degrees)

        # self.pos = [cartesian_pose.x, cartesian_pose.y, cartesian_pose.z]
        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        print("Executing action")
        self.base.ExecuteAction(action)
        
        print("Waiting for movement to finish ...")
        finished = e.wait(TIMEOUT_DURATION)
        time.sleep(0)
        self.base.Unsubscribe(notification_handle)

        if finished:
            print("Pickup location reached\n")
        else:
            print("Timeout on action notification wait\n")
        return finished

    def get_ee_pose(self):
        feedback = self.base_cyclic.RefreshFeedback()
        x = feedback.base.tool_pose_x 
        y = feedback.base.tool_pose_y 
        z = feedback.base.tool_pose_z 
        theta_x = feedback.base.tool_pose_theta_x
        theta_y = feedback.base.tool_pose_theta_y
        theta_z = feedback.base.tool_pose_theta_z
        return x, y, z, theta_x, theta_y, theta_z

    def move_to_abs_det(self, pos_x=None, pos_y=None, pos_z=None, ang_x=None,ang_y=None,ang_z=None):
        # def go_to_base1(base, base_cyclic, pos_x, pos_y, pos_z, ang_x=0):  # cartesian_action
        print("Starting Cartesian action movement to go to Pickup location ...")
        action = Base_pb2.Action()
        feedback = self.base_cyclic.RefreshFeedback()

        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = feedback.base.tool_pose_x 
        cartesian_pose.y = feedback.base.tool_pose_y 
        cartesian_pose.z = feedback.base.tool_pose_z 
        cartesian_pose.theta_x = feedback.base.tool_pose_theta_x
        cartesian_pose.theta_y = feedback.base.tool_pose_theta_y
        cartesian_pose.theta_z = feedback.base.tool_pose_theta_z
        if pos_x is not None:
            cartesian_pose.x = pos_x
        if pos_y is not None:
            cartesian_pose.y = pos_y
        if pos_z is not None:
            cartesian_pose.z = max(pos_z,0.03)
        # cartesian_pose.z = pos_z
        if ang_x is not None:
            cartesian_pose.theta_x = ang_x  # (degrees)
        if ang_y is not None:
            cartesian_pose.theta_y = ang_y  # (degrees)
        if ang_z is not None:
            cartesian_pose.theta_z = ang_z  # (degrees)

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        print("Executing action")
        self.base.ExecuteAction(action)
        
        print("Waiting for movement to finish ...")
        finished = e.wait(TIMEOUT_DURATION)
        time.sleep(0)
        self.base.Unsubscribe(notification_handle)

        if finished:
            print("Pickup location reached\n")
        else:
            print("Timeout on action notification wait\n")
        return finished
    
    def move_to_angular(self, goal_angles):  # angular_action
        print("Starting angular action movement to go to Dropout Location ...")
        action = Base_pb2.Action()

        measured_angles = self.base.GetMeasuredJointAngles()
        actuator_count = self.base.GetActuatorCount()
        angles = []
        for joint_id in range(actuator_count.count):
            angles.append(measured_angles.joint_angles[joint_id].value)
        print(angles)

        angles[0] = angles[0] - A0_TURN_DEGREE
        angles[5] = angles[5] - A5_TURN_DEGREE

        for joint_id in range(actuator_count.count):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = joint_id
            joint_angle.value = angles[joint_id]

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(self.check_for_end_or_abort(e), Base_pb2.NotificationOptions())

        print("Executing action")
        self.base.ExecuteAction(action)

        print("Waiting for movement to finish ...")
        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        if finished:
            print("Angular movement completed\n")
        else:
            print("Timeout on action notification wait\n")
        return finished


    def single_joint_move(self, change_joint_id, change_angle):
        action = Base_pb2.Action()

        measured_angles = self.base.GetMeasuredJointAngles()
        actuator_count = self.base.GetActuatorCount()
        angles = []
        for joint_id in range(actuator_count.count):
            angles.append(measured_angles.joint_angles[joint_id].value)

        angles[change_joint_id] = angles[change_joint_id] - change_angle

        for joint_id in range(actuator_count.count):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = joint_id
            joint_angle.value = angles[joint_id]

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(self.check_for_end_or_abort(e), Base_pb2.NotificationOptions())

        print("Executing action")
        self.base.ExecuteAction(action)

        print("Waiting for movement to finish ...")
        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        if finished:
            print("Angular movement completed\n")
        else:
            print("Timeout on action notification wait\n")
        return finished


    def go_to_base3(base, move_z=-0.02):  # twist_command
        print("Starting Twist command movement to go to dropout location ...")
        command = Base_pb2.TwistCommand()
        command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
        command.duration = 0

        twist = command.twist
        twist.linear_x = 0
        twist.linear_y = 0
        twist.linear_z = 0
        twist.angular_x = 5
        twist.angular_y = 0
        twist.angular_z = 0

        print("Sending the twist command for 5 seconds")
        base.SendTwistCommand(command)
        time.sleep(5)

        print("Stopping the robot\n")
        base.Stop()
        time.sleep(1)

        return True
    
    # def get_gripper(self):
    #     gripper_command = Base_pb2.GripperCommand()
    #     finger = gripper_command.gripper.finger.add()
    #     self.gripper = finger.value
    #     return self.gripper
    
    def gripper_control(self, position, rel=False):
        print("Starting Gripper control command ...")
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger.finger_identifier = 1
        if rel:
            position +=self.gripper
        position = np.clip(position , self.gripper_range[0], self.gripper_range[1])
        finger.value = position
        self.gripper = position
        self.base.SendGripperCommand(gripper_command)
        time.sleep(1)
        print("Gripper movement is finished\n")

    def get_image(self):
        src = "rtsp://admin:admin@192.168.1.10/color"
        cap = cv2.VideoCapture(src)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while True:
        # 读取一帧
            ret, frame = cap.read()
            
            # 检查帧是否正确读取
            if not ret or frame is None:
                print("Error: Failed to capture image.")
                break

            # 显示帧
            cv2.imshow('Video', frame)
            cv2.imwrite('E:/code/agent_demo/imgs/captured_image.jpg', frame)
            break
            # 按下 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # 释放资源
        cap.release()
        cv2.destroyAllWindows() 

    def camera_check(self):
        src = "rtsp://admin:admin@192.168.1.10/color"
        cap = cv2.VideoCapture(src)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while True:
        # 读取一帧
            ret, frame = cap.read()
            
            # 检查帧是否正确读取
            if not ret or frame is None:
                print("Error: Failed to capture image.")
                break

            # 显示帧
            cv2.imshow('Video', frame)
            cv2.imwrite('imgs/captured_image.jpg', frame)
            # 按下 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()

    
    def camera_capture_rgb(self):
        self.camera.align_frames()
        rgb = self.camera.get_rgb_frame()
        Image.fromarray(rgb).save('imgs/captured_image.jpg')

    def camera_capture_depth(self):
        self.camera.align_frames()
        depth = self.camera.get_depth_frame()
        print('depth',depth.max(), depth.min())
        Image.fromarray(depth, mode="I;16").save('imgs/captured_image_depth.png')

    
    def move_to_top_view(self):
        print('移动至俯视姿态')
        action = Base_pb2.Action()

        angles = [4.497,2.117, 184.994, 285.147, 358.324, 261.169, 97.048]
        actuator_count = self.base.GetActuatorCount()
        
        for joint_id in range(actuator_count.count):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = joint_id
            joint_angle.value = angles[joint_id]

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(self.check_for_end_or_abort(e), Base_pb2.NotificationOptions())

        print("Executing action")
        self.base.ExecuteAction(action)

        print("Waiting for movement to finish ...")
        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        if finished:
            print("Angular movement completed\n")
        else:
            print("Timeout on action notification wait\n")
        return finished

    def reset(self):
        self.move_to_abs_det(0.3257964789867401, 0.01590120978653431, 0.3860428988933563,177.494,-0.531,90.448)
        self.init_state = np.array([0.3257964789867401, 0.01590120978653431, 0.3860428988933563,177.494,-0.531,90.448])
        self.gripper_control(0.0)

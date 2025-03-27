import threading
import time

import cv2
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import \
    BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20  # (seconds)
GRIPPER_POS_01 = 0.00  # gripper full open
GRIPPER_POS_02 = 0.50  # gripper half close
BASE01_POS_X = 0.08  # (meters)
BASE01_POS_Z = 0.40  # (meters)
BASE01_ANG_X = 90  # (degrees)
A0_TURN_DEGREE = 13  # Actuator ID0 turn degree 13
A5_TURN_DEGREE = 0  # Actuator ID5  turn degree


# Create closure to set an event after an END or an ABORT
def check_for_sequence_end_or_abort(e):
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
def check_for_end_or_abort(e):
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


def go_to_base1(base, base_cyclic, pos_x, pos_y, pos_z, ang_x=0):  # cartesian_action
    print("Starting Cartesian action movement to go to Pickup location ...")
    action = Base_pb2.Action()
    feedback = base_cyclic.RefreshFeedback()
    print(feedback.base.tool_pose_z)

    # cartesian_pose = action.reach_pose.target_pose
    # # cartesian_pose.x = feedback.base.tool_pose_x + pos_x  # (meters)
    # cartesian_pose.x = pos_x
    # cartesian_pose.y = pos_y  # (meters)
    # cartesian_pose.z = pos_z  # (meters)
    # cartesian_pose.theta_x = feedback.base.tool_pose_theta_x + ang_x  # (degrees)
    # cartesian_pose.theta_y = feedback.base.tool_pose_theta_y  # (degrees)
    # cartesian_pose.theta_z = feedback.base.tool_pose_theta_z   # (degrees)

    # e = threading.Event()
    # notification_handle = base.OnNotificationActionTopic(
    #     check_for_end_or_abort(e),
    #     Base_pb2.NotificationOptions()
    # )

    # print("Executing action")
    # base.ExecuteAction(action)

    # print("Waiting for movement to finish ...")
    # finished = e.wait(TIMEOUT_DURATION)
    # time.sleep(0)
    # base.Unsubscribe(notification_handle)

    # if finished:
    #     print("Pickup location reached\n")
    # else:
    #     print("Timeout on action notification wait\n")
    # return finished



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

def move_to_top_view(base):
    print('移动至俯视姿态')
    action = Base_pb2.Action()

    angles = [4.138664722442627, 359.97039794921875, 184.54400634765625, 282.4424743652344, 359.2128601074219, 261.4925537109375, 98.03878784179688]
    actuator_count = base.GetActuatorCount()

    # measured_angles = base.GetMeasuredJointAngles()
    # actuator_count = base.GetActuatorCount()
    # angles = []
    # for joint_id in range(actuator_count.count):
    #     angles.append(measured_angles.joint_angles[joint_id].value)
    # print(angles)

    
    for joint_id in range(actuator_count.count):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = angles[joint_id]

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(check_for_end_or_abort(e), Base_pb2.NotificationOptions())

    print("Executing action")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Angular movement completed\n")
    else:
        print("Timeout on action notification wait\n")
    return finished


def main():
    import argparse
    import os
    import sys

    # Import the utilities helper module
    import utilities

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser()
    args = utilities.parseConnectionArguments(parser)

    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        go_to_base1(base, base_cyclic, 0, 0, 0, 2)
        # move_to_top_view(base)


if __name__ == "__main__":
    main()
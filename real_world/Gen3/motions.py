import threading
import time

import cv2
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import \
    BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2

import robot

NOD_ANGLE = 10  # (degrees)
NOD_CNT = 3
TIMEOUT_DURATION = 20  # (seconds)

motion_dict = {
    "salute": [
        282.1579895019531,
        301.9700622558594,
        166.41046142578125,
        240.829345703125,
        354.28985595703125,
        358.80615234375,
        5.997819900512695,
    ],
    "shake_head": [
        343.0167541503906,
        351.4176330566406,
        184.4325408935547,
        257.31549072265625,
        265.8954162597656,
        359.0321350097656,
        184.77743530273438,
    ],
    "nod": [
        344.94622802734375,
        349.2922058105469,
        181.08737182617188,
        259.10504150390625,
        3.949371814727783,
        1.3257977962493896,
        84.78924560546875,
    ],
}


def nod(base, base_syclic, nod_angle=NOD_ANGLE, nod_cnt=NOD_CNT):
    set_target_angles(base, base_syclic, motion_dict["nod"])
    for _ in range(nod_cnt):
        change_target_angles(base, base_syclic, [0, 0, 0, 0, 0, -40, 0])
        change_target_angles(base, base_syclic, [0, 0, 0, 0, 0, 40, 0])

    print("Nodding {nod_cnt} times completed\n")


def shake_head(base, base_syclic, shake_angle=90, shake_cnt=2):
    set_target_angles(base, base_syclic, motion_dict["shake_head"])
    change_target_angles(base, base_syclic, [0, 0, 0, 0, 0, 25, 0])
    for _ in range(shake_cnt):
        change_target_angles(base, base_syclic, [0, 0, 0, 0, 0, -50, 0])
        change_target_angles(base, base_syclic, [0, 0, 0, 0, 0, 50, 0])
    
    change_target_angles(base, base_syclic, [0, 0, 0, 0, 0, -25, 0])
    set_target_angles(base, base_syclic, motion_dict["shake_head"])
    print("Shaking head {shake_cnt} times completed\n")


# start angles: [0, 0, 0, 0, 0, 0]
def set_target_angles(base, base_syclic, target_angles=[0, 0, 0, 0, 0, 0]):

    print("Starting angular action movement to go to Dropout Location ...")
    action = Base_pb2.Action()

    measured_angles = base.GetMeasuredJointAngles()
    actuator_count = base.GetActuatorCount()

    for joint_id in range(actuator_count.count):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = target_angles[joint_id]

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        robot.check_for_end_or_abort(e), Base_pb2.NotificationOptions()
    )

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


# add rotations to the current joint angles
def change_target_angles(base, base_syclic, delta_angles=[0, 0, 0, 0, 0, 0]):

    print("Starting angular action movement to go to Dropout Location ...")
    action = Base_pb2.Action()

    measured_angles = base.GetMeasuredJointAngles()
    actuator_count = base.GetActuatorCount()
    angles = []
    for joint_id in range(actuator_count.count):
        angles.append(
            measured_angles.joint_angles[joint_id].value + delta_angles[joint_id]
        )
    print(angles)

    for joint_id in range(actuator_count.count):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = angles[joint_id]

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        robot.check_for_end_or_abort(e), Base_pb2.NotificationOptions()
    )

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


def wave(base, base_syclic, wave_angle=90, wave_cnt=2, start_angles=[0, 0, 0, 0, 0, 0]):
    set_target_angles(base, base_syclic, start_angles, wave_angle, wave_cnt)
    time.sleep(0.5)
    finished = False
    for _ in range(wave_cnt):

        finished = robot.go_to_base1(base, base_syclic, 0, 0, -wave_angle)
        if not finished:
            break
        finished = robot.go_to_base1(base, base_syclic, 0, 0, 2 * wave_angle)
        if not finished:
            break
        time.sleep(0.2)

    if finished:
        print("Waving {{wave_cnt}} times completed\n")
    else:
        print("Timeout on action notification wait\n")


def salute(
    base, base_syclic, start_angles=[0, 0, 0, 0, 0, 0, 0]
):
    set_target_angles(base, base_syclic, start_angles)

    #  TODO: set target angles for salute(the ending pose)
    set_target_angles(base, base_syclic, motion_dict["salute"])
    time.sleep(0.5)
    return


def main():
    # Import the utilities helper module
    import utilities

    args = utilities.parseConnectionArguments()
    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        # Observe
        # go_to_home(base)
        # go_to_base1(base, base_cyclic, 0.02, 0, BASE01_ANG_X, -0.1)
        # time.sleep(2)
        # gripper_control(base, GRIPPER_POS_01)
        # go_to_home(base)
        # go_to_base1(base, base_cyclic, pos_x=0, pos_z=-0.2, ang_x=0, pos_y=0)

        # print the current joint angles
        measured_angles = base.GetMeasuredJointAngles()
        print("Measured joint angles:" + str(measured_angles.joint_angles))
        # nod(base, base_cyclic)
        shake_head(base, base_cyclic)
        # salute(base, base_cyclic)
        # while True:
        #     # if the user presses 'q', exit the loop
        #     if cv2.waitKey(1) & 0xFF == ord("q"):
        #         break
        #     time.sleep(1)


if __name__ == "__main__":
    main()

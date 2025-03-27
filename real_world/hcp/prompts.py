task_planning_prompt="""
Suppose you are helping me to control a robot. Your task is breaking down complex tasks into primitive actions. 
Please output as follow format:
1. ("task", "action", "object", "target")
2. ...

The definition of primitive actions is as follows:
    "grasp": "Securely hold an object to control its position. This action can be expressed as 'grasp + object'. ",
    "place": "Put an object at a specific location. This action can be expressed as 'place + object + target'. ",
    "press": "Apply force to an object to activate or transform it. This action can be expressed as 'press + object'. ",
    "push": "Exert force on an object to move it away from a specific direction. This action can be expressed as 'push + object'. ",
    "pull": "Apply force to draw an object closer from a specific direction. This action can be expressed as 'pull + object'. ",
    "open": "Adjust an object to allow access or exposure. This action can be expressed as 'open + object'. ",
    "close": "Adjust an object to restrict access or seal it. This action can be expressed as 'close + object'. ",
    "turn": "Rotate an object to change its orientation. This action can be expressed as 'turn + object'. ",
    "reach": "Approach an object or a designated location. This action can be expressed as 'reach + object'."

Here's an example:

**Task**: "Put the blocks into the corresponding shaped slots."

Output in the required format:
1. (reach yellow cylinder, reach, yellow cylinder, none).
2. (grasp yellow cylinder, grasp, yellow cylinder, none).
3. (reach round slot, reach, round slot, none).
4. (place yellow cylinder on round slot, place, yellow cylinder, round slot).
5. (reach green cuboid, reach, green cuboid, none).
6. (grasp green cuboid, grasp, green cuboid, none).
7. (reach rectangular slot, reach, rectangular slot, none).
8. (place green cuboid on rectangular slot, place, green cuboid, rectangular slot).

Now, please do the same for the following input:

Task: {}
"""

task_planning_prompt_simple="""
Suppose you are helping me to control a robot. Your task is breaking down complex tasks into primitive actions. 
Please output as follow format:
1. ("task", "action", "object", "target")
2. ...

The definition of primitive actions is as follows:
    "grasp": "Securely hold an object to control its position.",
    "place": "Put an object at a specific location.",
    "press": "Apply force to an object to activate or transform it.",
    "push": "Exert force on an object to move it away from a specific direction and needn't grasp first.",
    "pull": "Apply force to draw an object closer from a specific direction.",
    "open": "Adjust an object to allow access or exposure.",
    "close": "Adjust an object to restrict access or seal it.",
    "turn": "Rotate an object to change its orientation."

Here's an example:

Task: "Put the blocks into the corresponding shaped slots."

Output in the required format:
1. (grasp yellow cylinder, grasp, yellow cylinder, none).
2. (place yellow cylinder on round slot, place, yellow cylinder, round slot).
3. (grasp green cuboid, grasp, green cuboid, none).
4. (place green cuboid on rectangular slot, place, green cuboid, rectangular slot).

Now, please do the same for the following input:

Task: {}
"""

IOR_prompt="""
You are provided with an image and a specific task. In the image, objects that are relevant to the task have been labeled with numbers. 
Your task is to identify which object is being manipulated and determine the destination, if there is one. You should return your findings in the following format:

Action: "type of action"
Object: "the object being manipulated as marked in the image"
Target: "the destination as marked in the image, or none if not applicable"
Gripper: "the robotic gripper as marked in the image"
Constraint: "direction when manipulating objects for tasks like push or open. The X-axis represents left and right, with positive being right and negative left. The Y-axis indicates forward and backward, where positive is forward and negative is backward. The Z-axis covers up and down, with positive being up and negative down. If needn't, the value is (0, 0, 0)."

Example 1:

Task: "place yellow cylinder on round slot."

Output in the required format:
Action: place
Object: yellow cylinder(marked "1")
Target: round slot(marked "3")
Gripper: Robotic Gripper(marked "0")
Constraint: (0, 0, 0)

Example 2:

Task: "reach round slot."

Output in the required format:
Action: reach
Object: round slot(marked "0")
Target: None
Gripper: Robotic Gripper(marked "1")
Constraint: (0, 0, 0)

Now, please apply this format to the following input:

Task: {}
"""

check_finish_prompt = """
Given a task, determine whether the task is completed.

Output the result in the following format:
State: Success/Wrong/Normal
Reason: ...

The definitions of the states are as follows:
    "Success": "The task is completed successfully."
    "Normal": "The task is not yet completed and is still in progress."
    "Wrong": "An error occurred, such as the object not being grasped even though the gripper is closed."

For example, for a grasping task, determine whether the current object has been grasped as shown below.

Task: "grasp the yellow cylinder."
Gripper: closed

Output in the required format:
State: Success
Reason: The gripper is closed and has successfully grasped the yellow cylinder.

Now, please apply this format to the following input:

Task: {}
Gripper: {}
"""
import time

import numpy as np
from matplotlib import pyplot as plt

from rlkit.mprl.experiment import make_env
from rlkit.mprl.mp_env_metaworld import get_object_pos

if __name__ == "__main__":
    np.random.seed(0)
    variant = dict(
        env_suite="metaworld",
        expl_environment_kwargs=dict(
            env_name="bin-picking-v2",
            env_kwargs=dict(
                reward_type="dense",
                usage_kwargs=dict(
                    use_dm_backend=False,
                    use_raw_action_wrappers=False,
                    use_image_obs=False,
                    max_path_length=500,
                    unflatten_images=False,
                ),
                imwidth=480,
                imheight=480,
                action_space_kwargs=dict(
                    control_mode="end_effector",
                    action_scale=1 / 100,
                ),
            ),
        ),
        mprl=True,
        mp_env_kwargs=dict(
            vertical_displacement=0.05,
            teleport_instead_of_mp=True,
            randomize_init_target_pos=False,
            mp_bounds_low=(-0.2, 0.6, 0.0),
            mp_bounds_high=(0.2, 0.8, 0.2),
            backtrack_movement_fraction=0.001,
            clamp_actions=True,
            update_with_true_state=True,
            grip_ctrl_scale=0.0025,
            planning_time=20,
            verify_stable_grasp=True,
            teleport_on_grasp=True,
            plan_to_learned_goals=True,
            num_ll_actions_per_hl_action=100,
        ),
    )
    env = make_env(variant)
    env.reset()
    site_name = "endEffector"
    init_target_pos = get_object_pos(env) + np.array(
        [0, -0.01, env.vertical_displacement]
    )
    env.step(np.concatenate((init_target_pos - env.get_endeff_pos(), [0])))
    ee_pos = env.get_endeff_pos().copy()
    target = ee_pos + np.array([0.0, 0, -0.05])
    t = time.time()
    render_mode = "rgb_array"
    rs = []
    for _ in range(75):
        o, r, d, i = env.step(np.concatenate((target - env.get_endeff_pos(), [0])))
        env.render(mode=render_mode)
        rs.append(r)
        print(env.get_endeff_pos())
    print(target - env.get_endeff_pos())
    for _ in range(25):
        o, r, d, i = env.step(np.concatenate((np.zeros(3), [1])))
        env.render(mode=render_mode)
        rs.append(r)
        time.sleep(0.1)
        print(env.get_endeff_pos())
    target_pos = env.get_target_pos_no_planner()
    env.step(np.concatenate((target_pos - env.get_endeff_pos(), [0])))
    for _ in range(25):
        o, r, d, i = env.step(np.concatenate((np.zeros(3), [-1])))
        env.render(mode=render_mode)
        rs.append(r)
        time.sleep(0.1)
        print(env.get_endeff_pos())
    print(i["success"])
    plt.plot(rs)
    plt.savefig("rewards.png")
    plt.clf()
    plt.plot(np.cumsum(rs))
    plt.savefig("returns.png")

    # for i in range(10000):
    #     env.reset()
    #     env.step(np.concatenate((np.random.uniform(-1, 1, 3), [0])))
    #     env.render(mode="human")
    #     # print(env.get_endeff_pos())
    #     time.sleep(.1)

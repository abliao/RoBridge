import copy
import traceback
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

create_rollout_function = partial


def multitask_rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    observation_key=None,
    desired_goal_key=None,
    get_action_kwargs=None,
    return_dict_obs=False,
    full_o_postprocess_func=None,
):
    if full_o_postprocess_func:

        def wrapped_fun(env, agent, o):
            full_o_postprocess_func(env, agent, observation_key, o)

    else:
        wrapped_fun = None

    def obs_processor(o):
        return np.hstack((o[observation_key], o[desired_goal_key]))

    paths = rollout(
        env,
        agent,
        max_path_length=max_path_length,
        render=render,
        render_kwargs=render_kwargs,
        get_action_kwargs=get_action_kwargs,
        preprocess_obs_for_policy_fn=obs_processor,
        full_o_postprocess_func=wrapped_fun,
    )
    if not return_dict_obs:
        paths["observations"] = paths["observations"][observation_key]
    return paths


def contextual_rollout(
    env,
    agent,
    observation_key=None,
    context_keys_for_policy=None,
    obs_processor=None,
    **kwargs,
):
    if context_keys_for_policy is None:
        context_keys_for_policy = ["context"]

    if not obs_processor:

        def obs_processor(o):
            combined_obs = [o[observation_key]]
            for k in context_keys_for_policy:
                combined_obs.append(o[k])
            return np.concatenate(combined_obs, axis=0)

    paths = rollout(env, agent, preprocess_obs_for_policy_fn=obs_processor, **kwargs)
    return paths


def rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    preprocess_obs_for_policy_fn=None,
    get_action_kwargs=None,
    return_dict_obs=False,
    full_o_postprocess_func=None,
    reset_callback=None,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x
    raw_obs = []
    raw_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if reset_callback:
        reset_callback(env, agent, o)
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        raw_obs.append(o)
        o_for_agent = preprocess_obs_for_policy_fn(o)
        a, agent_info = agent.get_action(o_for_agent, **get_action_kwargs)

        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        next_o, r, d, env_info = env.step(copy.deepcopy(a))
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        raw_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = raw_obs
        next_observations = raw_next_obs
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        full_observations=raw_obs,
        full_next_observations=raw_obs,
    )


@torch.no_grad()
def vec_rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    preprocess_obs_for_policy_fn=None,
    get_action_kwargs=None,
    return_dict_obs=False,
    full_o_postprocess_func=None,
    reset_callback=None,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x
    raw_obs = []
    raw_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    bad_masks = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if reset_callback:
        reset_callback(env, agent, o)
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        raw_obs.append(o)
        o_for_agent = preprocess_obs_for_policy_fn(o)
        a, agent_info = agent.get_action(o_for_agent, **get_action_kwargs)

        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        next_o, r, d, env_info = env.step(copy.deepcopy(a))
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        raw_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        bad_masks.append(env_info["bad_mask"])
        path_length += 1
        o = next_o

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = raw_obs
        next_observations = raw_next_obs
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    terminals = np.array(terminals)
    bad_masks = np.array(bad_masks)
    observations = [observations[:, i] for i in range(env.num_envs)]
    next_observations = [next_observations[:, i] for i in range(env.num_envs)]
    actions = [actions[:, i] for i in range(env.num_envs)]
    rewards = [rewards[:, i] for i in range(env.num_envs)]
    terminals = [terminals[:, i] for i in range(env.num_envs)]
    bad_masks = [bad_masks[:, i][:, 0, 0] for i in range(env.num_envs)]
    env_infos = [
        [
            {key: env_infos[j][key][i] for key in env_infos[j]}
            for j in range(max_path_length)
        ]
        for i in range(env.num_envs)
    ]  # should be a list of list of dicts (length num_envs) (length of path)

    # convert all arrays to masked arrays
    for i in range(env.num_envs):
        mask = bad_masks[i]
        observations[i] = np.ma.array(
            observations[i],
            mask=mask.reshape(-1, 1).repeat(observations[i].shape[1], axis=1),
        )
        next_observations[i] = np.ma.array(
            next_observations[i],
            mask=mask.reshape(-1, 1).repeat(next_observations[i].shape[1], axis=1),
        )
        actions[i] = np.ma.array(
            actions[i], mask=mask.reshape(-1, 1).repeat(actions[i].shape[1], axis=1)
        )
        rewards[i] = np.ma.array(rewards[i], mask=mask)
        terminals[i] = np.ma.array(terminals[i], mask=mask)
        # NOTE: adding masks to env_infos does not work, because masking a scalar just makes it a null value
    paths = []
    for i in range(env.num_envs):
        paths.append(
            dict(
                observations=observations[i],
                actions=actions[i],
                rewards=rewards[i],
                next_observations=next_observations[i],
                terminals=terminals[i],
                agent_infos=agent_infos,
                env_infos=env_infos[i],
                bad_masks=bad_masks[i],
            )
        )
    return paths, paths


@torch.no_grad()
def rollout_modular(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    preprocess_obs_for_policy_fn=None,
    get_action_kwargs=None,
    return_dict_obs=False,
    full_o_postprocess_func=None,
    reset_callback=None,
):
    from torchvision.utils import save_image

    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x

    raw_obs = []
    raw_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    bad_masks = []

    path_length = 0
    agent.reset()
    o = env.reset()
    if render:
        env.render(**render_kwargs)
    episode_breaks = []
    terminate_each_stage = agent.terminate_each_stage
    terminate_planner_actions = agent.terminate_planner_actions
    planner_indices = []
    while path_length < max_path_length:
        o_for_agent = preprocess_obs_for_policy_fn(o)
        a, agent_info = agent.get_action(o_for_agent, **get_action_kwargs)
        # use_planner is updated after get_action call
        use_planner = agent.current_policy_str == "policy1"
        if use_planner and len(observations) > 0:
            episode_breaks.append(path_length)
        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        next_o, r, d, env_info = env.step(copy.deepcopy(a))

        if use_planner:
            planner_indices.append(path_length)
            if terminate_each_stage and len(observations) > 0:
                # assumption is that the previous stage was control policy execution if len(obs) > 0 and use_planner is True
                terminals[-1] = np.array([True] * env.num_envs)

        raw_obs.append(o)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        next_observations.append(next_o)
        raw_next_obs.append(next_o)
        bad_masks.append(env_info["bad_mask"])

        if use_planner:
            if terminate_each_stage or terminate_planner_actions:
                terminals[-1] = np.array([True] * env.num_envs)
        o = next_o
        path_length += 1
    if terminate_each_stage and len(observations) > 0:
        terminals[-1] = np.array([True] * env.num_envs)
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = raw_obs
        next_observations = raw_next_obs
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    bad_masks = np.array(bad_masks)

    terminals = np.array(terminals)
    observations = [observations[:, i] for i in range(env.num_envs)]
    next_observations = [next_observations[:, i] for i in range(env.num_envs)]
    actions = [actions[:, i] for i in range(env.num_envs)]
    rewards = [rewards[:, i] for i in range(env.num_envs)]
    terminals = [terminals[:, i] for i in range(env.num_envs)]
    bad_masks = [bad_masks[:, i] for i in range(env.num_envs)]
    env_infos = [
        [
            {key: env_infos[j][key][i] for key in env_infos[j]}
            for j in range(len(env_infos))
        ]
        for i in range(env.num_envs)
    ]  # should be a list of list of dicts (length num_envs) (length of path)

    paths = []
    merged_paths = []
    only_keep_trajs_after_grasp_success = (
        agent.only_keep_trajs_after_grasp_success  # do NOT use episode breaks if we are only keeping trajs after grasp success
    )
    only_keep_trajs_stagewise = agent.only_keep_trajs_stagewise

    # convert all arrays to masked arrays
    for i in range(env.num_envs):
        mask = bad_masks[i]
        observations[i] = np.ma.array(
            observations[i],
            mask=mask.reshape(-1, 1).repeat(observations[i].shape[1], axis=1),
        )
        next_observations[i] = np.ma.array(
            next_observations[i],
            mask=mask.reshape(-1, 1).repeat(next_observations[i].shape[1], axis=1),
        )
        actions[i] = np.ma.array(
            actions[i], mask=mask.reshape(-1, 1).repeat(actions[i].shape[1], axis=1)
        )
        rewards[i] = np.ma.array(rewards[i], mask=mask)
        terminals[i] = np.ma.array(terminals[i], mask=mask)
        # NOTE: adding masks to env_infos does not work, because masking a scalar just makes it a null value

    # control indices should be all indices in range(0, max_path_length) that are not in planner_indices
    control_indices = list(set(range(max_path_length)) - set(planner_indices))
    for i in range(env.num_envs):
        merged_paths.append(
            dict(
                type="merged",
                observations=observations[i],
                actions=actions[i],
                rewards=rewards[i],
                next_observations=next_observations[i],
                terminals=terminals[i],
                agent_infos=agent_infos,
                env_infos=env_infos[i],
                bad_masks=bad_masks[i],
            )
        )
        if agent.use_episode_breaks:
            prev_episode_break = 0
            for idx, episode_break in enumerate(episode_breaks):
                paths.append(
                    dict(
                        type="control",
                        observations=observations[i][prev_episode_break:episode_break],
                        actions=actions[i][prev_episode_break:episode_break],
                        rewards=rewards[i][prev_episode_break:episode_break],
                        next_observations=next_observations[i][
                            prev_episode_break:episode_break
                        ],
                        terminals=terminals[i][prev_episode_break:episode_break],
                        agent_infos=agent_infos,
                        env_infos=env_infos[i][prev_episode_break:episode_break],
                        bad_masks=bad_masks[i][prev_episode_break:episode_break],
                    )
                )
                paths.append(
                    dict(
                        type="planner",
                        observations=planner_observations[i][idx : idx + 1],
                        next_observations=planner_next_observations[i][idx : idx + 1],
                        actions=planner_actions[i][idx : idx + 1],
                        rewards=planner_rewards[i][idx : idx + 1],
                        terminals=planner_terminals[i][idx : idx + 1],
                        agent_infos=planner_agent_infos,
                        env_infos=planner_env_infos[i][idx : idx + 1],
                        bad_masks=planner_bad_masks[i][idx : idx + 1],
                    )
                )
                prev_episode_break = episode_break
            paths.append(
                dict(
                    type="control",
                    observations=observations[i][prev_episode_break:],
                    actions=actions[i][prev_episode_break:],
                    rewards=rewards[i][prev_episode_break:],
                    next_observations=next_observations[i][prev_episode_break:],
                    terminals=terminals[i][prev_episode_break:],
                    agent_infos=agent_infos,
                    env_infos=env_infos[i][prev_episode_break:],
                    bad_masks=bad_masks[i][prev_episode_break:],
                )
            )
            paths.append(
                dict(
                    type="planner",
                    observations=planner_observations[i][-1:],
                    next_observations=planner_next_observations[i][-1:],
                    actions=planner_actions[i][-1:],
                    rewards=planner_rewards[i][-1:],
                    terminals=planner_terminals[i][-1:],
                    agent_infos=planner_agent_infos,
                    env_infos=planner_env_infos[i][-1:],
                    bad_masks=planner_bad_masks[i][-1:],
                )
            )
        else:
            if only_keep_trajs_after_grasp_success and not env_infos[i][25]["grasped"]:
                paths.append(
                    dict(
                        type="control",
                        observations=observations[i][: episode_breaks[0]],
                        actions=actions[i][: episode_breaks[0]],
                        rewards=rewards[i][: episode_breaks[0]],
                        next_observations=next_observations[i][: episode_breaks[0]],
                        terminals=terminals[i][: episode_breaks[0]],
                        agent_infos=agent_infos,
                        env_infos=env_infos[i][: episode_breaks[0]],
                        bad_masks=bad_masks[i][: episode_breaks[0]],
                    )
                )
                paths.append(
                    dict(
                        type="planner",
                        observations=planner_observations[i][:1],
                        next_observations=planner_next_observations[i][:1],
                        actions=planner_actions[i][:1],
                        rewards=planner_rewards[i][:1],
                        terminals=planner_terminals[i][:1],
                        agent_infos=planner_agent_infos,
                        env_infos=planner_env_infos[i][:1],
                        bad_masks=planner_bad_masks[i][:1],
                    )
                )
            elif only_keep_trajs_stagewise:
                if planner_rewards[i][0] > 0.06:
                    if rewards[i][: episode_breaks[0]][-1] > 0.3:
                        if planner_rewards[i][1] > 0.6:
                            # add full traj
                            paths.append(
                                dict(
                                    type="control",
                                    observations=observations[i],
                                    actions=actions[i],
                                    rewards=rewards[i],
                                    next_observations=next_observations[i],
                                    terminals=terminals[i],
                                    agent_infos=agent_infos,
                                    env_infos=env_infos[i],
                                    bad_masks=bad_masks[i],
                                )
                            )
                            paths.append(
                                dict(
                                    type="planner",
                                    observations=planner_observations[i],
                                    next_observations=planner_next_observations[i],
                                    actions=planner_actions[i],
                                    rewards=planner_rewards[i],
                                    terminals=planner_terminals[i],
                                    agent_infos=planner_agent_infos,
                                    env_infos=planner_env_infos[i],
                                    bad_masks=planner_bad_masks[i],
                                )
                            )
                        else:
                            # only add second planner action, not second control traj
                            paths.append(
                                dict(
                                    type="control",
                                    observations=observations[i][: episode_breaks[0]],
                                    actions=actions[i][: episode_breaks[0]],
                                    rewards=rewards[i][: episode_breaks[0]],
                                    next_observations=next_observations[i][
                                        : episode_breaks[0]
                                    ],
                                    terminals=terminals[i][: episode_breaks[0]],
                                    agent_infos=agent_infos,
                                    env_infos=env_infos[i][: episode_breaks[0]],
                                    bad_masks=bad_masks[i][: episode_breaks[0]],
                                )
                            )
                            paths.append(
                                dict(
                                    type="planner",
                                    observations=planner_observations[i],
                                    next_observations=planner_next_observations[i],
                                    actions=planner_actions[i],
                                    rewards=planner_rewards[i],
                                    terminals=planner_terminals[i],
                                    agent_infos=planner_agent_infos,
                                    env_infos=planner_env_infos[i],
                                    bad_masks=planner_bad_masks[i],
                                )
                            )
                    else:
                        # add first planner action, first control traj
                        paths.append(
                            dict(
                                type="control",
                                observations=observations[i][: episode_breaks[0]],
                                actions=actions[i][: episode_breaks[0]],
                                rewards=rewards[i][: episode_breaks[0]],
                                next_observations=next_observations[i][
                                    : episode_breaks[0]
                                ],
                                terminals=terminals[i][: episode_breaks[0]],
                                agent_infos=agent_infos,
                                env_infos=env_infos[i][: episode_breaks[0]],
                                bad_masks=bad_masks[i][: episode_breaks[0]],
                            )
                        )
                        paths.append(
                            dict(
                                type="planner",
                                observations=planner_observations[i][:1],
                                next_observations=planner_next_observations[i][:1],
                                actions=planner_actions[i][:1],
                                rewards=planner_rewards[i][:1],
                                terminals=planner_terminals[i][:1],
                                agent_infos=planner_agent_infos,
                                env_infos=planner_env_infos[i][:1],
                                bad_masks=planner_bad_masks[i][:1],
                            )
                        )
                else:
                    # add first planner action only
                    paths.append(
                        dict(
                            type="planner",
                            observations=planner_observations[i][:1],
                            next_observations=planner_next_observations[i][:1],
                            actions=planner_actions[i][:1],
                            rewards=planner_rewards[i][:1],
                            terminals=planner_terminals[i][:1],
                            agent_infos=planner_agent_infos,
                            env_infos=planner_env_infos[i][:1],
                            bad_masks=planner_bad_masks[i][:1],
                        )
                    )
            else:
                paths.append(
                    dict(
                        type="control",
                        observations=observations[i][control_indices],
                        actions=actions[i][control_indices],
                        rewards=rewards[i][control_indices],
                        next_observations=next_observations[i][control_indices],
                        terminals=terminals[i][control_indices],
                        agent_infos=agent_infos,
                        env_infos=[env_infos[i][idx] for idx in control_indices],
                    )
                )
                paths.append(
                    dict(
                        type="planner",
                        observations=observations[i][planner_indices],
                        actions=actions[i][planner_indices],
                        rewards=rewards[i][planner_indices],
                        next_observations=next_observations[i][planner_indices],
                        terminals=terminals[i][planner_indices],
                        agent_infos=agent_infos,
                        env_infos=[env_infos[i][idx] for idx in planner_indices],
                    )
                )
    return paths, merged_paths


@torch.no_grad()
def rollout_multi_stage_modular(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    preprocess_obs_for_policy_fn=None,
    get_action_kwargs=None,
    return_dict_obs=False,
    full_o_postprocess_func=None,
    reset_callback=None,
):
    from torchvision.utils import save_image

    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x

    raw_obs = []
    raw_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    bad_masks = []

    path_length = 0
    agent.reset()
    o = env.reset()
    if render:
        env.render(**render_kwargs)
    episode_breaks = []
    terminate_each_stage = agent.active_policy.terminate_each_stage
    terminate_planner_actions = agent.active_policy.terminate_planner_actions
    planner_indices = []
    control_indices = []
    stage_indices = [0]
    while path_length < max_path_length:
        if agent.active_policy.take_policy1_step and len(observations) > 0:
            episode_breaks.append(path_length)
            stage_indices.append(path_length)
        o_for_agent = preprocess_obs_for_policy_fn(o)
        a, agent_info = agent.get_action(o_for_agent, **get_action_kwargs)
        # use_planner is updated after get_action call
        use_planner = agent.active_policy.current_policy_str == "policy1"
        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        next_o, r, d, env_info = env.step(copy.deepcopy(a))

        if use_planner:
            planner_indices.append([])
            planner_indices[agent.stage].append(path_length)
            if terminate_each_stage and len(observations) > 0:
                # assumption is that the previous stage was control policy execution if len(obs) > 0 and use_planner is True
                terminals[-1] = np.array([True] * env.num_envs)

        raw_obs.append(o)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        next_observations.append(next_o)
        raw_next_obs.append(next_o)
        bad_masks.append(env_info["bad_mask"])

        if use_planner:
            if terminate_each_stage or terminate_planner_actions:
                terminals[-1] = np.array([True] * env.num_envs)
        o = next_o
        path_length += 1
    if terminate_each_stage and len(observations) > 0:
        terminals[-1] = np.array([True] * env.num_envs)
    stage_indices.append(path_length)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = raw_obs
        next_observations = raw_next_obs
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    bad_masks = np.array(bad_masks)

    terminals = np.array(terminals)
    observations = [observations[:, i] for i in range(env.num_envs)]
    next_observations = [next_observations[:, i] for i in range(env.num_envs)]
    actions = [actions[:, i] for i in range(env.num_envs)]
    rewards = [rewards[:, i] for i in range(env.num_envs)]
    terminals = [terminals[:, i] for i in range(env.num_envs)]
    bad_masks = [bad_masks[:, i] for i in range(env.num_envs)]
    env_infos = [
        [
            {key: env_infos[j][key][i] for key in env_infos[j]}
            for j in range(len(env_infos))
        ]
        for i in range(env.num_envs)
    ]  # should be a list of list of dicts (length num_envs) (length of path)

    paths = []
    merged_paths = []

    # convert all arrays to masked arrays
    for i in range(env.num_envs):
        mask = bad_masks[i]
        observations[i] = np.ma.array(
            observations[i],
            mask=mask.reshape(-1, 1).repeat(observations[i].shape[1], axis=1),
        )
        next_observations[i] = np.ma.array(
            next_observations[i],
            mask=mask.reshape(-1, 1).repeat(next_observations[i].shape[1], axis=1),
        )
        actions[i] = np.ma.array(
            actions[i], mask=mask.reshape(-1, 1).repeat(actions[i].shape[1], axis=1)
        )
        rewards[i] = np.ma.array(rewards[i], mask=mask)
        terminals[i] = np.ma.array(terminals[i], mask=mask)
        # NOTE: adding masks to env_infos does not work, because masking a scalar just makes it a null value
    num_stages = len(stage_indices) - 1
    # control indices should be all indices in range(0, max_path_length) that are not in planner_indices
    for stage in range(1, num_stages + 1):
        control_indices.append([])
        try:
            control_indices[stage - 1] = list(
                set(range(stage_indices[stage - 1], stage_indices[stage]))
                - set(planner_indices[stage - 1])
            )
        except:
            print(traceback.format_exc())
    filter_stage1_based_on_stage0_grasp = (
        agent.active_policy.filter_stage1_based_on_stage0_grasp
    )
    for i in range(env.num_envs):
        merged_paths.append(
            dict(
                type="merged",
                observations=observations[i],
                actions=actions[i],
                rewards=rewards[i],
                next_observations=next_observations[i],
                terminals=terminals[i],
                agent_infos=agent_infos,
                env_infos=env_infos[i],
                bad_masks=bad_masks[i],
            )
        )
        for stage in range(0, num_stages):
            if (
                stage == 1
                and filter_stage1_based_on_stage0_grasp
                and not env_infos[i][control_indices[stage - 1][-1]]["grasped"]
            ):
                continue
            if len(control_indices[stage]) > 0:
                paths.append(
                    dict(
                        type=f"control_{stage}",
                        observations=observations[i][control_indices[stage]],
                        actions=actions[i][control_indices[stage]],
                        rewards=rewards[i][control_indices[stage]],
                        next_observations=next_observations[i][control_indices[stage]],
                        terminals=terminals[i][control_indices[stage]],
                        agent_infos=agent_infos,
                        env_infos=[env_infos[i][idx] for idx in control_indices[stage]],
                    )
                )
            paths.append(
                dict(
                    type=f"planner_{stage}",
                    observations=observations[i][planner_indices[stage]],
                    actions=actions[i][planner_indices[stage]],
                    rewards=rewards[i][planner_indices[stage]],
                    next_observations=next_observations[i][planner_indices[stage]],
                    terminals=terminals[i][planner_indices[stage]],
                    agent_infos=agent_infos,
                    env_infos=[env_infos[i][idx] for idx in planner_indices[stage]],
                )
            )
    return paths, merged_paths


def deprecated_rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack((observations[1:, :], np.expand_dims(next_o, 0)))
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )

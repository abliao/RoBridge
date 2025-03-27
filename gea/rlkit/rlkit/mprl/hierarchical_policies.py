from rlkit.policies.base import Policy


class StepBasedSwitchingPolicy(Policy):
    """
    A policy that switches between two underlying policies based on the number of steps taken.
    """

    def __init__(
        self,
        policy1,
        policy2,
        policy2_steps_per_policy1_step,
        use_episode_breaks=False,
        only_keep_trajs_after_grasp_success=False,
        only_keep_trajs_stagewise=False,
        terminate_each_stage=False,
        filter_stage1_based_on_stage0_grasp=False,
        terminate_planner_actions=True,
    ):
        """
        Initializes a new instance of the StepBasedSwitchingPolicy class.

        Args:
            policy1 (Policy): The first underlying policy.
            policy2 (Policy): The second underlying policy.
            policy2_path_length (int): The number of steps to take before switching to policy1.
            use_episode_breaks (bool): Whether or not to use episode breaks when switching policies.
            only_keep_trajs_after_grasp_success (bool): Whether or not to only keep trajectories
                after a grasp success.
            only_keep_trajs_stagewise (bool): Whether or not to only keep stages of trajectories.
        """
        self.policy1 = policy1
        self.policy2 = policy2
        self.policy2_steps_per_policy1_step = policy2_steps_per_policy1_step
        self.use_episode_breaks = use_episode_breaks
        self.only_keep_trajs_after_grasp_success = only_keep_trajs_after_grasp_success
        self.only_keep_trajs_stagewise = only_keep_trajs_stagewise
        self.terminate_each_stage = terminate_each_stage
        self.filter_stage1_based_on_stage0_grasp = filter_stage1_based_on_stage0_grasp
        self.terminate_planner_actions = terminate_planner_actions
        self.reset()

    def get_action(self, observation):
        """
        Gets an action from the currently active underlying policy.

        Args:
            observation: An observation of the environment.

        Returns:
            An action to take in the environment.
        """
        if self.take_policy1_step:
            self.current_policy = self.policy1
            self.current_policy_str = "policy1"
            self.take_policy1_step = False
        else:
            self.current_policy = self.policy2
            self.current_policy_str = "policy2"
            self.current_policy2_steps += 1
            if self.current_policy2_steps == self.policy2_steps_per_policy1_step:
                self.current_policy2_steps = 0
                self.take_policy1_step = True
        action = self.current_policy.get_action(observation)
        self.num_steps += 1
        return action

    def reset(self):
        """
        Resets the underlying policies and sets the number of steps and current policy to their initial values.
        """
        self.policy1.reset()
        self.policy2.reset()
        self.num_steps = 0
        self.current_policy = self.policy1
        self.current_policy_str = "policy1"
        self.take_policy1_step = True
        self.current_policy2_steps = 0


class MultiStageStepBasedSwitchingPolicy(Policy):
    def __init__(self, policies):
        self.policies = policies
        self.active_policy = self.policies[0]
        self.active_policy_idx = 0

    def increment_stage(self):
        self.stage += 1
        self.active_policy = self.policies[self.stage]

    def get_action(self, observation):
        if self.active_policy.take_policy1_step and self.active_policy.num_steps > 0:
            self.increment_stage()
        return self.active_policy.get_action(observation)

    def reset(self):
        self.stage = 0
        self.active_policy = self.policies[0]
        for policy in self.policies:
            policy.reset()

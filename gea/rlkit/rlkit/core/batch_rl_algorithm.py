import abc

import gtimer as gt
from tqdm import tqdm

from rlkit.core import eval_util, logger
from rlkit.core.rl_algorithm import BaseRLAlgorithm, _get_epoch_timings
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
        self,
        trainer,
        exploration_env,
        evaluation_env,
        exploration_data_collector: PathCollector,
        evaluation_data_collector: PathCollector,
        replay_buffer: ReplayBuffer,
        batch_size,
        max_path_length,
        num_epochs,
        num_eval_steps_per_epoch,
        num_expl_steps_per_train_loop,
        num_trains_per_train_loop,
        num_train_loops_per_epoch=1,
        min_num_steps_before_training=0,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in tqdm(
            gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
            )
        ):
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp("evaluation sampling")

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp("exploration sampling", unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp("data storing", unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp("training", unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)


class BatchModularRLAlgorithm(BatchRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
        self,
        trainer,
        exploration_env,
        evaluation_env,
        exploration_data_collector: PathCollector,
        evaluation_data_collector: PathCollector,
        replay_buffer: ReplayBuffer,
        batch_size,
        max_path_length,
        num_epochs,
        num_eval_steps_per_epoch,
        num_expl_steps_per_train_loop,
        num_trains_per_train_loop,
        num_train_loops_per_epoch=1,
        min_num_steps_before_training=0,
        planner_replay_buffer=None,
        planner_trainer=None,
        planner_num_trains_per_train_loop=40,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch,
            min_num_steps_before_training,
        )
        self.planner_replay_buffer = planner_replay_buffer
        self.planner_trainer = planner_trainer
        self.planner_num_trains_per_train_loop = planner_num_trains_per_train_loop

    def get_planner_and_control_paths(self, paths):
        control_paths = []
        planner_paths = []
        for path in paths:
            if path["type"] == "control":
                control_paths.append(path)
            elif path["type"] == "planner":
                planner_paths.append(path)
        return control_paths, planner_paths

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            control_paths, planner_paths = self.get_planner_and_control_paths(
                init_expl_paths
            )
            self.replay_buffer.add_paths(control_paths)
            self.planner_replay_buffer.add_paths(planner_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=False,  # NOTE: paths are necessarily shorter due to switches between planner and policy
            )
            gt.stamp("evaluation sampling")

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp("exploration sampling", unique=False)

                control_paths, planner_paths = self.get_planner_and_control_paths(
                    new_expl_paths
                )
                self.replay_buffer.add_paths(control_paths)
                self.planner_replay_buffer.add_paths(planner_paths)
                gt.stamp("data storing", unique=False)

                self.training_mode(True)
                if self.replay_buffer._size > self.batch_size:
                    for _ in range(self.num_trains_per_train_loop):
                        train_data = self.replay_buffer.random_batch(self.batch_size)
                        self.trainer.train(train_data)
                if self.planner_replay_buffer._size > self.batch_size:
                    for _ in range(self.planner_num_trains_per_train_loop):
                        train_data = self.planner_replay_buffer.random_batch(
                            self.batch_size
                        )
                        self.planner_trainer.train(train_data)
                gt.stamp("training", unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)


class BatchMultiStageModularRLAlgorithm(BatchRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
        self,
        trainers,
        exploration_env,
        evaluation_env,
        exploration_data_collector: PathCollector,
        evaluation_data_collector: PathCollector,
        replay_buffers: ReplayBuffer,
        batch_size,
        max_path_length,
        num_epochs,
        num_eval_steps_per_epoch,
        num_expl_steps_per_train_loop,
        num_trains_per_train_loop,
        num_train_loops_per_epoch=1,
        min_num_steps_before_training=0,
        planner_replay_buffers=None,
        planner_trainers=None,
        planner_num_trains_per_train_loop=40,
        num_stages=1,
        epoch_to_start_training_second_stage=0,
    ):
        super().__init__(
            trainers,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffers,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch,
            min_num_steps_before_training,
        )
        self.replay_buffers = replay_buffers
        self.trainers = trainers

        self.planner_replay_buffers = planner_replay_buffers
        self.planner_trainers = planner_trainers
        self.planner_num_trains_per_train_loop = planner_num_trains_per_train_loop
        self.num_stages = num_stages
        self.epoch_to_start_training_second_stage = epoch_to_start_training_second_stage

    def get_planner_and_control_paths(self, paths, stage):
        control_paths = []
        planner_paths = []
        for path in paths:
            if path["type"] == f"control_{stage}":
                control_paths.append(path)
            elif path["type"] == f"planner_{stage}":
                planner_paths.append(path)
        return control_paths, planner_paths

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            for stage in range(self.num_stages):
                control_paths, planner_paths = self.get_planner_and_control_paths(
                    init_expl_paths, stage
                )
                self.replay_buffers[stage].add_paths(control_paths)
                self.planner_replay_buffers[stage].add_paths(planner_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=False,  # NOTE: paths are necessarily shorter due to switches between planner and policy
            )
            gt.stamp("evaluation sampling")

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp("exploration sampling", unique=False)

                for stage in range(self.num_stages):
                    if stage > 0 and epoch < self.epoch_to_start_training_second_stage:
                        continue
                    control_paths, planner_paths = self.get_planner_and_control_paths(
                        new_expl_paths, stage
                    )
                    self.replay_buffers[stage].add_paths(control_paths)
                    self.planner_replay_buffers[stage].add_paths(planner_paths)
                gt.stamp("data storing", unique=False)

                self.training_mode(True)
                for stage in range(self.num_stages):
                    if stage > 0 and epoch < self.epoch_to_start_training_second_stage:
                        continue
                    if self.replay_buffers[stage]._size > self.batch_size:
                        for _ in range(self.num_trains_per_train_loop):
                            train_data = self.replay_buffers[stage].random_batch(
                                self.batch_size
                            )
                            self.trainers[stage].train(train_data)
                    if self.planner_replay_buffers[stage]._size > self.batch_size:
                        for _ in range(self.planner_num_trains_per_train_loop):
                            train_data = self.planner_replay_buffers[
                                stage
                            ].random_batch(self.batch_size)
                            self.planner_trainers[stage].train(train_data)
                gt.stamp("training", unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)

    def _get_snapshot(self):
        snapshot = {}
        for stage in range(self.num_stages):
            for k, v in self.trainers[stage].get_snapshot().items():
                snapshot[f"trainer_{stage}/" + k] = v
            for k, v in self.planner_trainers[stage].get_snapshot().items():
                snapshot[f"planner_trainer_{stage}/" + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot["exploration/" + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot["evaluation/" + k] = v
        for stage in range(self.num_stages):
            for k, v in self.replay_buffers[stage].get_snapshot().items():
                snapshot[f"replay_buffer_{stage}/" + k] = v
            for k, v in self.planner_replay_buffers[stage].get_snapshot().items():
                snapshot[f"planner_replay_buffer_{stage}/" + k] = v
        return snapshot

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        for stage in range(self.num_stages):
            logger.record_dict(
                self.replay_buffers[stage].get_diagnostics(),
                prefix=f"replay_buffer_{stage}/",
            )
            logger.record_dict(
                self.planner_replay_buffers[stage].get_diagnostics(),
                prefix=f"planner_replay_buffer_{stage}/",
            )
        """
        Trainer
        """
        for stage in range(self.num_stages):
            logger.record_dict(
                self.trainers[stage].get_diagnostics(), prefix=f"trainer_{stage}/"
            )
            logger.record_dict(
                self.planner_trainers[stage].get_diagnostics(),
                prefix=f"planner_trainer_{stage}/",
            )

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(), prefix="exploration/"
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
        if hasattr(self.expl_env, "get_diagnostics"):
            logger.record_dict(
                self.expl_env.get_diagnostics(expl_paths),
                prefix="exploration/",
            )
        logger.record_dict(
            eval_util.get_generic_path_information(expl_paths),
            prefix="exploration/",
        )
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix="evaluation/",
        )
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, "get_diagnostics"):
            logger.record_dict(
                self.eval_env.get_diagnostics(eval_paths),
                prefix="evaluation/",
            )
        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths),
            prefix="evaluation/",
        )

        """
        Misc
        """
        gt.stamp("logging")
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular("Epoch", epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    def _end_epoch(self, epoch):
        snapshot = self._get_snapshot()
        logger.save_itr_params(epoch, snapshot)
        gt.stamp("saving")
        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        for stage in range(self.num_stages):
            self.replay_buffers[stage].end_epoch(epoch)
            self.planner_replay_buffers[stage].end_epoch(epoch)
            self.trainers[stage].end_epoch(epoch)
            self.planner_trainers[stage].end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

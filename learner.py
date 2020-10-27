# -*- coding: utf-8 -*-
"""
@author: ryuichi takanobu
"""

import os
import pickle
import torch
import torch.nn as nn
import logging
import random
import numpy as np
from torch import optim
from policy import MultiDiscretePolicy
from utils import state_vectorize, state_vectorize_user
from hybridv import HybridValue
from torch import multiprocessing as mp
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter

try:
    mp = mp.get_context('spawn')
except RuntimeError:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 记录状态动作等的信息
Transition = namedtuple('Transition', ('state_usr', 'action_usr', 'reward_usr', 'state_usr_next', \
                                       'state_sys', 'action_sys', 'reward_sys', 'state_sys_next', \
                                       'mask', 'reward_global'))


class Memory(object):
    """
    封装 Transition， 提供push，append， get_batch操作
    """

    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def get_batch(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)


def sampler(pid, queue, evt, env, policy_usr, policy_sys, batchsz):
    """
    This is a sampler function, and it will be called by multiprocess.Process to sample data from environment by multiple
    processes.
    随机生成batchsz数量的对话数据, 并记录每一轮的状态和奖励,
    这个对话数据是通过系统生成的,由policy和env共同维护实现

    :param pid: process id
    :param queue: multiprocessing.Queue, to collect sampled data
    :param evt: multiprocessing.Event, to keep the process alive
    :param env: environment instance
    :param policy: policy network, to generate action from current policy
    :param batchsz: total sampled items
    :return:
    """
    buff = Memory()

    # we need to sample batchsz of (state, action, next_state, reward, mask)
    # each trajectory contains `trajectory_len` num of items, so we only need to sample
    # `batchsz//trajectory_len` num of trajectory totally
    # the final sampled number may be larger than batchsz.

    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 40
    real_traj_len = 0

    # sampled_num 表示对话轮数
    while sampled_num < batchsz:
        # for each trajectory, we reset the env and get initial state
        # 初始化状态
        s = env.reset()

        # 完成一个trajectory
        for t in range(traj_len):

            # [s_dim_usr] => [a_dim_usr]
            s_vec = torch.Tensor(state_vectorize_user(s, env.cfg, env.evaluator.cur_domain))
            # 选择动作
            a = policy_usr.select_action(s_vec.to(device=DEVICE)).cpu()

            # interact with env, done is a flag indicates ending or not
            next_s, done = env.step_usr(s, a)

            # [s_dim] => [a_dim]
            next_s_vec = torch.Tensor(state_vectorize(next_s, env.cfg, env.db))
            next_a = policy_sys.select_action(next_s_vec.to(device=DEVICE)).cpu()

            # interact with env
            s = env.step_sys(next_s, next_a)

            # get reward compared to demonstrations
            if done:
                env.set_rollout(True)
                # 貌似是清理env做的操作
                s_vec_next = torch.Tensor(state_vectorize_user(s, env.cfg, env.evaluator.cur_domain))
                a_next = torch.zeros_like(a)
                next_s_next, _ = env.step_usr(s, a_next)
                next_s_vec_next = torch.Tensor(state_vectorize(next_s_next, env.cfg, env.db))
                env.set_rollout(False)

                r_usr = 20 if env.evaluator.inform_F1(ans_by_sys=False)[1] == 1. else -5
                r_sys = 20 if env.evaluator.task_success(False) else -5
                r_global = 20 if env.evaluator.task_success() else -5
            else:
                # one step roll out
                env.set_rollout(True)
                s_vec_next = torch.Tensor(state_vectorize_user(s, env.cfg, env.evaluator.cur_domain))
                a_next = policy_usr.select_action(s_vec_next.to(device=DEVICE)).cpu()
                next_s_next, _ = env.step_usr(s, a_next)
                next_s_vec_next = torch.Tensor(state_vectorize(next_s_next, env.cfg, env.db))
                env.set_rollout(False)

                r_usr = 0
                if not s['user_action']:
                    # 上一轮没有用户动作
                    r_usr -= 5
                if env.evaluator.cur_domain:
                    for da in s['user_action']:
                        d, i, k = da.split('-')
                        if i == 'request':
                            for slot, value in s['goal_state'][d].items():
                                if value != '?' and slot in s['user_goal'][d] \
                                        and s['user_goal'][d][slot] != '?':
                                    # request before express constraint
                                    r_usr -= 1
                r_sys = 0
                if not next_s['sys_action']:
                    # 上一轮没有系统动作
                    r_sys -= 5
                if env.evaluator.cur_domain:
                    for slot, value in next_s['belief_state'][env.evaluator.cur_domain].items():
                        if value == '?':
                            for da in next_s['sys_action']:
                                d, i, k, p = da.split('-')
                                if i in ['inform', 'recommend', 'offerbook', 'offerbooked'] and k == slot:
                                    break
                            else:
                                # not answer request
                                # 重要有一个slot没有回答，就抠一点奖励
                                r_sys -= 1
                r_global = 5 if env.evaluator.cur_domain and env.evaluator.domain_success(
                    env.evaluator.cur_domain) else -1

            # save to queue
            # 训练数据导入到Memory中
            buff.push(s_vec.numpy(), a.numpy(), r_usr, s_vec_next.numpy(), next_s_vec.numpy(), next_a.numpy(), r_sys,
                      next_s_vec_next.numpy(), done, r_global)

            # update per step
            real_traj_len = t

            if done:
                break

        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1
        # t indicates the valid trajectory length

    # this is end of sampling all batchsz of items.
    # when sampling is over, push all buff data into queue
    queue.put([pid, buff])
    evt.wait()


class Learner():

    def __init__(self, env_cls, args, cfg, process_num, infer=False):
        self.policy_sys = MultiDiscretePolicy(cfg).to(device=DEVICE)
        self.policy_usr = MultiDiscretePolicy(cfg, 'usr').to(device=DEVICE)
        self.vnet = HybridValue(cfg).to(device=DEVICE)

        # initialize envs for each process
        self.env_list = []
        for _ in range(process_num):
            self.env_list.append(env_cls(args.data_dir, cfg))

        self.policy_sys.eval()
        self.policy_usr.eval()
        self.vnet.eval()
        self.infer = infer

        if not infer:
            self.l2_loss = nn.MSELoss()
            self.multi_entropy_loss = nn.BCEWithLogitsLoss()
            self.target_vnet = HybridValue(cfg).to(device=DEVICE)
            self.episode_num = 0
            self.last_target_update_episode = 0
            # 目标网络更新间隔
            self.target_update_interval = args.interval

            self.policy_sys_optim = optim.RMSprop(self.policy_sys.parameters(), lr=args.lr_policy)
            self.policy_usr_optim = optim.RMSprop(self.policy_usr.parameters(), lr=args.lr_policy_usr)
            self.vnet_optim = optim.RMSprop(self.vnet.parameters(), lr=args.lr_vnet, weight_decay=args.weight_decay)

        self.gamma = args.gamma
        self.grad_norm_clip = args.clip
        self.optim_batchsz = args.batchsz
        self.save_per_epoch = args.save_per_epoch
        self.save_dir = args.save_dir
        self.process_num = process_num
        self.writer = SummaryWriter()

    def _update_targets(self):
        """
        将vnet的参数更新至target_vnet
        """
        self.target_vnet.load_state_dict(self.vnet.state_dict())
        logging.info('Updated target network')

    def evaluate(self, N):
        """
        评估效果，强化学习由于是通过不断模拟甚至用户的操作从而生成训练数据，
        因此验证的方式也是采用多次生成数据来进行验证
        """
        logging.info('eval: user 2 system')
        env = self.env_list[0]
        traj_len = 40
        turn_tot, inform_tot, match_tot, success_tot = [], [], [], []
        for seed in range(N):
            s = env.reset(seed)
            print('seed', seed)
            print('origin goal', env.goal)
            print('goal', env.evaluator.goal)
            for t in range(traj_len):
                s_vec = torch.Tensor(state_vectorize_user(s, env.cfg, env.evaluator.cur_domain)).to(device=DEVICE)
                # mode with policy during evaluation
                a = self.policy_usr.select_action(s_vec, False)
                next_s, done = env.step_usr(s, a)

                next_s_vec = torch.Tensor(state_vectorize(next_s, env.cfg, env.db)).to(device=DEVICE)
                next_a = self.policy_sys.select_action(next_s_vec, False)
                s = env.step_sys(next_s, next_a)

                print('usr', s['user_action'])
                print('sys', s['sys_action'])

                if done:
                    break

            turn_tot.append(env.time_step // 2)
            match_tot += env.evaluator.match_rate(aggregate=False)
            inform_tot.append(env.evaluator.inform_F1(aggregate=False))
            # 计算单个会话的成功率
            print('turn', env.time_step // 2)
            match_session = env.evaluator.match_rate()
            print('match', match_session)
            inform_session = env.evaluator.inform_F1()
            print('inform', inform_session)
            if (match_session == 1 and inform_session[1] == 1) \
                    or (match_session == 1 and inform_session[1] is None) \
                    or (match_session is None and inform_session[1] == 1):
                print('success', 1)
                success_tot.append(1)
            else:
                print('success', 0)
                success_tot.append(0)

        # 计算所有次试验的效果
        logging.info('turn {}'.format(np.mean(turn_tot)))
        logging.info('match {}'.format(np.mean(match_tot)))
        TP, FP, FN = np.sum(inform_tot, 0)
        prec = TP / (TP + FP)
        rec = TP / (TP + FN)
        F1 = 2 * prec * rec / (prec + rec)
        logging.info('inform rec {}, F1 {}'.format(rec, F1))
        logging.info('success {}'.format(np.mean(success_tot)))

    def evaluate_with_agenda(self, env, N):
        """
        和上面不同的是，这里是使用定义好的env，上面实现的是内置的env,
        用于针对系统agent，专门进行验证的
        """
        logging.info('eval: agenda 2 system')
        traj_len = 40
        turn_tot, inform_tot, match_tot, success_tot = [], [], [], []
        for seed in range(N):
            s = env.reset(seed)
            print('seed', seed)
            print('goal', env.goal.domain_goals)
            print('usr', s['user_action'])
            for t in range(traj_len):
                s_vec = torch.Tensor(state_vectorize(s, env.cfg, env.db)).to(device=DEVICE)
                # mode with policy during evaluation
                a = self.policy_sys.select_action(s_vec, False)
                # 这一步操作，应该已经在env中包含了用户的操作，以及状态变更
                next_s, done = env.step(s, a.cpu())
                s = next_s
                print('sys', s['sys_action'])
                print('usr', s['user_action'])
                if done:
                    break
            s_vec = torch.Tensor(state_vectorize(s, env.cfg, env.db)).to(device=DEVICE)
            # mode with policy during evaluation
            a = self.policy_sys.select_action(s_vec, False)
            s = env.update_belief_sys(s, a.cpu())
            print('sys', s['sys_action'])

            assert (env.time_step % 2 == 0)
            turn_tot.append(env.time_step // 2)
            match_tot += env.evaluator.match_rate(aggregate=False)
            inform_tot.append(env.evaluator.inform_F1(aggregate=False))
            print('turn', env.time_step // 2)
            match_session = env.evaluator.match_rate()
            print('match', match_session)
            inform_session = env.evaluator.inform_F1()
            print('inform', inform_session)
            if (match_session == 1 and inform_session[1] == 1) \
                    or (match_session == 1 and inform_session[1] is None) \
                    or (match_session is None and inform_session[1] == 1):
                print('success', 1)
                success_tot.append(1)
            else:
                print('success', 0)
                success_tot.append(0)

        logging.info('turn {}'.format(np.mean(turn_tot)))
        logging.info('match {}'.format(np.mean(match_tot)))
        TP, FP, FN = np.sum(inform_tot, 0)
        prec = TP / (TP + FP)
        rec = TP / (TP + FN)
        F1 = 2 * prec * rec / (prec + rec)
        logging.info('inform rec {}, F1 {}'.format(rec, F1))
        logging.info('success {}'.format(np.mean(success_tot)))

    def evaluate_with_rule(self, env, N):
        """
        单独针对用户agent准备的验证方法
        """
        logging.info('eval: user 2 rule')
        traj_len = 40
        turn_tot, inform_tot, match_tot, success_tot = [], [], [], []
        for seed in range(N):
            s = env.reset(seed)
            #print('seed', seed)
            #print('goal', env.evaluator.goal)
            for t in range(traj_len):
                s_vec = torch.Tensor(state_vectorize_user(s, env.cfg, env.evaluator.cur_domain)).to(device=DEVICE)
                # mode with policy during evaluation
                a = self.policy_usr.select_action(s_vec, False)
                next_s = env.step(s, a.cpu())
                s = next_s
            #    print('usr', s['user_action'])
             #   print('sys', s['sys_action'])
                done = s['others']['terminal']
                if done:
                    break

            assert (env.time_step % 2 == 0)
            turn_tot.append(env.time_step // 2)
            match_tot += env.evaluator.match_rate(aggregate=False)
            inform_tot.append(env.evaluator.inform_F1(aggregate=False))
            #print('turn', env.time_step // 2)
            match_session = env.evaluator.match_rate()
           # print('match', match_session)
            inform_session = env.evaluator.inform_F1()
            #print('inform', inform_session)
            if (match_session == 1 and inform_session[1] == 1) \
                    or (match_session == 1 and inform_session[1] is None) \
                    or (match_session is None and inform_session[1] == 1):
             #   print('success', 1)
                success_tot.append(1)
            else:
              #  print('success', 0)
                success_tot.append(0)

        logging.info('turn {}'.format(np.mean(turn_tot)))
        logging.info('match {}'.format(np.mean(match_tot)))
        TP, FP, FN = np.sum(inform_tot, 0)
        prec = TP / (TP + FP)
        rec = TP / (TP + FN)
        F1 = 2 * prec * rec / (prec + rec)
        logging.info('inform rec {}, F1 {}'.format(rec, F1))
        logging.info('success {}'.format(np.mean(success_tot)))

    def save(self, directory, epoch):
        """
        保存模型
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.makedirs(directory + '/usr')
            os.makedirs(directory + '/sys')
            os.makedirs(directory + '/vnet')

        torch.save(self.policy_usr.state_dict(), directory + '/usr/' + str(epoch) + '_pol.mdl')
        torch.save(self.policy_sys.state_dict(), directory + '/sys/' + str(epoch) + '_pol.mdl')
        torch.save(self.vnet.state_dict(), directory + '/vnet/' + str(epoch) + '_vnet.mdl')

        logging.info('<<multi agent learner>> epoch {}: saved network to mdl'.format(epoch))

    def load(self, filename):
        """
        加载模型
        """

        directory, epoch = filename.rsplit('/', 1)

        # 加载用户策略模型
        policy_usr_mdl = directory + '/usr/' + epoch + '_pol.mdl'
        if os.path.exists(policy_usr_mdl):
            self.policy_usr.load_state_dict(torch.load(policy_usr_mdl))
            logging.info('<<dialog policy usr>> loaded checkpoint from file: {}'.format(policy_usr_mdl))

        # 加载系统策略模型
        policy_sys_mdl = directory + '/sys/' + epoch + '_pol.mdl'
        if os.path.exists(policy_sys_mdl):
            self.policy_sys.load_state_dict(torch.load(policy_sys_mdl))
            logging.info('<<dialog policy sys>> loaded checkpoint from file: {}'.format(policy_sys_mdl))

        # vnet只是用来计算奖励函数的，预测过程中是不需要的
        if not self.infer:
            self._update_targets()

        best_pkl = filename + '.pkl'
        if os.path.exists(best_pkl):
            with open(best_pkl, 'rb') as f:
                best = pickle.load(f)
        else:
            best = float('-inf')
        return best

    def sample(self, batchsz):
        """
        Given batchsz number of task, the batchsz will be split equally to each processes
        and when processes return, it merge all data and return
        将数据按照process_num进行划分，多进程处理
        :param batchsz:
        :return: batch
        """

        # batchsz will be split into each process,
        # final batchsz maybe larger than batchsz parameters
        process_batchsz = np.ceil(batchsz / self.process_num).astype(np.int32)
        # buffer to save all data
        queue = mp.Queue()

        # start processes for pid in range(1, processnum)
        # if processnum = 1, this part will be ignored.
        # when save tensor in Queue, the process should keep alive till Queue.get(),
        # please refer to : https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847
        # however still some problem on CUDA tensors on multiprocessing queue,
        # please refer to : https://discuss.pytorch.org/t/cuda-tensors-on-multiprocessing-queue/28626
        # so just transform tensors into numpy, then put them into queue.
        evt = mp.Event()
        processes = []
        for i in range(self.process_num):
            process_args = (i, queue, evt, self.env_list[i], self.policy_usr, self.policy_sys, process_batchsz)
            processes.append(mp.Process(target=sampler, args=process_args))
        for p in processes:
            # set the process as daemon, and it will be killed once the main process is stoped.
            p.daemon = True
            p.start()

        # we need to get the first Memory object and then merge others Memory use its append function.
        pid0, buff0 = queue.get()
        for _ in range(1, self.process_num):
            pid, buff_ = queue.get()
            buff0.append(buff_)  # merge current Memory into buff0
        evt.set()

        # now buff saves all the sampled data
        buff = buff0

        return buff.get_batch()

    def update(self, batchsz, epoch, best=None):
        """
        firstly sample batchsz items and then perform optimize algorithms.
        比较关键的算法部分

        :param batchsz:
        :param epoch:
        :param best:
        :return:
        """
        backward = True if best is None else False
        if backward:
            self.policy_usr.train()
            self.policy_sys.train()
            self.vnet.train()

        # 1. sample data asynchronously
        # 随机采样得到训练数据,得到一个batch的数据,数据以轮为基本单位，每轮包含一次交互,用户和系统和执行一次动作
        # 训练数据主要基于原始模型生成，因此数据的质量并不好，如何提升训练数据的质量？
        batch = self.sample(batchsz)

        policy_usr_loss, policy_sys_loss, vnet_usr_loss, vnet_sys_loss, vnet_glo_loss = 0., 0., 0., 0., 0.

        # data in batch is : batch.state: ([1, s_dim], [1, s_dim]...)
        # batch.action: ([1, a_dim], [1, a_dim]...)
        # batch.reward/batch.mask: ([1], [1]...)
        s_usr = torch.from_numpy(np.stack(batch.state_usr)).to(device=DEVICE)
        a_usr = torch.from_numpy(np.stack(batch.action_usr)).to(device=DEVICE)
        r_usr = torch.Tensor(np.stack(batch.reward_usr)).to(device=DEVICE)
        s_usr_next = torch.from_numpy(np.stack(batch.state_usr_next)).to(device=DEVICE)
        s_sys = torch.from_numpy(np.stack(batch.state_sys)).to(device=DEVICE)
        a_sys = torch.from_numpy(np.stack(batch.action_sys)).to(device=DEVICE)
        r_sys = torch.Tensor(np.stack(batch.reward_sys)).to(device=DEVICE)
        s_sys_next = torch.from_numpy(np.stack(batch.state_sys_next)).to(device=DEVICE)
        ternimal = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
        r_glo = torch.Tensor(np.stack(batch.reward_global)).to(device=DEVICE)
        batchsz = s_usr.size(0)

        if not backward:
            # 不进行梯度更新
            # 将所有奖励取平均相加
            reward = r_usr.mean().item() + r_sys.mean().item() + r_glo.mean().item()
            logging.debug('validation, epoch {}, reward {}'.format(epoch, reward))
            self.writer.add_scalar('train/reward', reward, epoch)
            # 更新最大奖励值并记录，结束更新
            if reward > best:
                logging.info('best model saved')
                best = reward
                self.save(self.save_dir, 'best')
            with open(self.save_dir + '/best.pkl', 'wb') as f:
                pickle.dump(best, f)
            return best
        else:
            logging.debug(
                'epoch {}, reward: usr {}, sys {}, global {}'.format(epoch, r_usr.mean().item(), r_sys.mean().item(),
                                                                     r_glo.mean().item()))

        # 6. update dialog policy

        # 1. shuffle current batch
        # 打乱序列，保证样本独立性
        perm = torch.randperm(batchsz)
        # shuffle the variable for mutliple optimize
        s_usr_shuf, a_usr_shuf, r_usr_shuf, s_usr_next_shuf, s_sys_shuf, a_sys_shuf, r_sys_shuf, s_sys_next_shuf, terminal_shuf, r_glo_shuf = \
            s_usr[perm], a_usr[perm], r_usr[perm], s_usr_next[perm], s_sys[perm], a_sys[perm], r_sys[perm], s_sys_next[
                perm], ternimal[perm], r_glo[perm]

        # 2. get mini-batch for optimizing
        # 按照更新次数，切分数据
        print("\nbatchsz:", batchsz, "optim_batchsz:", self.optim_batchsz,"\n")
        optim_chunk_num = int(np.ceil(batchsz / self.optim_batchsz))

        # chunk the optim_batch for total batch
        s_usr_shuf, a_usr_shuf, r_usr_shuf, s_usr_next_shuf, s_sys_shuf, a_sys_shuf, r_sys_shuf, s_sys_next_shuf, terminal_shuf, r_glo_shuf = \
            torch.chunk(s_usr_shuf, optim_chunk_num), torch.chunk(a_usr_shuf, optim_chunk_num), torch.chunk(r_usr_shuf,
                                                                                                            optim_chunk_num), torch.chunk(
                s_usr_next_shuf, optim_chunk_num), \
            torch.chunk(s_sys_shuf, optim_chunk_num), torch.chunk(a_sys_shuf, optim_chunk_num), torch.chunk(r_sys_shuf,
                                                                                                            optim_chunk_num), torch.chunk(
                s_sys_next_shuf, optim_chunk_num), \
            torch.chunk(terminal_shuf, optim_chunk_num), torch.chunk(r_glo_shuf, optim_chunk_num)

        # 3. iterate all mini-batch to optimize
        for s_usr_b, a_usr_b, r_usr_b, s_usr_next_b, s_sys_b, a_sys_b, r_sys_b, s_sys_next_b, t_b, r_glo_b in \
                zip(s_usr_shuf, a_usr_shuf, r_usr_shuf, s_usr_next_shuf, \
                    s_sys_shuf, a_sys_shuf, r_sys_shuf, s_sys_next_shuf, \
                    terminal_shuf, r_glo_shuf):

            # 1. update critic network

            # update usr vnet
            vals_usr = self.vnet(s_usr_b, 'usr')
            # 如果为终止状态，则没有next_b
            target_usr = r_usr_b + self.gamma * (1 - t_b) * self.target_vnet(s_usr_next_b, 'usr')
            # 比较target_vnet推导出来的值和vnet预测的到的值函数的差异
            loss_usr = self.l2_loss(vals_usr, target_usr)
            vnet_usr_loss += loss_usr.item()

            # update sys vnet
            vals_sys = self.vnet(s_sys_b, 'sys')
            target_sys = r_sys_b + self.gamma * (1 - t_b) * self.target_vnet(s_sys_next_b, 'sys')
            loss_sys = self.l2_loss(vals_sys, target_sys)
            vnet_sys_loss += loss_sys.item()

            # update global vnet
            vals_glo = self.vnet((s_usr_b, s_sys_b), 'global')
            target_glo = r_glo_b + self.gamma * (1 - t_b) * self.target_vnet((s_usr_next_b, s_sys_next_b), 'global')
            loss_glo = self.l2_loss(vals_glo, target_glo)
            vnet_glo_loss += loss_glo.item()

            self.vnet_optim.zero_grad()
            loss = loss_usr + loss_sys + loss_glo
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vnet.parameters(), self.grad_norm_clip)
            self.vnet_optim.step()

            self.episode_num += 1
            if (self.episode_num - self.last_target_update_episode) / self.target_update_interval >= 1.0:
                self._update_targets()
                self.last_target_update_episode = self.episode_num

            # 2. update actor network

            # estimate advantage using current critic
            # 奖励函数
            td_error_usr = r_usr_b + self.gamma * (1 - t_b) * self.vnet(s_usr_next_b, 'usr') - self.vnet(s_usr_b, 'usr')
            td_error_sys = r_sys_b + self.gamma * (1 - t_b) * self.vnet(s_sys_next_b, 'sys') - self.vnet(s_sys_b, 'sys')
            td_error_glo = r_glo_b + self.gamma * (1 - t_b) * self.vnet((s_usr_next_b, s_sys_next_b),
                                                                        'global') - self.vnet((s_usr_b, s_sys_b),
                                                                                              'global')

            self.policy_usr_optim.zero_grad()
            # [b, 1]
            log_pi_sa = self.policy_usr.get_log_prob(s_usr_b, a_usr_b)
            # this is element-wise comparing.
            # we add negative symbol to convert gradient ascent to gradient descent
            # log_pi_sa为预测的对数概率决定该结果是否可信，td_error为该结果对奖励的贡献程度
            surrogate = - (log_pi_sa * (td_error_usr + td_error_glo)).mean()
            policy_usr_loss += surrogate.item()

            # backprop
            surrogate.backward(retain_graph=True)
            # gradient clipping, for stability
            torch.nn.utils.clip_grad_norm(self.policy_usr.parameters(), self.grad_norm_clip)
            # self.lock.acquire() # retain lock to update weights
            self.policy_usr_optim.step()
            # self.lock.release() # release lock

            self.policy_sys_optim.zero_grad()
            # [b, 1]
            log_pi_sa = self.policy_sys.get_log_prob(s_sys_b, a_sys_b)
            # this is element-wise comparing.
            # we add negative symbol to convert gradient ascent to gradient descent
            surrogate = - (log_pi_sa * (td_error_sys + td_error_glo)).mean()
            policy_sys_loss += surrogate.item()

            # backprop
            surrogate.backward()
            # gradient clipping, for stability
            torch.nn.utils.clip_grad_norm(self.policy_sys.parameters(), self.grad_norm_clip)
            # self.lock.acquire() # retain lock to update weights
            self.policy_sys_optim.step()
            # self.lock.release() # release lock

        vnet_usr_loss /= optim_chunk_num
        vnet_sys_loss /= optim_chunk_num
        vnet_glo_loss /= optim_chunk_num
        policy_usr_loss /= optim_chunk_num
        policy_sys_loss /= optim_chunk_num

        logging.debug('epoch {}, policy: usr {}, sys {}, value network: usr {}, sys {}, global {}'.format(epoch, \
                                                                                                          policy_usr_loss,
                                                                                                          policy_sys_loss,
                                                                                                          vnet_usr_loss,
                                                                                                          vnet_sys_loss,
                                                                                                          vnet_glo_loss))
        self.writer.add_scalar('train/usr_policy_loss', policy_usr_loss, epoch)
        self.writer.add_scalar('train/sys_policy_loss', policy_sys_loss, epoch)
        self.writer.add_scalar('train/vnet_usr_loss', vnet_usr_loss, epoch)
        self.writer.add_scalar('train/vnet_sys_loss', vnet_sys_loss, epoch)
        self.writer.add_scalar('train/vnet_glo_loss', vnet_glo_loss, epoch)

        if (epoch + 1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)
            with open(self.save_dir + '/' + str(epoch) + '.pkl', 'wb') as f:
                pickle.dump(best, f)
        self.policy_usr.eval()
        self.policy_sys.eval()
        self.vnet.eval()

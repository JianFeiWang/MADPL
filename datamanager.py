# -*- coding: utf-8 -*-
"""
@author: ryuichi takanobu
"""
import os
import json
import logging
import torch
import torch.utils.data as data
import copy
from copy import deepcopy
from evaluator import MultiWozEvaluator
from utils import init_session, init_goal, state_vectorize, action_vectorize, \
    state_vectorize_user, action_vectorize_user, discard, reload


def expand_da(meta):
    """
    对元数据？进行一些变化，
    对于 request请求 插入？符号
    对于 slot为none， 插入none
    对于 slot不为none， 插入slot出现的次数
    """
    for k, v in meta.items():
        domain, intent = k.split('-')
        if intent.lower() == "request":
            for pair in v:
                pair.insert(1, '?')
        else:
            counter = {}
            for pair in v:
                if pair[0] == 'none':
                    pair.insert(1, 'none')
                else:
                    if pair[0] in counter:
                        counter[pair[0]] += 1
                    else:
                        counter[pair[0]] = 1
                    pair.insert(1, str(counter[pair[0]]))


def add_domain_mask(data):
    """
    为每个turn 增加next_avaliable_domain 和 invisible_domains信息
    """
    # 限制需要从文档中读取的domains，如果需要从源文件单独提取某个domain，可以通过修改这个来实现
    all_domains = ["train", "hotel", "restaurant", "attraction", "taxi", "police", "hospital"]
    parts = ["train", "valid", "test"]
    # 分别对不同的parts进行处理
    for part in parts:
        dataset = data[part]
        domains_in_order_dict = {}  # {session_id:domains_in_order}
        domains_in_order = []
        current_session_id = ""
        for turn in dataset:
            # 判断是否为新的会话
            session_id = turn["others"]["session_id"]
            if session_id != current_session_id:
                if current_session_id != "":
                    # 对上一个会话session，记录domains_in_order信息
                    domains_in_order_dict[current_session_id] = domains_in_order
                domains_in_order = []
                current_session_id = session_id

            if "trg_user_action" in turn:
                # 通过用户执行动作来决定domain的变更
                user_das = turn["trg_user_action"]
                for user_da in user_das:
                    [domain, intent, slot] = user_da.split('-')
                    if domain in all_domains and domain not in domains_in_order:
                        # 按照顺序记录出现的domain
                        domains_in_order.append(domain)
        domains_in_order_dict[current_session_id] = domains_in_order  # for last dialog 最后一个session

        current_session_id = ""
        next_available_domain = ""
        invisible_domains = []
        # 为每个turn提供next_available_domain和invisible_domains信息
        for turn in dataset:
            session_id = turn["others"]["session_id"]
            if session_id != current_session_id:
                domains_in_order = domains_in_order_dict[session_id]
                if domains_in_order:
                    next_available_domain = domains_in_order[0]
                    invisible_domains = domains_in_order[1:]
                else:
                    next_available_domain = ""
                    invisible_domains = []
                current_session_id = session_id
            turn["next_available_domain"] = next_available_domain
            turn["invisible_domains"] = copy.copy(invisible_domains)

            if "trg_user_action" in turn:
                user_das = turn["trg_user_action"]
                # 依次更新next_available_domain 和 invisible_domains
                for user_da in user_das:
                    [domain, intent, slot] = user_da.split('-')
                    if domain == next_available_domain:
                        if invisible_domains:
                            next_available_domain = invisible_domains[0]
                            invisible_domains.remove(next_available_domain)


class DataManager():
    """Offline data manager"""

    def __init__(self, data_dir, cfg, domain=None):
        """
        DataManager的两种加载方式，第一种是直接通过源文件进行加载，还有一种是通过已经处理好的数据进行加载
        DataManager 只会在预训练的时候使用到
        """
        # 主要实现对data和goal的管理
        self.data = {}
        self.goal = {}

        if domain:
            self.data_dir_new = data_dir + '/processed_data_' + domain
        else:
            self.data_dir_new = data_dir + '/processed_data'

        # **********************
        # 实验一 2020/10/15
        # 对原始数据，按照domain划分为不同的文件，对于对话中同时具有多个domain的暂时不做处理。
        # 验证MADPL对单个domain的效果
        # **********************
        if False and os.path.exists(self.data_dir_new):
            # 对于已经划分并转化后的train valid test的文件进行读取
            logging.info('Load processed data file')
            for part in ['train', 'valid', 'test']:
                with open(self.data_dir_new + '/' + part + '.json', 'r') as f:
                    self.data[part] = json.load(f)
                with open(self.data_dir_new + '/' + part + '_goal.json', 'r') as f:
                    self.goal[part] = json.load(f)
        else:
            # 否则基于数据库完成数据的原始构件
            from dbquery import DBQuery
            db = DBQuery(data_dir, cfg)
            logging.info('Start preprocessing the dataset')
            # 创建session数据
            self._build_data(data_dir, self.data_dir_new, cfg, db)

        # 构建pt文件, 记录pytorch训练数据
        for part in ['train', 'valid', 'test']:
            file_dir = self.data_dir_new + '/' + part + '_sys.pt'
            if not os.path.exists(file_dir):
                from dbquery import DBQuery
                db = DBQuery(data_dir, cfg)
                self.create_dataset_sys(part, file_dir, data_dir, cfg, db)

            file_dir = self.data_dir_new + '/' + part + '_usr.pt'
            if not os.path.exists(file_dir):
                from dbquery import DBQuery
                db = DBQuery(data_dir, cfg)
                self.create_dataset_usr(part, file_dir, data_dir, cfg, db)

            file_dir = self.data_dir_new + '/' + part + '_glo.pt'
            if not os.path.exists(file_dir):
                from dbquery import DBQuery
                db = DBQuery(data_dir, cfg)
                self.create_dataset_global(part, file_dir, data_dir, cfg, db)

    def _build_data(self, data_dir, data_dir_new, cfg, db):
        """
        按照train，dev，test构建session数据
        belief_state, goal_state,others,sys_action,trg_sys_action,usr_action,trg_usr_action
        以及附加的状态包括
        final_belief_state, final_goal_state, next_avaliable_domain, invisible_domains
        """
        data_filename = data_dir + '/' + cfg.data_file
        with open(data_filename, 'r') as f:
            origin_data = json.load(f)

        for part in ['train', 'valid', 'test']:
            self.data[part] = []
            self.goal[part] = {}

        valList = []
        with open(data_dir + '/' + cfg.val_file) as f:
            for line in f:
                valList.append(line.split('.')[0])
        testList = []
        with open(data_dir + '/' + cfg.test_file) as f:
            for line in f:
                testList.append(line.split('.')[0])

        num_sess = 0
        for k_sess in origin_data:
            sess = origin_data[k_sess]
            if k_sess in valList:
                part = 'valid'
            elif k_sess in testList:
                part = 'test'
            else:
                part = 'train'
            turn_data, session_data = init_session(k_sess, cfg)
            # belief_state
            belief_state = turn_data['belief_state']
            # goal_state
            goal_state = turn_data['goal_state']
            init_goal(session_data, goal_state, sess['goal'], cfg)
            # 直接跳过多domain场景
            if "SNG" not in k_sess:
                continue

            # 判断如果数据中没有指定domain场景，则不使用该数据w
            contain_domain=False
            for domain in session_data:
                content = session_data[domain]
                if domain in cfg.belief_domains and len(content) > 0:
                    contain_domain=True
                    break
            if contain_domain is False:
                continue

            num_sess += 1

            # goal
            # 完整的用户目标， goal 和 goal_state 的差异在于， goal为任务最终的目标， goal_state 表示用户目标在某一时刻的状态
            self.goal[part][k_sess] = deepcopy(session_data)
            current_domain = ''
            book_domain = ''
            turn_data['trg_user_action'] = {}
            turn_data['trg_sys_action'] = {}

            for i, turn in enumerate(sess['log']):
                # 注意区分 turn_data为session数据，turn为原始数据
                turn_data['others']['turn'] = i
                turn_data['others']['terminal'] = i + 2 >= len(sess['log'])
                # 记录本轮的动作
                da_origin = turn['dialog_act']
                expand_da(da_origin)
                turn_data['belief_state'] = deepcopy(belief_state)  # from previous turn
                turn_data['goal_state'] = deepcopy(goal_state)

                if i % 2 == 0:  # user
                    # sys_action为上一轮系统动作，trg_sys_action为本轮系统动作,
                    # 因此需要将trg_sys_action赋值给sys_action,之后删除trg_sys_action
                    turn_data['sys_action'] = deepcopy(turn_data['trg_sys_action'])
                    del (turn_data['trg_sys_action'])

                    turn_data['trg_user_action'] = dict()
                    for domint in da_origin:
                        domain_intent = da_origin[domint]
                        _domint = domint.lower()
                        _domain, _intent = _domint.split('-')
                        # current_domain 防止出现booking domain的情况
                        if _domain in cfg.belief_domains:
                            current_domain = _domain
                        for slot, p, value in domain_intent:
                            _slot = slot.lower()
                            _value = value.strip()
                            _da = '-'.join((_domint, _slot))
                            # 符合要求的用户动作，如果需要限制用户动作的范围，可以通过修改cfg.da_user即可
                            if _da in cfg.da_usr:
                                turn_data['trg_user_action'][_da] = _value
                                if _intent == 'inform':
                                    inform_da = _domain + '-' + _slot
                                    # 根据动作，调整belief_state和goal_state
                                    if inform_da in cfg.inform_da:
                                        belief_state[_domain][_slot] = _value
                                    if inform_da in cfg.inform_da_usr and _slot in session_data[_domain] \
                                            and session_data[_domain][_slot] != '?':
                                        discard(goal_state[_domain], _slot)
                                elif _intent == 'request':
                                    request_da = _domain + '-' + _slot
                                    if request_da in cfg.request_da:
                                        belief_state[_domain][_slot] = '?'

                else:  # sys
                    # metadata 记录book的状态
                    book_status = turn['metadata']
                    for domain in cfg.belief_domains:
                        if book_status[domain]['book']['booked']:
                            entity = book_status[domain]['book']['booked'][0]
                            # 表示已经完成booked，可以不需要再更新订购状体
                            if 'booked' in belief_state[domain]:
                                continue

                            book_domain = domain
                            if domain in ['taxi', 'hospital', 'police']:
                                belief_state[domain]['booked'] = f'{domain}-booked'
                            elif domain == 'train':
                                found = db.query(domain, [('trainID', entity['trainID'])])
                                belief_state[domain]['booked'] = found[0]['ref']
                            else:
                                found = db.query(domain, [('name', entity['name'])])
                                belief_state[domain]['booked'] = found[0]['ref']

                    # 对用户动作做相同的处理
                    turn_data['user_action'] = deepcopy(turn_data['trg_user_action'])
                    del (turn_data['trg_user_action'])

                    # 保证只会影响到下一个用户动作
                    turn_data['others']['change'] = False
                    turn_data['trg_sys_action'] = dict()
                    for domint in da_origin:
                        domain_intent = da_origin[domint]
                        _domint = domint.lower()
                        _domain, _intent = _domint.split('-')
                        for slot, p, value in domain_intent:
                            _slot = slot.lower()
                            _value = value.strip()
                            _da = '-'.join((_domint, _slot, p))
                            if _da in cfg.da and current_domain:
                                if _slot == 'ref':
                                    turn_data['trg_sys_action'][_da] = belief_state[book_domain]['booked']
                                else:
                                    turn_data['trg_sys_action'][_da] = _value
                                if _intent in ['inform', 'recommend', 'offerbook', 'offerbooked', 'book']:
                                    inform_da = current_domain + '-' + _slot
                                    if inform_da in cfg.request_da:
                                        discard(belief_state[current_domain], _slot, '?')
                                    if inform_da in cfg.request_da_usr and _slot in session_data[current_domain] \
                                            and session_data[current_domain][_slot] == '?':
                                        goal_state[current_domain][_slot] = _value
                                elif _intent in ['nooffer', 'nobook']:
                                    # TODO: better transition
                                    for da in turn_data['user_action']:
                                        __domain, __intent, __slot = da.split('-')
                                        # 如果系统无法提供服务，则删除上一次用户针对本domain提供的inform
                                        if __intent == 'inform' and __domain == current_domain:
                                            discard(belief_state[current_domain], __slot)
                                    # 改变change标志位，表示用户目标发生转变
                                    turn_data['others']['change'] = True
                                    reload(goal_state, session_data, current_domain)
                # 最后一轮，登入final状态
                if i + 1 == len(sess['log']):
                    turn_data['final_belief_state'] = belief_state
                    turn_data['final_goal_state'] = goal_state

                self.data[part].append(deepcopy(turn_data))
        print("session number: ",num_sess)
        add_domain_mask(self.data)

        def _set_default(obj):
            if isinstance(obj, set):
                return list(obj)
            raise TypeError

        # 实验一
        # 按照单个domain划分

        os.makedirs(data_dir_new)
        for part in ['train', 'valid', 'test']:
            with open(data_dir_new + '/' + part + '.json', 'w') as f:
                self.data[part] = json.dumps(self.data[part], default=_set_default)
                f.write(self.data[part])
                self.data[part] = json.loads(self.data[part])
            with open(data_dir_new + '/' + part + '_goal.json', 'w') as f:
                self.goal[part] = json.dumps(self.goal[part], default=_set_default)
                f.write(self.goal[part])
                self.goal[part] = json.loads(self.goal[part])

    def create_dataset_sys(self, part, file_dir, data_dir, cfg, db):
        """
        创建sys的训练数据
        """
        datas = self.data[part]
        goals = self.goal[part]
        # 系统状态+系统动作+回报+上一轮系统状态+末轮标志位
        s, a, r, next_s, t = [], [], [], [], []
        # evaluator 全称记录数据
        evaluator = MultiWozEvaluator(data_dir, cfg.d)
        for idx, turn_data in enumerate(datas):
            # user
            # 用户侧并没有做数据的更新操作
            if turn_data['others']['turn'] % 2 == 0:
                # 首轮对话加载用户目标
                if turn_data['others']['turn'] == 0:
                    evaluator.add_goal(goals[turn_data['others']['session_id']])
                #
                evaluator.add_usr_da(turn_data['trg_user_action'])
                continue

            # 错位了，确实表示的下一轮状态
            if turn_data['others']['turn'] != 1:
                next_s.append(s[-1])

            # 将当前数据转化为状态向量
            s.append(torch.Tensor(state_vectorize(turn_data, cfg, db, True)))
            # 将当前动作转化为动作向量
            a.append(torch.Tensor(action_vectorize(turn_data['trg_sys_action'], cfg)))
            evaluator.add_sys_da(turn_data['trg_sys_action'])
            if turn_data['others']['terminal']:
                # 结束轮
                next_turn_data = deepcopy(turn_data)
                next_turn_data['others']['turn'] = -1
                next_turn_data['user_action'] = {}
                next_turn_data['sys_action'] = turn_data['trg_sys_action']
                next_turn_data['trg_sys_action'] = {}
                next_turn_data['belief_state'] = turn_data['final_belief_state']
                # 统计next_s
                next_s.append(torch.Tensor(state_vectorize(next_turn_data, cfg, db, True)))
                # 统计奖励, 对于系统动作，判决任务是否完成作为最终奖励依据,
                # 系统是否完成了真实用户动作所提出的订阅请求，且系统是否回答了真实用户动作所咨询的所有问题
                reward = 20 if evaluator.task_success(False) else -5
                r.append(reward)
                # 结束标志位
                t.append(1)
            else:
                reward = 0
                if evaluator.cur_domain:
                    for slot, value in turn_data['belief_state'][evaluator.cur_domain].items():
                        if value == '?':
                            for da in turn_data['trg_sys_action']:
                                d, i, k, p = da.split('-')
                                if i in ['inform', 'recommend', 'offerbook', 'offerbooked'] and k == slot:
                                    break
                            else:
                                # not answer request
                                # 没有完成对belief_state中的提问，奖励减一
                                reward -= 1
                if not turn_data['trg_sys_action']:
                    # 本轮没有回复奖励减五
                    reward -= 5
                r.append(reward)
                t.append(0)

        torch.save((s, a, r, next_s, t), file_dir)

    def create_dataset_usr(self, part, file_dir, data_dir, cfg, db):
        """
        创建用户数据
        """
        datas = self.data[part]
        goals = self.goal[part]
        s, a, r, next_s, t = [], [], [], [], []
        evaluator = MultiWozEvaluator(data_dir, cfg.d)
        current_goal = None
        for idx, turn_data in enumerate(datas):
            if turn_data['others']['turn'] % 2 == 1:
                evaluator.add_sys_da(turn_data['trg_sys_action'])
                continue
            if turn_data['others']['turn'] == 0:
                current_goal = goals[turn_data['others']['session_id']]
                evaluator.add_goal(current_goal)
            else:
                # 错位了，确实记录的就是下一个状态
                next_s.append(s[-1])

            # 当用户目标无法满足时，切换用户目标
            if turn_data['others']['change'] and evaluator.cur_domain:
                if 'final' in current_goal[evaluator.cur_domain]:
                    for key in current_goal[evaluator.cur_domain]['final']:
                        current_goal[evaluator.cur_domain][key] = current_goal[evaluator.cur_domain]['final'][key]
                    del (current_goal[evaluator.cur_domain]['final'])
            turn_data['user_goal'] = deepcopy(current_goal)

            s.append(torch.Tensor(state_vectorize_user(turn_data, cfg, evaluator.cur_domain)))
            a.append(
                torch.Tensor(action_vectorize_user(turn_data['trg_user_action'], turn_data['others']['terminal'], cfg)))
            evaluator.add_usr_da(turn_data['trg_user_action'])
            if turn_data['others']['terminal']:
                next_turn_data = deepcopy(turn_data)
                next_turn_data['others']['turn'] = -1
                next_turn_data['user_action'] = turn_data['trg_user_action']
                next_turn_data['sys_action'] = datas[idx + 1]['trg_sys_action']
                next_turn_data['trg_user_action'] = {}
                next_turn_data['goal_state'] = datas[idx + 1]['final_goal_state']
                next_s.append(torch.Tensor(state_vectorize_user(next_turn_data, cfg, evaluator.cur_domain)))
                # 对于用户侧任务是否成功，采用inform_F1进行判决, 1表示的是召回率，就是用户目标的所有需求和提供都已经完成了
                reward = 20 if evaluator.inform_F1(ans_by_sys=False)[1] == 1. else -5
                r.append(reward)
                t.append(1)
            else:
                reward = 0
                if evaluator.cur_domain:
                    for da in turn_data['trg_user_action']:
                        d, i, k = da.split('-')
                        if i == 'request':
                            for slot, value in turn_data['goal_state'][d].items():
                                if value != '?' and slot in turn_data['user_goal'][d] \
                                        and turn_data['user_goal'][d][slot] != '?':
                                    # request before express constraint
                                    # 用户原本应该提供的信息，提前执行了request的操作，扣除一点奖励
                                    reward -= 1
                if not turn_data['trg_user_action']:
                    reward -= 5
                r.append(reward)
                t.append(0)

        torch.save((s, a, r, next_s, t), file_dir)

    def create_dataset_global(self, part, file_dir, data_dir, cfg, db):
        """
        创建global数据，这个数据记录了用户侧和系统侧的所有状态以及奖励
        """
        datas = self.data[part]
        goals = self.goal[part]
        s_usr, s_sys, r_g, next_s_usr, next_s_sys, t = [], [], [], [], [], []
        evaluator = MultiWozEvaluator(data_dir, cfg.d)
        for idx, turn_data in enumerate(datas):
            if turn_data['others']['turn'] % 2 == 0:
                if turn_data['others']['turn'] == 0:
                    current_goal = goals[turn_data['others']['session_id']]
                    evaluator.add_goal(current_goal)
                else:
                    next_s_usr.append(s_usr[-1])

                # 当用户目标无法满足时，切换用户目标
                if turn_data['others']['change'] and evaluator.cur_domain:
                    if 'final' in current_goal[evaluator.cur_domain]:
                        for key in current_goal[evaluator.cur_domain]['final']:
                            current_goal[evaluator.cur_domain][key] = current_goal[evaluator.cur_domain]['final'][key]
                        del (current_goal[evaluator.cur_domain]['final'])
                turn_data['user_goal'] = deepcopy(current_goal)

                s_usr.append(torch.Tensor(state_vectorize_user(turn_data, cfg, evaluator.cur_domain)))
                evaluator.add_usr_da(turn_data['trg_user_action'])

                if turn_data['others']['terminal']:
                    next_turn_data = deepcopy(turn_data)
                    next_turn_data['others']['turn'] = -1
                    next_turn_data['user_action'] = turn_data['trg_user_action']
                    next_turn_data['sys_action'] = datas[idx + 1]['trg_sys_action']
                    next_turn_data['trg_user_action'] = {}
                    next_turn_data['goal_state'] = datas[idx + 1]['final_goal_state']
                    next_s_usr.append(torch.Tensor(state_vectorize_user(next_turn_data, cfg, evaluator.cur_domain)))

            else:
                if turn_data['others']['turn'] != 1:
                    next_s_sys.append(s_sys[-1])

                s_sys.append(torch.Tensor(state_vectorize(turn_data, cfg, db, True)))
                evaluator.add_sys_da(turn_data['trg_sys_action'])

                if turn_data['others']['terminal']:
                    next_turn_data = deepcopy(turn_data)
                    next_turn_data['others']['turn'] = -1
                    next_turn_data['user_action'] = {}
                    next_turn_data['sys_action'] = turn_data['trg_sys_action']
                    next_turn_data['trg_sys_action'] = {}
                    next_turn_data['belief_state'] = turn_data['final_belief_state']
                    next_s_sys.append(torch.Tensor(state_vectorize(next_turn_data, cfg, db, True)))
                    # 由于多轮对话系统，默认最终都是系统说结束语，因此通过系统判断任务是否成功作为整体的奖励
                    reward_g = 20 if evaluator.task_success() else -5
                    r_g.append(reward_g)
                    t.append(1)
                else:
                    # 增加domain_success的奖励，其他的则每增加一轮减少一点损失，用于缩短轮数 todo 什么是 domain_success
                    reward_g = 5 if evaluator.cur_domain and evaluator.domain_success(evaluator.cur_domain) else -1
                    r_g.append(reward_g)
                    t.append(0)

        torch.save((s_usr, s_sys, r_g, next_s_usr, next_s_sys, t), file_dir)

    def create_dataset_policy(self, part, batchsz, cfg, db, character='sys'):
        """

        """
        assert part in ['train', 'valid', 'test']
        logging.debug('start loading {}'.format(part))

        if character == 'sys':
            file_dir = self.data_dir_new + '/' + part + '_sys.pt'
        elif character == 'usr':
            file_dir = self.data_dir_new + '/' + part + '_usr.pt'
        else:
            raise NotImplementedError('Unknown character {}'.format(character))

        # 加载训练数据, 只关心state和actions
        s, a, *_ = torch.load(file_dir)
        new_s, new_a = [], []
        for state, action in zip(s, a):
            if action.nonzero().size(0):
                new_s.append(state)
                new_a.append(action)
        # 得到Dataset_policy 对象, 简单封装了一层, 方便于torch的dataset调用
        dataset = Dataset_Policy(new_s, new_a)
        # 得到torch的数据加载器
        dataloader = data.DataLoader(dataset, batchsz, True)

        logging.debug('finish loading {}'.format(part))
        return dataloader

    def create_dataset_vnet(self, part, batchsz, cfg, db):
        """
        用于提供vnet的预训练，实际系统中没有使用到
        """
        assert part in ['train', 'valid', 'test']
        logging.debug('start loading {}'.format(part))

        file_dir_1 = self.data_dir_new + '/' + part + '_sys.pt'
        file_dir_2 = self.data_dir_new + '/' + part + '_usr.pt'
        file_dir_3 = self.data_dir_new + '/' + part + '_glo.pt'

        s, _, r, next_s, t = torch.load(file_dir_1)
        dataset_sys = Dataset_Vnet(s, r, next_s, t)
        dataloader_sys = data.DataLoader(dataset_sys, batchsz, True)

        s, _, r, next_s, t = torch.load(file_dir_2)
        dataset_usr = Dataset_Vnet(s, r, next_s, t)
        dataloader_usr = data.DataLoader(dataset_usr, batchsz, True)

        s_usr, s_sys, r_g, next_s_usr, next_s_sys, t = torch.load(file_dir_3)
        dataset_global = Dataset_Vnet_G(s_usr, s_sys, r_g, next_s_usr, next_s_sys, t)
        dataloader_global = data.DataLoader(dataset_global, batchsz, True)

        logging.debug('finish loading {}'.format(part))
        return dataloader_sys, dataloader_usr, dataloader_global


class Dataset_Policy(data.Dataset):
    """
    简单封装用于policy有监督预训练的训练数据
    """

    def __init__(self, s, a):
        self.s = s
        self.a = a
        self.num_total = len(s)

    def __getitem__(self, index):
        s = self.s[index]
        a = self.a[index]
        return s, a

    def __len__(self):
        return self.num_total


class Dataset_Vnet(data.Dataset):
    def __init__(self, s, r, next_s, t):
        self.s = s
        self.r = r
        self.next_s = next_s
        self.t = t
        self.num_total = len(s)

    def __getitem__(self, index):
        s = self.s[index]
        r = self.r[index]
        next_s = self.next_s[index]
        t = self.t[index]
        return s, r, next_s, t

    def __len__(self):
        return self.num_total


class Dataset_Vnet_G(data.Dataset):
    def __init__(self, s_usr, s_sys, r, next_s_usr, next_s_sys, t):
        self.s_usr = s_usr
        self.s_sys = s_sys
        self.r = r
        self.next_s_usr = next_s_usr
        self.next_s_sys = next_s_sys
        self.t = t
        self.num_total = len(s_sys)

    def __getitem__(self, index):
        s_usr = self.s_usr[index]
        s_sys = self.s_sys[index]
        r = self.r[index]
        next_s_usr = self.next_s_usr[index]
        next_s_sys = self.next_s_sys[index]
        t = self.t[index]
        return s_usr, s_sys, r, next_s_usr, next_s_sys, t

    def __len__(self):
        return self.num_total

# -*- coding: utf-8 -*-
"""
@author: keshuichonglx 
"""

import json
import random
from copy import deepcopy

import torch
from goal_generator import GoalGenerator
from tracker import StateTracker
from utils import init_session, init_goal

REF_USR_DA = {
    'Attraction': {
        'area': 'Area', 'type': 'Type', 'name': 'Name',
        'entrance fee': 'Fee', 'address': 'Addr',
        'postcode': 'Post', 'phone': 'Phone'
    },
    'Hospital': {
        'department': 'Department', 'address': 'Addr', 'postcode': 'Post',
        'phone': 'Phone'
    },
    'Hotel': {
        'type': 'Type', 'parking': 'Parking', 'pricerange': 'Price',
        'internet': 'Internet', 'area': 'Area', 'stars': 'Stars',
        'name': 'Name', 'stay': 'Stay', 'day': 'Day', 'people': 'People',
        'address': 'Addr', 'postcode': 'Post', 'phone': 'Phone'
    },
    'Police': {
        'address': 'Addr', 'postcode': 'Post', 'phone': 'Phone'
    },
    'Restaurant': {
        'food': 'Food', 'pricerange': 'Price', 'area': 'Area',
        'name': 'Name', 'time': 'Time', 'day': 'Day', 'people': 'People',
        'phone': 'Phone', 'postcode': 'Post', 'address': 'Addr'
    },
    'Taxi': {
        'leaveAt': 'Leave', 'destination': 'Dest', 'departure': 'Depart', 'arriveBy': 'Arrive',
        'car type': 'Car', 'phone': 'Phone'
    },
    'Train': {
        'destination': 'Dest', 'day': 'Day', 'arriveBy': 'Arrive',
        'departure': 'Depart', 'leaveAt': 'Leave', 'people': 'People',
        'duration': 'Time', 'price': 'Ticket', 'trainID': 'Id'
    }
}

REF_SYS_DA = {
    'Attraction': {
        'Addr': "address", 'Area': "area", 'Choice': "choice",
        'Fee': "entrance fee", 'Name': "name", 'Phone': "phone",
        'Post': "postcode", 'Price': "pricerange", 'Type': "type",
        'none': None, 'Open': None
    },
    'Hospital': {
        'Department': 'department', 'Addr': 'address', 'Post': 'postcode',
        'Phone': 'phone', 'none': None
    },
    'Booking': {
        'Day': 'day', 'Name': 'name', 'People': 'people',
        'Ref': 'ref', 'Stay': 'stay', 'Time': 'time',
        'none': None
    },
    'Hotel': {
        'Addr': "address", 'Area': "area", 'Choice': "choice",
        'Internet': "internet", 'Name': "name", 'Parking': "parking",
        'Phone': "phone", 'Post': "postcode", 'Price': "pricerange",
        'Ref': "ref", 'Stars': "stars", 'Type': "type",
        'none': None
    },
    'Restaurant': {
        'Addr': "address", 'Area': "area", 'Choice': "choice",
        'Name': "name", 'Food': "food", 'Phone': "phone",
        'Post': "postcode", 'Price': "pricerange", 'Ref': "ref",
        'none': None
    },
    'Taxi': {
        'Arrive': "arriveBy", 'Car': "car type", 'Depart': "departure",
        'Dest': "destination", 'Leave': "leaveAt", 'Phone': "phone",
        'none': None
    },
    'Train': {
        'Arrive': "arriveBy", 'Choice': "choice", 'Day': "day",
        'Depart': "departure", 'Dest': "destination", 'Id': "trainID",
        'Leave': "leaveAt", 'People': "people", 'Ref': "ref",
        'Time': "duration", 'none': None, 'Ticket': 'price',
    },
    'Police': {
        'Addr': "address", 'Post': "postcode", 'Phone': "phone"
    },
}

DEF_VAL_UNK = '?'  # Unknown
DEF_VAL_DNC = 'don\'t care'  # Do not care
DEF_VAL_NUL = 'none'  # for none
DEF_VAL_BOOKED = 'yes'  # for booked
DEF_VAL_NOBOOK = 'no'  # for booked
NOT_SURE_VALS = [DEF_VAL_UNK, DEF_VAL_DNC, DEF_VAL_NUL, DEF_VAL_NOBOOK]

# import reflect table
REF_USR_DA_M = deepcopy(REF_USR_DA)
REF_SYS_DA_M = {}
for dom, ref_slots in REF_SYS_DA.items():
    dom = dom.lower()
    REF_SYS_DA_M[dom] = {}
    for slot_a, slot_b in ref_slots.items():
        REF_SYS_DA_M[dom][slot_a.lower()] = slot_b
    REF_SYS_DA_M[dom]['none'] = None

# def book slot
BOOK_SLOT = ['people', 'day', 'stay', 'time']


class UserAgenda(StateTracker):
    """ 基于规则的用户行为策略代理 """

    def __init__(self, data_dir, cfg):
        super(UserAgenda, self).__init__(data_dir, cfg)
        # 最大对话轮数
        self.max_turn = 40
        # todo 最多连续执行动作数 (待确认)
        self.max_initiative = 4

        # ontology_file = value_set.json 只在这里使用
        # value_set 记录了所有domain的所有slot对应的值
        with open(data_dir + '/' + cfg.ontology_file) as f:
            self.stand_value_dict = json.load(f)

        # 根据 goal_model.pkl 构建用户目标生成器，
        # goal_model.pkl 中记录了真实数据中的 domain，book，slot， slot-value的出现概率
        self.goal_generator = GoalGenerator(data_dir, cfg,
                                            goal_model_path='processed_data/goal_model.pkl',
                                            corpus_path=cfg.data_file)

        self.goal = None
        self.agenda = None

    def _action_to_dict(self, das):
        """
        将 dialog-actions 转化为 dict 的形式
        dialog-actions 形式为 domain-intent-slot-p， p为占位符（数字，符号）
        dict 形式为 {domain-intent:[[slot,value]]}
        """
        da_dict = {}
        for da, value in das.items():
            domain, intent, slot, p = da.split('-')
            domint = '-'.join((domain, intent))
            if domint not in da_dict:
                da_dict[domint] = []
            da_dict[domint].append([slot, value])
        return da_dict

    def _dict_to_vec(self, das):
        """
        将 dict 形式转化为 vec格式
        dict 为上面定义的形式
        vec 为预先定义好的用户行为向量
        """

        # 根据配置文件中的 a_dim_usr 预先将 vec 定义为全零向量
        da_vector = torch.zeros(self.cfg.a_dim_usr, dtype=torch.int32)
        for domint in das:
            pairs = das[domint]
            for slot, value in pairs:
                # 组合成 domain-intent-slot 形式
                da = '-'.join((domint, slot)).lower()
                if da in self.cfg.da2idx_u:
                    idx = self.cfg.da2idx_u[da]
                    # 由于动作空间是预先定义好的，因此直接对对应的index置1即可
                    da_vector[idx] = 1
        return da_vector

    def reset(self, random_seed=None):
        """
        为下一段对话创建用户目标 goal 和代理器 agenda
        """
        self.time_step = 0
        # 当前对话的主题，这里就是指 domain
        self.topic = ''

        # 创建用户目标，需要使用到目标生成器
        self.goal = Goal(self.goal_generator, seed=random_seed)
        # 创建用户代理器
        self.agenda = Agenda(self.goal)

        # 初始化 状态和目标
        # 状态采用 dict 形式，区别于 向量形式状态
        # 状态中包含 下一个domain， 下一个之后的所有domain， 用户目标，目标状态等
        dummy_state, dummy_goal = init_session(-1, self.cfg)
        init_goal(dummy_goal, dummy_state['goal_state'], self.goal.domain_goals, self.cfg)
        domain_ordering = self.goal.domains
        dummy_state['next_available_domain'] = domain_ordering[0]
        dummy_state['invisible_domains'] = domain_ordering[1:]
        dummy_state['user_goal'] = dummy_goal

        # 将初始的用户目标加入到 evaluator 中，用于评价该用户目标是否完成
        self.evaluator.add_goal(dummy_goal)

        # 默认为用户先说话，因此先生成用户的动作
        usr_a, terminal = self.predict(None, {})
        usr_a = self._dict_to_vec(usr_a)
        usr_a[-1] = 1 if terminal else 0
        # 并通过用户的动作 更新初始的状态
        init_state = self.update_belief_usr(dummy_state, usr_a)
        return init_state

    def step(self, s, sys_a):
        """
        接收来自系统方的动作，执行一个回合的用户侧动作
        """
        # 根据系统方的动作，更新状态
        current_s = self.update_belief_sys(s, sys_a)
        if current_s['others']['terminal']:
            # 在上个用户侧对话时，用户已经结束的会话, 则将terminal信号设置为True， 同时清空用户动作向量
            usr_a, terminal = torch.zeros(self.cfg.a_dim_usr, dtype=torch.int32), True
        else:
            # todo 这里的 sys_action 和 sys_a 有什么区别？
            da_dict = self._action_to_dict(current_s['sys_action'])
            # 这里的状态是通过 agenda代理器来维护的所以设置为None
            usr_a, terminal = self.predict(None, da_dict)
            usr_a = self._dict_to_vec(usr_a)

        # 更新系统状态
        usr_a[-1] = 1 if terminal else 0
        next_s = self.update_belief_usr(current_s, usr_a)
        return next_s, terminal

    def predict(self, state, sys_action):
        """
        根据预定义好的系统动作预测和状态，预测用户动作
        输入为{domain-intent:[[slot,slot-value]]}
        输出为{Domain-Intent:[[REF_USR_DA_M~slot, slot-value]]
        """
        if self.time_step >= self.max_turn:
            self.agenda.close_session()
        else:
            # 将sys_action 转化为 agenda可以读取的格式
            sys_action = self._transform_sysact_in(sys_action)
            # 根据系统动作和用户目标，更新代理器
            self.agenda.update(sys_action, self.goal)
            if self.goal.task_complete():
                self.agenda.close_session()

        # A -> A' + user_action
        # 根据用户代理器当前记录的状态得到对应的用户动作
        action = self.agenda.get_action(self.max_initiative)

        # Is there any action to say?
        # 如果代理器没有待执行的任务，则会话终止
        session_over = self.agenda.is_empty()

        # transform to DA
        # 将action 转化为 正常的形式
        action = self._transform_usract_out(action)

        return action, session_over

    def _transform_usract_out(self, action):
        """
        将agenda生成的用户动作转化为输出形式
        用户动作形式 {domain-intent:[[slot, slot-value]]
        转化之后的形式 {Domain-Intent:[[REF_USR_DA_M~slot, slot-value]]
        """
        new_action = {}
        for act in action.keys():
            if '-' in act:
                # general domain 另作了处理， REF中也没有进行定义
                if 'general' not in act:
                    (dom, intent) = act.split('-')
                    new_act = dom.capitalize() + '-' + intent.capitalize()
                    new_action[new_act] = []
                    for pairs in action[act]:
                        slot = REF_USR_DA_M[dom.capitalize()].get(pairs[0], None)
                        if slot is not None:
                            new_action[new_act].append([slot, pairs[1]])
                else:
                    new_action[act] = action[act]
            else:
                pass
        return new_action

    def _transform_sysact_in(self, action):
        """
        将系统方生成的动作转化为agenda读取的格式
        系统动作形式 {domain-intent:[[slot,slot-value]]}
        转化之后的形式{domain-intent:[[REF_SYS_DA_M~slot, normalized-value]]}
        """
        new_action = {}
        if not isinstance(action, dict):
            print('illegal da:', action)
            return new_action

        for act in action.keys():
            if not isinstance(act, str) or '-' not in act:
                print('illegal act: %s' % act)
                continue

            if 'general' not in act:
                (dom, intent) = act.lower().split('-')
                if dom in REF_SYS_DA_M.keys():
                    new_list = []
                    for pairs in action[act]:
                        if (not isinstance(pairs, list) and not isinstance(pairs, tuple)) or \
                                (len(pairs) < 2) or \
                                (not isinstance(pairs[0], str) or not isinstance(pairs[1], str)):
                            print('illegal pairs:', pairs)
                            continue

                        if REF_SYS_DA_M[dom].get(pairs[0].lower(), None) is not None:
                            # slot 转化为 REF_SYS_DA_M~slot，
                            # slot-value 转化为 normalize之后的值
                            new_list.append([REF_SYS_DA_M[dom][pairs[0].lower()],
                                             self._normalize_value(dom, intent, REF_SYS_DA_M[dom][pairs[0].lower()],
                                                                   pairs[1])])

                    if len(new_list) > 0:
                        new_action[act.lower()] = new_list
            else:
                # 对于general的动作，只是将对应的act，进行了转小写,实际上也应该是小写形式
                new_action[act.lower()] = action[act]

        return new_action

    def _normalize_value(self, domain, intent, slot, value):
        """
        对value进行后处理, 针对stand_value对value进行细微的调整
        """
        # 对所有的request 统一输出？符号作为value
        if intent == 'request':
            return DEF_VAL_UNK

        # 如果domain或者slot不在stand_value_dict中，则直接返回
        if domain not in self.stand_value_dict.keys():
            return value
        if slot not in self.stand_value_dict[domain]:
            return value

        # 如果domain为taxi且slot为phone，直接返回
        if domain == 'taxi' and slot == 'phone':
            return value

        # 如果domain和slot均在stand_value_dict中，但是value不在对应的value_list中
        # 首先检查是否为格式上不一致，对于格式不一致的，统一为标准格式
        # 否则直接返回value
        value_list = self.stand_value_dict[domain][slot]
        if value not in value_list and value != 'none':
            v0 = ' '.join(value.split())
            v0N = ''.join(value.split())
            for val in value_list:
                v1 = ' '.join(val.split())
                if v0 in v1 or v1 in v0 or v0N in v1 or v1 in v0N:
                    return v1
            print('illegal value: %s, slot: %s domain: %s' % (value, slot, domain))
        return value


def check_constraint(slot, val_usr, val_sys):
    """
    判断系统动作是否满足用户动作的约束，
    满足返回False，如果用户动作和系统动作不一致，则返回True
    """
    try:
        if slot == 'arriveBy':
            val1 = int(val_usr.split(':')[0]) * 100 + int(val_usr.split(':')[1])
            val2 = int(val_sys.split(':')[0]) * 100 + int(val_sys.split(':')[1])
            if val1 < val2:
                return True
        elif slot == 'leaveAt':
            val1 = int(val_usr.split(':')[0]) * 100 + int(val_usr.split(':')[1])
            val2 = int(val_sys.split(':')[0]) * 100 + int(val_sys.split(':')[1])
            if val1 > val2:
                return True
        else:
            if val_usr != val_sys:
                return True
        return False
    except:
        return False


class Goal(object):
    """ User Goal Model Class. """

    def __init__(self, goal_generator: GoalGenerator, seed=None):
        """
        create new Goal by random
        Args:
            goal_generator (GoalGenerator): Goal Gernerator.
        """
        # 随机生成用户目标
        self.domain_goals = goal_generator.get_user_goal(seed)
        # 单独拎出来domains
        self.domains = list(self.domain_goals['domain_ordering'])
        del self.domain_goals['domain_ordering']

        for domain in self.domains:
            # 对目标中的reqt，由list转化为dict形式，value为DEF_VAL_UNK符号
            if 'reqt' in self.domain_goals[domain].keys():
                self.domain_goals[domain]['reqt'] = {slot: DEF_VAL_UNK for slot in self.domain_goals[domain]['reqt']}
            # 如果目标中存在book，则新增booked属性为DEF_VAL_UNK符号
            if 'book' in self.domain_goals[domain].keys():
                self.domain_goals[domain]['booked'] = DEF_VAL_UNK

    def task_complete(self):
        """
        用户目标中的reqt和booked均已经赋值，表示任务已经完成
        Returns:
            (boolean): True to accomplish.
        """
        for domain in self.domains:
            if 'reqt' in self.domain_goals[domain]:
                reqt_vals = self.domain_goals[domain]['reqt'].values()
                for val in reqt_vals:
                    if val in NOT_SURE_VALS:
                        return False

            if 'booked' in self.domain_goals[domain]:
                if self.domain_goals[domain]['booked'] in NOT_SURE_VALS:
                    return False
        return True

    def next_domain_incomplete(self):
        """
        判断下一个回合是否需要切换主题domain
        """
        # 按照domains的顺序依次判断是否需要切换主题
        for domain in self.domains:
            # 如果当前主题的用户请求还没有完成，则继续返回当前domain，
            # 意图为'reqt',
            # slot中如果name存在未知请求列表中，优先询问name，否则返回未知列表
            if 'reqt' in self.domain_goals[domain]:
                requests = self.domain_goals[domain]['reqt']
                unknow_reqts = [key for (key, val) in requests.items() if val in NOT_SURE_VALS]
                if len(unknow_reqts) > 0:
                    return domain, 'reqt', ['name'] if 'name' in unknow_reqts else unknow_reqts

            # 如果需求已经完成，还没有进入booked状态，则返回book内容，
            # 如果存在fail_book 则优先返回 fail_book
            # fail_book 表示当前预定会失败，失败之后会选择book返回
            if 'booked' in self.domain_goals[domain]:
                if self.domain_goals[domain]['booked'] in NOT_SURE_VALS:
                    return domain, 'book', \
                           self.domain_goals[domain]['fail_book'] if 'fail_book' in self.domain_goals[
                               domain].keys() else self.domain_goals[domain]['book']
        # 所有任务均已完成，返回三个NONE
        return None, None, None


class Agenda(object):
    def __init__(self, goal: Goal):
        """
        通过用户目标构建Agenda
        """

        def random_sample(data, minimum=0, maximum=1000):
            return random.sample(data, random.randint(min(len(data), minimum), min(len(data), maximum)))

        self.CLOSE_ACT = 'general-bye'
        self.HELLO_ACT = 'general-greet'
        self.__cur_push_num = 0

        self.__stack = []

        # there is a 'bye' action at the bottom of the stack
        self.__push(self.CLOSE_ACT)

        for idx in range(len(goal.domains) - 1, -1, -1):
            domain = goal.domains[idx]

            # inform
            if 'fail_info' in goal.domain_goals[domain]:
                for slot in random_sample(goal.domain_goals[domain]['fail_info'].keys(),
                                          len(goal.domain_goals[domain]['fail_info'])):
                    self.__push(domain + '-inform', slot, goal.domain_goals[domain]['fail_info'][slot])
            elif 'info' in goal.domain_goals[domain]:
                for slot in random_sample(goal.domain_goals[domain]['info'].keys(),
                                          len(goal.domain_goals[domain]['info'])):
                    self.__push(domain + '-inform', slot, goal.domain_goals[domain]['info'][slot])

        self.cur_domain = None

    def update(self, sys_action, goal: Goal):
        """
        update Goal by current agent action and current goal. { A' + G" + sys_action -> A" }
        Args:
            sys_action (tuple): Preorder system action.s
            goal (Goal): User Goal
        """
        self.__cur_push_num = 0
        self._update_current_domain(sys_action, goal)

        for diaact in sys_action.keys():
            slot_vals = sys_action[diaact]
            if 'nooffer' in diaact:
                if self.update_domain(diaact, slot_vals, goal):
                    return
            elif 'nobook' in diaact:
                if self.update_booking(diaact, slot_vals, goal):
                    return

        for diaact in sys_action.keys():
            if 'nooffer' in diaact or 'nobook' in diaact:
                continue

            slot_vals = sys_action[diaact]
            if 'booking' in diaact:
                if self.update_booking(diaact, slot_vals, goal):
                    return
            elif 'general' in diaact:
                if self.update_general(diaact, slot_vals, goal):
                    return
            else:
                if self.update_domain(diaact, slot_vals, goal):
                    return

        unk_dom, unk_type, data = goal.next_domain_incomplete()
        if unk_dom is not None:
            if unk_type == 'reqt' and not self._check_reqt_info(unk_dom) and not self._check_reqt(unk_dom):
                for slot in data:
                    self._push_item(unk_dom + '-request', slot, DEF_VAL_UNK)
            elif unk_type == 'book' and not self._check_reqt_info(unk_dom) and not self._check_book_info(unk_dom):
                for (slot, val) in data.items():
                    self._push_item(unk_dom + '-inform', slot, val)

    def update_booking(self, diaact, slot_vals, goal: Goal):
        """
        Handel Book-XXX
        :param diaact:      Dial-Act
        :param slot_vals:   slot value pairs
        :param goal:        Goal
        :return:            True:user want to close the session. False:session is continue
        """
        _, intent = diaact.split('-')
        domain = self.cur_domain

        if domain not in goal.domains:
            return False

        g_reqt = goal.domain_goals[domain].get('reqt', dict({}))
        g_info = goal.domain_goals[domain].get('info', dict({}))
        g_fail_info = goal.domain_goals[domain].get('fail_info', dict({}))
        g_book = goal.domain_goals[domain].get('book', dict({}))
        g_fail_book = goal.domain_goals[domain].get('fail_book', dict({}))

        if intent in ['book', 'inform']:
            info_right = True
            for [slot, value] in slot_vals:
                if domain == 'train' and slot == 'time':
                    slot = 'duration'

                if slot in g_reqt:
                    if not self._check_reqt_info(domain):
                        self._remove_item(domain + '-request', slot)
                        if value in NOT_SURE_VALS:
                            g_reqt[slot] = '\"' + value + '\"'
                        else:
                            g_reqt[slot] = value

                elif slot in g_fail_info and value != g_fail_info[slot]:
                    self._push_item(domain + '-inform', slot, g_fail_info[slot])
                    info_right = False
                elif len(g_fail_info) <= 0 and slot in g_info and check_constraint(slot, g_info[slot], value):
                    self._push_item(domain + '-inform', slot, g_info[slot])
                    info_right = False

                elif slot in g_fail_book and value != g_fail_book[slot]:
                    self._push_item(domain + '-inform', slot, g_fail_book[slot])
                    info_right = False
                elif len(g_fail_book) <= 0 and slot in g_book and value != g_book[slot]:
                    self._push_item(domain + '-inform', slot, g_book[slot])
                    info_right = False

                else:
                    pass

            if intent == 'book' and info_right:
                # booked ok
                if 'booked' in goal.domain_goals[domain]:
                    goal.domain_goals[domain]['booked'] = DEF_VAL_BOOKED
                self._push_item('general-thank')

        elif intent in ['nobook']:
            if len(g_fail_book) > 0:
                # Discard fail_book data and update the book data to the stack
                for slot in g_book.keys():
                    if (slot not in g_fail_book) or (slot in g_fail_book and g_fail_book[slot] != g_book[slot]):
                        self._push_item(domain + '-inform', slot, g_book[slot])

                # change fail_info name
                goal.domain_goals[domain]['fail_book_fail'] = goal.domain_goals[domain].pop('fail_book')
            elif 'booked' in goal.domain_goals[domain].keys():
                self.close_session()
                return True

        elif intent in ['request']:
            for [slot, _] in slot_vals:
                if domain == 'train' and slot == 'time':
                    slot = 'duration'

                if slot in g_reqt:
                    pass
                elif slot in g_fail_info:
                    self._push_item(domain + '-inform', slot, g_fail_info[slot])
                elif len(g_fail_info) <= 0 and slot in g_info:
                    self._push_item(domain + '-inform', slot, g_info[slot])

                elif slot in g_fail_book:
                    self._push_item(domain + '-inform', slot, g_fail_book[slot])
                elif len(g_fail_book) <= 0 and slot in g_book:
                    self._push_item(domain + '-inform', slot, g_book[slot])

                else:

                    if domain == 'taxi' and (slot == 'destination' or slot == 'departure'):
                        places = [dom for dom in goal.domains[: goal.domains.index('taxi')] if
                                  'address' in goal.domain_goals[dom]['reqt']]

                        if len(places) >= 1 and slot == 'destination' and \
                                goal.domain_goals[places[-1]]['reqt']['address'] not in NOT_SURE_VALS:
                            self._push_item(domain + '-inform', slot, goal.domain_goals[places[-1]]['reqt']['address'])

                        elif len(places) >= 2 and slot == 'departure' and \
                                goal.domain_goals[places[-2]]['reqt']['address'] not in NOT_SURE_VALS:
                            self._push_item(domain + '-inform', slot, goal.domain_goals[places[-2]]['reqt']['address'])

                        elif random.random() < 0.5:
                            self._push_item(domain + '-inform', slot, DEF_VAL_DNC)

                    elif random.random() < 0.5:
                        self._push_item(domain + '-inform', slot, DEF_VAL_DNC)

        return False

    def update_domain(self, diaact, slot_vals, goal: Goal):
        """
        Handel Domain-XXX
        :param diaact:      Dial-Act
        :param slot_vals:   slot value pairs
        :param goal:        Goal
        :return:            True:user want to close the session. False:session is continue
        """
        domain, intent = diaact.split('-')

        if domain not in goal.domains:
            return False

        g_reqt = goal.domain_goals[domain].get('reqt', dict({}))
        g_info = goal.domain_goals[domain].get('info', dict({}))
        g_fail_info = goal.domain_goals[domain].get('fail_info', dict({}))
        g_book = goal.domain_goals[domain].get('book', dict({}))
        g_fail_book = goal.domain_goals[domain].get('fail_book', dict({}))

        if intent in ['inform', 'recommend', 'offerbook', 'offerbooked']:
            info_right = True
            for [slot, value] in slot_vals:
                if slot in g_reqt:
                    if not self._check_reqt_info(domain):
                        self._remove_item(domain + '-request', slot)
                        if value in NOT_SURE_VALS:
                            g_reqt[slot] = '\"' + value + '\"'
                        else:
                            g_reqt[slot] = value

                elif slot in g_fail_info and value != g_fail_info[slot]:
                    self._push_item(domain + '-inform', slot, g_fail_info[slot])
                    info_right = False
                elif len(g_fail_info) <= 0 and slot in g_info and check_constraint(slot, g_info[slot], value):
                    self._push_item(domain + '-inform', slot, g_info[slot])
                    info_right = False

                elif slot in g_fail_book and value != g_fail_book[slot]:
                    self._push_item(domain + '-inform', slot, g_fail_book[slot])
                    info_right = False
                elif len(g_fail_book) <= 0 and slot in g_book and value != g_book[slot]:
                    self._push_item(domain + '-inform', slot, g_book[slot])
                    info_right = False

                else:
                    pass

            if intent == 'offerbooked' and info_right:
                # booked ok
                if 'booked' in goal.domain_goals[domain]:
                    goal.domain_goals[domain]['booked'] = DEF_VAL_BOOKED
                self._push_item('general-thank')

        elif intent in ['request']:
            for [slot, _] in slot_vals:
                if slot in g_reqt:
                    pass
                elif slot in g_fail_info:
                    self._push_item(domain + '-inform', slot, g_fail_info[slot])
                elif len(g_fail_info) <= 0 and slot in g_info:
                    self._push_item(domain + '-inform', slot, g_info[slot])

                elif slot in g_fail_book:
                    self._push_item(domain + '-inform', slot, g_fail_book[slot])
                elif len(g_fail_book) <= 0 and slot in g_book:
                    self._push_item(domain + '-inform', slot, g_book[slot])

                else:

                    if domain == 'taxi' and (slot == 'destination' or slot == 'departure'):
                        places = [dom for dom in goal.domains[: goal.domains.index('taxi')] if
                                  'address' in goal.domain_goals[dom]['reqt']]

                        if len(places) >= 1 and slot == 'destination' and \
                                goal.domain_goals[places[-1]]['reqt']['address'] not in NOT_SURE_VALS:
                            self._push_item(domain + '-inform', slot, goal.domain_goals[places[-1]]['reqt']['address'])

                        elif len(places) >= 2 and slot == 'departure' and \
                                goal.domain_goals[places[-2]]['reqt']['address'] not in NOT_SURE_VALS:
                            self._push_item(domain + '-inform', slot, goal.domain_goals[places[-2]]['reqt']['address'])

                        elif random.random() < 0.5:
                            self._push_item(domain + '-inform', slot, DEF_VAL_DNC)

                    elif random.random() < 0.5:
                        self._push_item(domain + '-inform', slot, DEF_VAL_DNC)

        elif intent in ['nooffer']:
            if len(g_fail_info) > 0:
                # update info data to the stack
                for slot in g_info.keys():
                    if (slot not in g_fail_info) or (slot in g_fail_info and g_fail_info[slot] != g_info[slot]):
                        self._push_item(domain + '-inform', slot, g_info[slot])

                # change fail_info name
                goal.domain_goals[domain]['fail_info_fail'] = goal.domain_goals[domain].pop('fail_info')
            elif len(g_reqt.keys()) > 0:
                self.close_session()
                return True

        elif intent in ['select']:
            # delete Choice
            slot_vals = [[slot, val] for [slot, val] in slot_vals if slot != 'choice']

            if len(slot_vals) > 0:
                slot = slot_vals[0][0]

                if slot in g_fail_info:
                    self._push_item(domain + '-inform', slot, g_fail_info[slot])
                elif len(g_fail_info) <= 0 and slot in g_info:
                    self._push_item(domain + '-inform', slot, g_info[slot])

                elif slot in g_fail_book:
                    self._push_item(domain + '-inform', slot, g_fail_book[slot])
                elif len(g_fail_book) <= 0 and slot in g_book:
                    self._push_item(domain + '-inform', slot, g_book[slot])

                else:
                    if not self._check_reqt_info(domain):
                        [slot, value] = random.choice(slot_vals)
                        self._push_item(domain + '-inform', slot, value)

                        if slot in g_reqt:
                            self._remove_item(domain + '-request', slot)
                            g_reqt[slot] = value

        return False

    def update_general(self, diaact, slot_vals, goal: Goal):
        domain, intent = diaact.split('-')

        if intent == 'bye':
            pass
        elif intent == 'greet':
            pass
        elif intent == 'reqmore':
            pass
        elif intent == 'welcome':
            pass

        return False

    def close_session(self):
        """ Clear up all actions """
        self.__stack = []
        self.__push(self.CLOSE_ACT)

    def get_action(self, initiative=1):
        """
        get multiple acts based on initiative
        Args:
            initiative (int): number of slots , just for 'inform'
        Returns:
            action (dict): user diaact
        """
        diaacts, slots, values = self.__pop(initiative)
        action = {}
        for (diaact, slot, value) in zip(diaacts, slots, values):
            if diaact not in action.keys():
                action[diaact] = []
            action[diaact].append([slot, value])

        return action

    def is_empty(self):
        """
        Is the agenda already empty
        Returns:
            (boolean): True for empty, False for not.
        """
        return len(self.__stack) <= 0

    def _update_current_domain(self, sys_action, goal: Goal):
        for diaact in sys_action.keys():
            domain, _ = diaact.split('-')
            if domain in goal.domains:
                self.cur_domain = domain

    def _remove_item(self, diaact, slot=DEF_VAL_UNK):
        for idx in range(len(self.__stack)):
            if 'general' in diaact:
                if self.__stack[idx]['diaact'] == diaact:
                    self.__stack.remove(self.__stack[idx])
                    break
            else:
                if self.__stack[idx]['diaact'] == diaact and self.__stack[idx]['slot'] == slot:
                    self.__stack.remove(self.__stack[idx])
                    break

    def _push_item(self, diaact, slot=DEF_VAL_NUL, value=DEF_VAL_NUL):
        self._remove_item(diaact, slot)
        self.__push(diaact, slot, value)
        self.__cur_push_num += 1

    def _check_item(self, diaact, slot=None):
        for idx in range(len(self.__stack)):
            if slot is None:
                if self.__stack[idx]['diaact'] == diaact:
                    return True
            else:
                if self.__stack[idx]['diaact'] == diaact and self.__stack[idx]['slot'] == slot:
                    return True
        return False

    def _check_reqt(self, domain):
        for idx in range(len(self.__stack)):
            if self.__stack[idx]['diaact'] == domain + '-request':
                return True
        return False

    def _check_reqt_info(self, domain):
        for idx in range(len(self.__stack)):
            if self.__stack[idx]['diaact'] == domain + '-inform' and self.__stack[idx]['slot'] not in BOOK_SLOT:
                return True
        return False

    def _check_book_info(self, domain):
        for idx in range(len(self.__stack)):
            if self.__stack[idx]['diaact'] == domain + '-inform' and self.__stack[idx]['slot'] in BOOK_SLOT:
                return True
        return False

    def __check_next_diaact_slot(self):
        if len(self.__stack) > 0:
            return self.__stack[-1]['diaact'], self.__stack[-1]['slot']
        return None, None

    def __check_next_diaact(self):
        if len(self.__stack) > 0:
            return self.__stack[-1]['diaact']
        return None

    def __push(self, diaact, slot=DEF_VAL_NUL, value=DEF_VAL_NUL):
        self.__stack.append({'diaact': diaact, 'slot': slot, 'value': value})

    def __pop(self, initiative=1):
        diaacts = []
        slots = []
        values = []

        p_diaact, p_slot = self.__check_next_diaact_slot()
        if p_diaact.split('-')[1] == 'inform' and p_slot in BOOK_SLOT:
            for _ in range(10 if self.__cur_push_num == 0 else self.__cur_push_num):
                try:
                    item = self.__stack.pop(-1)
                    diaacts.append(item['diaact'])
                    slots.append(item['slot'])
                    values.append(item['value'])

                    cur_diaact = item['diaact']

                    next_diaact, next_slot = self.__check_next_diaact_slot()
                    if next_diaact is None or \
                            next_diaact != cur_diaact or \
                            next_diaact.split('-')[1] != 'inform' or next_slot not in BOOK_SLOT:
                        break
                except:
                    break
        else:
            for _ in range(initiative if self.__cur_push_num == 0 else self.__cur_push_num):
                try:
                    item = self.__stack.pop(-1)
                    diaacts.append(item['diaact'])
                    slots.append(item['slot'])
                    values.append(item['value'])

                    cur_diaact = item['diaact']

                    next_diaact = self.__check_next_diaact()
                    if next_diaact is None or \
                            next_diaact != cur_diaact or \
                            (cur_diaact.split('-')[1] == 'request' and item['slot'] == 'name'):
                        break
                except:
                    break

        return diaacts, slots, values

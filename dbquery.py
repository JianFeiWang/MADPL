import json
import random
import numpy as np


class DBQuery():

    def __init__(self, data_dir, cfg):
        # 加载各个domain的数据库信息
        self.cfg = cfg
        self.dbs = {}
        for domain in cfg.belief_domains:
            # belief_domains 记录的都有数据库数据
            with open('{}/{}_db.json'.format(data_dir, domain)) as f:
                self.dbs[domain] = json.load(f)

    def query(self, domain, constraints, ignore_open=True):
        """
        根据constraints请求domain的数据
        返回列表，列表中唯美一条符合要求的数据，数据采用字典形式, 且除了taxi，hospital，police，一般domain的记录中均包含ref字段
        """
        # taxi， hospital， police 三个domain的数据信息收集的不是很完整，因此这里采用比较直接的方式。
        if domain == 'taxi':
            # taxi 的数据仅限于 taxi_type 和 taxi_phone, 这里通过随机组合的方式随机生成
            return [{'taxi_type': random.choice(self.dbs[domain]['taxi_colors']) + ' ' + random.choice(
                self.dbs[domain]['taxi_types']),
                     'taxi_phone': ''.join([str(random.randint(0, 9)) for _ in range(10)])}]
        elif domain == 'hospital':
            return self.dbs['hospital']
        elif domain == 'police':
            return self.dbs['police']

        found = []
        for i, record in enumerate(self.dbs[domain]):
            for key, val in constraints:
                # 忽略 不关心的 slot
                if val == "" or val == "dont care" or val == 'not mentioned' or val == "don't care" or val == "dontcare" or val == "do n't care":
                    pass
                else:
                    try:
                        record_keys = [key.lower() for key in record]
                        # 对于没有数据库中没记录的slot不需要处理
                        if key.lower() not in record_keys:
                            continue
                        # 对于有时间要求的进行特殊处理
                        if key == 'leaveAt':
                            val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                            val2 = int(record['leaveAt'].split(':')[0]) * 100 + int(record['leaveAt'].split(':')[1])
                            if val1 > val2:
                                break
                        elif key == 'arriveBy':
                            val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                            val2 = int(record['arriveBy'].split(':')[0]) * 100 + int(record['arriveBy'].split(':')[1])
                            if val1 < val2:
                                break
                        elif ignore_open and key in ['destination', 'departure']:
                            # 假设出租车是可以从任意地点出发去往任意地点的
                            continue
                        else:
                            if val.strip() != record[key].strip():
                                break
                    except:
                        continue
            else:
                # 以上所有限制都满足，为该记录添加标志, 统计所有满足限制的数据
                record['ref'] = f'{domain}-{i:08d}'
                found.append(record)

        return found

    def pointer(self, turn, mapping, db_domains, requestable, noisy):
        """Create database pointer for all related domains."""
        # 对应指示向量，反应数据库中符合要求数据的量级
        # 实体熵 反应各个实体的分布情况
        pointer_vector = np.zeros(6 * len(db_domains))
        # 对应所有可以请求的domain-slot
        entropy = np.zeros(len(requestable))
        for domain in db_domains:
            constraint = []
            for k, v in turn[domain].items():
                # 通过 turn 数据，查询对话中的限制
                if k in mapping[domain] and v != '?':
                    constraint.append((mapping[domain][k], v))
            # 查询所有满足要求的数据
            entities = self.query(domain, constraint, noisy)
            # 计算 pointer_vector 和 entropy
            #
            pointer_vector = self.one_hot_vector(len(entities), domain, pointer_vector, db_domains)
            entropy = self.calc_slot_entropy(entities, domain, entropy, requestable)

        return pointer_vector, entropy

    def one_hot_vector(self, num, domain, vector, db_domains):
        """Return number of available entities for particular domain."""
        # 如果 domain 不是 train， 则直接根据返回的数据个数，设置指示向量
        if domain != 'train':
            idx = db_domains.index(domain)
            if num == 0:
                vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
            elif num == 1:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
            elif num == 2:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
            elif num == 3:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
            elif num == 4:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
            elif num >= 5:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])
        else:
            # 对于train domain， 由于检测出来的班次可能比较多，因此按照一定范围进行量化
            idx = db_domains.index(domain)
            if num == 0:
                vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
            elif num <= 2:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
            elif num <= 5:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
            elif num <= 10:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
            elif num <= 40:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
            elif num > 40:
                vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])

        return vector

    def calc_slot_entropy(self, entities, domain, vector, requestable):
        """Calculate entropy of requestable slot values in results"""
        N = len(entities)
        if not N:
            return vector

        # Count the values
        value_probabilities = {}
        # 计算所有符合要求记录中，各个slot-value出现的次数
        for index, entity in enumerate(entities):
            if index == 0:
                for key, value in entity.items():
                    if key in self.cfg.map_inverse[domain] and \
                            domain + '-' + self.cfg.map_inverse[domain][key] in requestable:
                        value_probabilities[key] = {value: 1}
            else:
                for key, value in entity.items():
                    if key in value_probabilities:
                        if value not in value_probabilities[key]:
                            value_probabilities[key][value] = 1
                        else:
                            value_probabilities[key][value] += 1

        # Calculate entropies
        # 按照slot 计算熵值
        for key in value_probabilities:
            entropy = 0
            for count in value_probabilities[key].values():
                entropy -= count / N * np.log(count / N)
            vector[requestable.index(domain + '-' + self.cfg.map_inverse[domain][key])] = entropy

        return vector

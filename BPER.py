import os
import numpy as np
import numpy.random as rd
import torch






class ReplayBuffer:  # for off-policy
    def __init__(self, args):
        
        pass
    
    def update_now_len(self):
        """ This function is designed to assess the current quantity of experience available. """
        self.now_len = self.max_len if self.if_full else self.next_idx
        
    def td_error_update(self, td_error):
        self.per_tree.td_error_update(td_error)
    
    def update_priority(self, args, cycle = 1):
        """ This is the primary function for the BPER algorithm to update the experience priorities. """
        self.update_now_len()
        
        state_dim = self.state_dim
        states = self.buf_state[:self.now_len,0:state_dim]
        for i in range(state_dim):
            states1 = states[:,i]
            if(np.max(states1) == np.min(states1)):
                continue
            bins = np.linspace(np.min(states1), np.max(states1), args.histogram_bins + 1)
            
            index = np.digitize(states1, bins) - 1
            for j in range(args.histogram_bins - 1):
                a = np.argwhere(index == j)[:,0]
                self.buf_priority[a,i] = a.shape[0] / self.now_len
            a = np.argwhere(index >= args.histogram_bins - 1)[:,0]
            self.buf_priority[a,i] = a.shape[0] / self.now_len
            
              
        self.buf_priority[:self.now_len] = self.buf_priority[:self.now_len] * 0.1 + 1 # Fixed importance coefficient k_r=0.1 The value of k_r can be adjusted based on the task, as explained in the referenced paper.
        priority = 1 / np.prod(self.buf_priority[:self.now_len], axis = 1)

        """ The following lines of code are designed to assign a higher rarity to the most recently added experience. You can modify it as needed. In my code, I have set max_step to 1000, worker_num represents the actors, and cycle indicates the number of rounds that have been completed. By multiplying these values, we obtain the total amount of new experience incorporated during this update. """
        latestExpCount = cycle * args.max_step * args.worker_num
        data_ids=np.arange(self.next_idx - latestExpCount, self.next_idx) % self.max_len
        priority[data_ids] = self.priority_max

        # Lastly, the priority of all experiences is updated within the tree.
        self.per_tree.update_priority(priority)
        
class BinarySearchTree:
    """Binary Search Tree for PER and BPER

    Contributor: Github GyChou, Github mississippiu
    Reference: https://github.com/kaixindelele/DRLib/tree/main/algos/pytorch/td3_sp
    Reference: https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    """

    def __init__(self, args):
        self.memo_len = args.max_memo  # replay buffer len
        self.if_use_PER = args.if_use_PER
        #self.if_use_BPER = args.if_use_BPER
        self.if_use_PSER = args.if_use_PSER
        self.prob_ary = np.zeros((self.memo_len - 1) + self.memo_len, dtype=np.float64)  # parent_nodes_num + leaf_nodes_num
        self.prob_before = np.zeros( self.memo_len, dtype=np.float64) + 10  # parent_nodes_num + leaf_nodes_num
        self.max_len = len(self.prob_ary)
        self.last_get_leaf_id = 0
        self.now_len = self.memo_len - 1  # pointer
        self.indices = None
        self.depth = int(np.log2(self.max_len))

        self.per_alpha = args.per_alpha  # alpha = (Uniform:0, Greedy:1)
        self.per_beta = 0.4  # beta = (PER:0, NotPER:1)
        
        self.PSER_rou = 0.4
        self.PSER_W = 5
        self.PSER_n = 0.7
        
        self.if_damping = args.if_damping
        self.damping = args.damping#衰减系数
        if self.if_use_PER:
            self.get_indices_is_weights = self.get_indices_is_weights_per
            self.td_error_update = self.td_error_update_PER
        elif args.if_use_BPER:
            self.get_indices_is_weights = self.get_indices_is_weights_BPER

    def update_id(self, data_id, prob=10):  # 10 is max_prob
        tree_id = data_id + self.memo_len - 1
        if self.now_len == tree_id:
            self.now_len += 1

        delta = prob - self.prob_ary[tree_id]
        self.prob_ary[tree_id] = prob

        while tree_id != 0:  # propagate the change through tree
            tree_id = (tree_id - 1) // 2  # faster than the recursive loop
            self.prob_ary[tree_id] += delta

    def update_ids(self, data_ids, prob=10):  # 10 is max_prob 10是最大的优先级，最开始都置为10
        ids = data_ids + self.memo_len - 1
        self.now_len += (ids >= self.now_len).sum()

        upper_step = self.depth - 1
        self.prob_ary[ids] = prob  # ids是子节点的indices
        p_ids = (ids - 1) // 2  # p_ids是父节点的indices

        while upper_step:  # 像上更新整个数
            ids = ( p_ids * 2 + 1 )  # in this while loop, ids means the indices of the left children
            self.prob_ary[p_ids] = self.prob_ary[ids] + self.prob_ary[ids + 1]
            p_ids = (p_ids - 1) // 2
            upper_step -= 1

        self.prob_ary[0] = self.prob_ary[1] + self.prob_ary[2]
        # because we take depth-1 upper steps, ps_tree[0] need to be updated alone

    def get_leaf_id(self, v):
        """Tree structure and array storage:

        Tree index:
              0       -> storing priority sum
            |  |
          1     2
         | |   | |
        3  4  5  6    -> storing priority for transitions
        Array type for storing: [0, 1, 2, 3, 4, 5, 6]
        """
        parent_idx = 0
        while True:
            l_idx = 2 * parent_idx + 1  # the leaf's left node
            r_idx = l_idx + 1  # the leaf's right node
            if l_idx >= (len(self.prob_ary)):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.prob_ary[l_idx]:
                    parent_idx = l_idx
                else:
                    v -= self.prob_ary[l_idx]
                    parent_idx = r_idx
        if(leaf_idx == self.last_get_leaf_id):
            self.last_get_leaf_id = leaf_idx = leaf_idx + 1
        else:
            self.last_get_leaf_id = leaf_idx
        return min(leaf_idx, self.now_len - 2)  # leaf_idx

    def get_indices_is_weights_per(self, batch_size, beg, end):
        self.per_beta = min(1.0, self.per_beta + 0.001)

        # get random values for searching indices with proportional prioritization
        values = (rd.rand(batch_size) + np.arange(batch_size)) * (
            self.prob_ary[0] / batch_size)

        # get proportional prioritization
        leaf_ids = np.array([self.get_leaf_id(v) for v in values])
        self.indices = leaf_ids - (self.memo_len - 1)
        #print(self.indices)

        prob_ary = self.prob_ary[leaf_ids] / self.prob_ary[beg:end].min()
        is_weights = np.power(prob_ary, -self.per_beta)  # important sampling weights
        return self.indices, is_weights
    
    def get_indices_is_weights_BPER(self, batch_size, beg, end):
        self.per_beta = min(1.0, self.per_beta + 0.001)

        # get random values for searching indices with proportional prioritization
        values = (rd.rand(batch_size) + np.arange(batch_size)) * (
            self.prob_ary[0] / batch_size)

        # get proportional prioritization
        leaf_ids = np.array([self.get_leaf_id(v) for v in values])
        self.indices = leaf_ids - (self.memo_len - 1)
        
        if(self.if_damping):
            self.prob_ary[leaf_ids] = self.prob_ary[leaf_ids] * self.damping
            p_ids = (leaf_ids - 1) // 2  # p_ids是父节点的indices
            upper_step = self.depth - 1
            while upper_step:  # 像上更新整个数
                ids = ( p_ids * 2 + 1 )  # in this while loop, ids means the indices of the left children
                self.prob_ary[p_ids] = self.prob_ary[ids] + self.prob_ary[ids + 1]
                p_ids = (p_ids - 1) // 2
                upper_step -= 1
            self.prob_ary[0] = self.prob_ary[1] + self.prob_ary[2]
        return self.indices, 0

    def td_error_update_PER(self, td_error):  # td_error = (q-q).detach_().abs() squeeze()将单维度条目删掉，
        prob = td_error.squeeze().clamp(1e-6, 10).pow(self.per_alpha)
        prob = prob.cpu().numpy()
        self.update_ids(self.indices, prob)
    
    def update_all_ids(self,priority):
        """ 
            BPER 更新优先级的方法 步骤2:更新数组 
            ids:子节点
            p_ids:父节点
        """
        lens = priority.shape[0]
        if(lens == 0):
            return
        ids=np.arange(self.memo_len - 1, lens + self.memo_len - 1)
        #ids = data_ids + self.memo_len - 1
        #self.now_len += (ids >= self.now_len).sum()
        
        upper_step = self.depth - 1
        self.prob_ary[ids] = priority  # here, ids means the indices of given children (maybe the right ones or left ones)
        p_ids = (ids - 1) // 2

        while upper_step:  # 像上更新整个树 Sum-Tree
            ids = ( p_ids * 2 + 1)  # in this while loop, ids means the indices of the left children
            self.prob_ary[p_ids] = self.prob_ary[ids] + self.prob_ary[ids + 1]
            p_ids = (p_ids - 1) // 2
            upper_step -= 1

        self.prob_ary[0] = self.prob_ary[1] + self.prob_ary[2]
    
    def update_priority(self, priority):
        """ BPER 更新优先级的方法 步骤1:剪裁 """
        prob = np.power(np.clip(priority, 1e-6, 10),self.per_alpha)
        
        self.update_all_ids(prob)
        #self.update_ids(self.indices, prob)


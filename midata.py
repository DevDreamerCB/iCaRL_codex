import os
import numpy as np
from scipy.linalg import fractional_matrix_power

def data_alignment(X, num_subjects):
    '''
    :param X: np array, EEG data
    :param num_subjects: int, number of total subjects in X
    :return: np array, aligned EEG data
    '''
    
    
    out = []
    for i in range(num_subjects):
        print('before EA:', X.shape)
        tmp_x = EA(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
        out.append(tmp_x)
        print('after EA:', X.shape)
    X = np.concatenate(out, axis=0)
    
    return X

def EA(x):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA

# if not args.cross_session:
#     data_path = '/data1/bochen/continental_leaning/data82/'
#     test_trials = 56
# else:
#     data_path = '/data1/bochen/continental_leaning/'
#     test_trials = 288

def filter(X, y, task):
        '''
            为了统计每个被试的准确率，所以需要记录测试数据对应的被试id
        '''
        mask = np.isin(y, task)
        return X[mask], y[mask]

class MIData():
    def __init__(self, seed, data_path, is_cross_session, trials_persession, is_align=False):
        super().__init__()
        self.seed = seed
        self.data_path = data_path
        self.trials_persession = trials_persession
        self.is_align = is_align
        if not is_cross_session:
            self._split_data()

    def _split_data(self):
        # 全局类别映射
        GLOBAL_CLASSES = ['left_hand', 'right_hand', 'feet', 'tongue']
        GLOBAL_MAP = {name: i for i, name in enumerate(GLOBAL_CLASSES)}

        def map_labels_array(y_arr):
            mapped = []
            for lab in y_arr:
                mapped.append(GLOBAL_MAP[lab])
            return np.array(mapped, dtype=np.int64)

        # ----- 用户可修改的参数 -----
        split_ratio = 0.3    # 训练集比例，0.82 表示 82%
        # -----------------------------

        # 加载原始数据
        X_full = np.load('/data1/bochen/DeepTransferEEG/data/BNCI2014001/X.npy')
        y_full = np.load('/data1/bochen/DeepTransferEEG/data/BNCI2014001/labels.npy')

        # 设置路径
        train_save_path = "/data1/bochen/cbcontinual/data37/train_data"
        test_save_path = "/data1/bochen/cbcontinual/data37/test_data"
        os.makedirs(train_save_path, exist_ok=True)
        os.makedirs(test_save_path, exist_ok=True) 

        # 实验参数
        num_subjects = 9
        trials_per_session = 288   # 每个 session（T 或 E）的 trial 数
        trials_per_subject = 576   # 每个 subject 总 trial 数（T + E）

        rng_global = np.random.RandomState(self.seed)

        # 分离并保存每个被试的数据（只用 Session T）
        for i in range(num_subjects):
            sub_id = i + 1  # 被试编号从 1 开始：S1, S2, ..., S9
            
            start = i * trials_per_subject
            indices_T = np.arange(start, start + trials_per_session)          # T session

            X_sub_T = X_full[indices_T]         # shape (288, ch, time)
            y_sub_T = y_full[indices_T]         # shape (288,)

            # 映射标签为整数（用于分层）
            y_sub_mapped = map_labels_array(y_sub_T)

            # 为了每个被试有不同但可重复的随机划分，可以基于全局 seed + sub_id 生成子 RNG
            rng = np.random.RandomState(self.seed + sub_id)

            # 按类别分层随机划分
            train_idx_list = []
            test_idx_list = []
            for cls in np.unique(y_sub_mapped):
                cls_idx = np.where(y_sub_mapped == cls)[0]
                perm = rng.permutation(cls_idx)
                cut = int(np.round(len(perm) * split_ratio))

                train_idx_list.extend(perm[:cut].tolist())
                test_idx_list.extend(perm[cut:].tolist())

            # 将索引转为 numpy array（可选择是否排序）
            train_idx = np.array(sorted(train_idx_list), dtype=np.int64)
            test_idx = np.array(sorted(test_idx_list), dtype=np.int64)

            # 切分数据
            X_train_sub = X_sub_T[train_idx]
            y_train_sub = y_sub_mapped[train_idx]
            X_test_sub = X_sub_T[test_idx]
            y_test_sub = y_sub_mapped[test_idx]

            # 保存为 .npz 文件
            train_file = os.path.join(train_save_path, f"S{sub_id}.npz")
            test_file = os.path.join(test_save_path, f"S{sub_id}.npz")
            
            np.savez(train_file, X=X_train_sub, y=y_train_sub)
            np.savez(test_file, X=X_test_sub, y=y_test_sub)
            
            print(f"Saved S{sub_id}: train {X_train_sub.shape}, test {X_test_sub.shape} (seed={self.seed})")

        print("✅ 所有被试数据已分离并保存！")

    def _load_data(self, is_train, idt):

        X = []
        y = []

        for sid in idt:
            if is_train:
                fname = os.path.join(self.data_path, 'train_data/', f"S{sid}.npz")
            else:
                fname = os.path.join(self.data_path, 'test_data/', f"S{sid}.npz")

            data = np.load(fname)
            X.append(data['X'])
            y.append(data['y'])

        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
  
        return X, y
            
    def get_train_data(self, train_idt, class_list):
        '''
            获取训练数据
        '''
        # 导入数据
        
        X_train, y_train = self._load_data(is_train=True, idt=train_idt)
        # 筛选指定任务的数据
        X_train, y_train = filter(X_train, y_train, class_list)
        if self.is_align:
            X_train = data_alignment(X_train, len(train_idt))
        return X_train, y_train

    def get_test_data(self, test_idt, class_list):
        '''
            只获取测试数据
        '''
        X_test, y_test = self._load_data(is_train=False, idt=test_idt)
        X_test, y_test = filter(X_test, y_test, class_list)
        if self.is_align:
            X_test = data_alignment(X_test, len(test_idt))
        return X_test, y_test

if __name__ == "__main__":
    Data = MIData(data_path='/data1/bochen/continental_leaning/', trials_persession=288, is_align=False)
    X_train, y_train, X_test, y_test, = Data.get_data(train_idt=[4], test_idt=[1,2,3,4,5,6], num_class=2)
    print(X_train.shape, y_train.shape, y_test.shape)
    X_test, y_test = Data.get_test_data(test_idt=[1], num_class=4)
    print(X_test.shape, y_test.shape)
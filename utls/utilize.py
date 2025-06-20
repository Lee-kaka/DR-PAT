import os
import random
import sys

import numpy as np
import torch

from datetime import datetime


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_run(log_path, args, seed=None):
    global original_stdout, original_stderr, outfile

    if seed is not None:
        set_seed(seed)

    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    log_file_name = f"log_{'attack_' + str(args.attack_type) if args.with_fakes else 'normal'}_{args.seed}_{args.keep_ratio}_{args.robustness_weight}.txt"
    f = open(os.path.join(log_path, log_file_name), 'w')
    f = Unbuffered(f)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    outfile = os.path.join(log_path, f"log_{datetime.now().strftime('%Y%m%d%H%M')}.txt")

    sys.stderr = f
    sys.stdout = f


def restore_stdout_stderr():
    global original_stdout, original_stderr, outfile

    sys.stdout = original_stdout
    sys.stderr = original_stderr


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def batch_split(users, batch_size, shuffle=True):
    """将用户列表分batch"""
    # 转换为可变的list
    if isinstance(users, (range, set)):
        users = list(users)
    else:
        users = users.copy()

    if shuffle:
        random.shuffle(users)

    for i in range(0, len(users), batch_size):
        yield users[i:i + batch_size]
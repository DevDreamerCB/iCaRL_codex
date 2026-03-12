import torch as tr
import os.path as osp
from datetime import datetime
from datetime import timedelta, timezone

from utils import create_folder


class LogRecord:
    def __init__(self, result_dir=None, data=None, method=None, align=False):

        self.result_dir = result_dir
        self.data_name = data
        self.method = method
        self.align = align

        create_folder(self.result_dir)

        # 更稳健的 GPU/CPU 环境检测
        try:
            self.data_env = 'gpu' if (tr.cuda.is_available() and tr.cuda.device_count() > 0) else 'local'
        except Exception:
            self.data_env = 'local'

        self.out_file = None

    def log_init(self, prefix='log'):
        """
        打开日志文件并写入启动参数。
        prefix: 文件名前缀
        filename: 如果指定则直接使用该文件名（会被拼接到 result_dir）
        extra_info: 可选字符串，写在 header 后面
        """

        if self.data_env in ['local', 'mac']:
            time_str = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(
                timezone(timedelta(hours=8), name='Asia/Shanghai')).strftime("%Y-%m-%d_%H_%M_%S")
        else:
            time_str = datetime.utcnow().replace(tzinfo=timezone.utc).strftime("%Y-%m-%d_%H_%M_%S")
        align_str = 'EA' if self.align else 'noalign'
        file_name = f"{prefix}_{self.method or 'method'}_{align_str}_{time_str}.txt"

        self.out_path = osp.join(self.result_dir, file_name)
        self.out_file = open(self.out_path, 'w', encoding='utf-8')

        # 写 header
        header = self._build_header()
        self.out_file.write(header + '\n')
        self.out_file.flush()
        return self.out_path

    def record(self, log_str):
        """写一行日志并 flush"""
        self.out_file.write(f"{log_str}\n")
        self.out_file.flush()
        return True

    def close(self):
        if self.out_file and not self.out_file.closed:
            self.out_file.close()

    def _build_header(self):
        s = "==========================================\n"
        s += f"start_time: {datetime.now().isoformat()}\n"
        s += f"env: {self.data_env}\n"
        s += f"data: {self.data_name}\n"
        s += f"method: {self.method}\n"
        s += f"result_dir: {self.result_dir}\n"
        s += "=========================================="
        return s



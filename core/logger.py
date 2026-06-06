# coding: utf-8
"""
日志拦截与管理工具 (Dual Logger)
用于将 print 输出的内容同时显示在屏幕上并保存到本地 log 文件中。
"""
import os
import sys
import datetime

class DualLogger(object):
    def __init__(self, log_dir="logs", prefix="run"):
        self.terminal = sys.stdout
        
        # 1. 自动在项目根目录创建 logs 文件夹
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 2. 生成带时间戳的文件名 (例如: evaluate_20260531_103000.log)
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"{prefix}_{current_time}.log")
        
        # 3. 打开文件准备追加写入
        self.log = open(log_file, "a", encoding="utf-8")
        
        # 提前在终端打个招呼
        self.terminal.write(f"📝 [系统日志] 终端输出将实时同步保存至: {log_file}\n\n")
        self.log.write(f"=== 🚀 实验启动时间: {current_time} ===\n\n")

    def write(self, message):
        # 屏幕打印一份
        self.terminal.write(message)
        # 文件里写一份
        self.log.write(message)
        self.log.flush() # 强制实时存盘，防止程序中途崩溃导致日志丢失

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_logger(log_dir="logs", prefix="run"):
    """一键替换系统的标准输出"""
    sys.stdout = DualLogger(log_dir=log_dir, prefix=prefix)
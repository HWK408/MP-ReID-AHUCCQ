import logging
import os
import sys
import os.path as osp
def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name) # 创建一个名为 name 的日志对象。
    logger.setLevel(logging.DEBUG)# 设置日志级别为 DEBUG。这意味着所有级别的信息（Debug, Info, Warning, Error）都会被记录，不会被过滤掉
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    ch = logging.StreamHandler(stream=sys.stdout) #StreamHandler：创建一个处理器，专门负责把日志输出到 sys.stdout（即你的屏幕/终端）。
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")# formatter：定义日志的格式。
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if if_train:
            fh = logging.FileHandler(os.path.join(save_dir, "train_log.log"), mode='w')
        else:
            fh = logging.FileHandler(os.path.join(save_dir, "test_log.log"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

import torch
import os
import logging
logger = logging.getLogger("ater")
import importlib.util
if importlib.util.find_spec('ater_') is not None:
    from ater_ import *
from .ops.quant import *
from .ops.activation import *
from .ops.attention import *
# from .ops.custom import *
from .ops.moe_op import *
from .ops.moe_sorting import *
from .ops.cache import *


def getLogger():
    global logger
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        if int(os.environ.get('ATER_LOG_MORE', 0)):
            formatter = logging.Formatter(
                fmt="[%(name)s %(levelname)s] %(asctime)s.%(msecs)03d - %(process)d:%(processName)s - %(pathname)s:%(lineno)d - %(funcName)s\n%(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger


logger = getLogger()

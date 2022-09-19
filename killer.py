# -*- coding: utf-8 -*-

import os
import sys
from IPython import embed

if __name__ == "__main__":

    start_pid = 35175
    end_pid = 64580
    for pid in range(start_pid, end_pid + 1):
        cmd = 'kill -9 %s' % str(pid)
        os.system(cmd)

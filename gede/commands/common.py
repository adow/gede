# coding=utf-8
#
# common.py
# Common utilities for commands
#

import os
import shutil


def cleanup_screen():
    columns, rows = shutil.get_terminal_size()

    # Fill the terminal
    for _ in range(rows * 2):  # Output blank lines twice the screen height
        print(" " * columns)

    # Various screen clearing commands combination
    print("\033c\033[3J\033[H\033[2J", end="")

    # tmux specific clearing
    if "TMUX" in os.environ:
        os.system("tmux clear-history")

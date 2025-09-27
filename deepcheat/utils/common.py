"""Common functions that are used across files"""
import os

def ensure_dir_exists(path:str) -> None:
    '''Checks if directory exists. Creates it if it doesn't'''
    if not os.path.exists(path):
        os.makedirs(path)

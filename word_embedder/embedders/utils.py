import os
from os.path import dirname
from mkdir_p import mkdir_p


def download_data(url, output_path):
    mkdir_p(dirname(output_path))
    os.system("wget {} -O {}".format(url, output_path))


def extract_gz(path):
    os.system("gzip -d {}".format(path))

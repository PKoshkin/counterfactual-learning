import os
import shutil


data_path = "../../../sorted_data"
new_pool_code_path = "../"
pool_filename = os.path.join(data_path, "pool.json")
days_data_path = os.path.join(data_path, "days_data")


def make_days_data():
    os.mkdir(days_data_path)
    os.system("python2 {} --pool_path {} --skip_prob 0.001 --out_folder {}".format(
        os.path.join(new_pool_code_path, "utils/make_days_jsons.py"),
        pool_filename,
        days_data_path
    ))


def rm_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)


def clear(folders_list):
    for folder in folders_list:
        rm_dir(folder)

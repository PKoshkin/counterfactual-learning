import os


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
        for filename in os.listdir(dir_name):
            filename = os.path.join(dir_name, filename)
            os.remove(filename)
        os.rmdir(dir_name)


def clear():
    rm_dir(days_data_path)
    rm_dir("res")

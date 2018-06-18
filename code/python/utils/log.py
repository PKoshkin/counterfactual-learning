import datetime


def log(message):
    print("\"{}\": {}".format(str(datetime.datetime.now()), message))

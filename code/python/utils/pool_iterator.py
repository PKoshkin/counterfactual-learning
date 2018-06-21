from json import loads as json_from_string


def pool_iterator(pool_path):
    if pool_path.endswith("json"):
        return json_pool_iterator(pool_path)
    elif pool_path.endswith("csv"):
        return csv_pool_iterator(pool_path)
    else:
        raise ValueError("Unknown pool format \"{}\"".format(pool_path))


def json_pool_iterator(pool_path):
    with open(pool_path) as handler:
        for line in handler:
            yield json_from_string(line.strip())


def csv_pool_iterator(pool_path):
    with open(pool_path) as handler:
        for line in handler:
            digits = list(map(float, line.strip().split()))
            line_dict = {}
            line_dict["target"] = int(digits[0])
            line_dict["p"] = digits[1]
            line_dict["pos"] = int(digits[2])
            line_dict["factors"] = digits[3:]
            yield line_dict

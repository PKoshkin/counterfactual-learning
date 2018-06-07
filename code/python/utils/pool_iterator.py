from json import loads as json_from_string


def pool_iterator(pool_path):
    with open(pool_path) as handler:
        for line in handler:
            yield json_from_string(line.strip())

from json import loads, dumps


with open("pool.json") as handler:
    with open("non_factors_pool.json", 'w') as result_file:
        for line in handler:
            json = loads(line.strip())
            json["factors"] = []
            print(dumps(json), file=result_file)

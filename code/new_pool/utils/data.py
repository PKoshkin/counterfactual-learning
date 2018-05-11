from json import loads as json_from_string
from datetime import datetime


def make_catboost_line(json):
    result_list = [json['target'], json['pos'], json['ts']] + json['factors']
    return "\t".join(map(str, result_list))


def make_catboost_file(json_filename, result_filename):
    with open(json_filename) as json_handler, open(result_filename, 'w') as result_handler:
        for line in json_handler:
            json = json_from_string(line.strip())
            print(make_catboost_line(json), file=result_handler)


def make_catboost_files(json_filename, filename_by_ts):
    handlers = {
        timestamp: open(filename_by_ts[timestamp], 'w')
        for timestamp in filename_by_ts.keys()
    }
    with open(json_filename) as json_handler:
        for line in json_handler:
            json = json_from_string(line.strip())
            timestamp = json["ts"]
            dt = datetime.fromtimestamp(timestamp)
            day_timestamp = int(datetime(
                year=dt.year,
                month=dt.month,
                day=dt.day
            ).timestamp())
            print(make_catboost_line(json), file=handlers[day_timestamp])


def check_sorted(json_filename):
    with open(json_filename) as json_handler:
        ts = -1
        for line in json_handler:
            json = json_from_string(line.strip())
            new_ts = json['ts']
            if ts > new_ts:
                return False
            else:
                ts = new_ts
        return True


def get_days(catboost_file):
    dates = set()
    with open(catboost_file) as handler:
        for line in handler:
            ts = int(line.split()[2])
            date = datetime.fromtimestamp(ts).date()
            dates.add(date)
    return dates

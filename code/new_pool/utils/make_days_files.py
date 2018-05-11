import argparse
import os
from datetime import datetime
from json import loads as json_from_string


def make_catboost_line(json):
    result_list = [json['target'], json['pos'], json['ts']] + json['factors']
    return "\t".join(map(str, result_list))


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


def main():
    timestamps = [
        1523048400,
        1523221200,
        1523480400,
        1523739600,
        1523394000,
        1523566800,
        1523912400,
        1523998800,
        1523653200,
        1522789200,
        1523826000,
        1522702800,
        1523134800,
        1522962000,
        1522875600,
        1523307600
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument(type=str, dest='json_filename')
    parser.add_argument(type=str, dest='out_folder')
    args = parser.parse_args()
    filename_by_ts = {
        timestamp: os.path.join(args.out_folder, "day_{}.txt".format(i))
        for i, timestamp in enumerate(timestamps)
    }
    make_catboost_files(args.json_filename, filename_by_ts)


if __name__ == "__main__":
    main()

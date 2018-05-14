from __future__ import print_function
import argparse
import os
from datetime import datetime
import time
from json import loads as json_from_string


def make_catboost_line(json):
    result_list = [json['target'], json['pos'], json['ts']] + json['factors']
    return "\t".join(map(str, result_list))


def make_info_line(json):
    result_list = [json['p'], json['uid'], map(int, json['query'].split())]
    return "\t".join(map(str, result_list))


def make_catboost_files(json_filename, out_folder, timestamps):
    days_handlers = {
        timestamp: open(os.path.join(out_folder, "day_{}.txt".format(i)), 'w')
        for i, timestamp in enumerate(timestamps)
    }
    info_handlers = {
        timestamp: open(os.path.join(out_folder, "info_{}.txt".format(i)), 'w')
        for i, timestamp in enumerate(timestamps)
    }
    with open(json_filename) as json_handler:
        for line in json_handler:
            json = json_from_string(line.strip())
            timestamp = json["ts"]
            date = datetime.fromtimestamp(timestamp).date()
            day_timestamp = int(time.mktime(date.timetuple()))
            print(make_catboost_line(json), file=days_handlers[day_timestamp])
            print(make_info_line(json), file=info_handlers[day_timestamp])


def main():
    timestamps = [
        1522702800,
        1522789200,
        1522875600,
        1522962000,
        1523048400,
        1523134800,
        1523221200,
        1523307600,
        1523394000,
        1523480400,
        1523566800,
        1523653200,
        1523739600,
        1523826000,
        1523912400,
        1523998800
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_filename", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    args = parser.parse_args()
    make_catboost_files(args.json_filename, args.out_folder, timestamps)


if __name__ == "__main__":
    main()

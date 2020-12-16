import argparse
import uproot
import simplejson as json

from os import listdir
from os.path import isfile, join


def get_config(sdir):

    config = {}
    folders = [f for f in listdir(sdir) if not isfile(join(sdir, f))]

    for folder_name in folders:

        print('Processing %s folder' % folder_name)

        current_path = join(sdir, folder_name)
        config[current_path] = {}
        config[current_path]['files'] = {}
        total_events_in_file = 0

        files = [f for f in listdir(current_path) if f[-5:] == '.root']

        for file_name in files:
            try:
                file_path = join(current_path, file_name)
                rfile = uproot.open(file_path)
                events = len(rfile['Delphes']['Tower'].array())
                config[current_path]['files'][file_path] = events
                total_events_in_file = total_events_in_file + events
            except ValueError:
                pass
            except KeyError:
                pass
        config[current_path]['events'] = total_events_in_file
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate file configuration file')
    parser.add_argument('source_dir', type=str,
                        help='Path to root files')
    parser.add_argument('destination_dir', type=str,
                        help='Save path for the configuration file')
    args = parser.parse_args()

    config = get_config(args.source_dir)
    with open(args.destination_dir, 'w') as f:
        json.dump(config, f)

import argparse
import h5py
import numpy as np
import yaml

from utils import IsValidFile

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Convert big h5 files to smaller ones')
    parser.add_argument('-c', '--config', action=IsValidFile, type=str,
                        help='Path to config file', default='ssd-config.yml')
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='Output verbosity')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))

    for x, (sdatasets, size) in enumerate(zip(config['dataset'],
                                              config['dataset_misc']['size'])):

        for sdataset in config['dataset'][sdatasets]:
            if args.verbose:
                print('Processing {}...'.format(sdataset))

            hdf5_dataset = h5py.File(sdataset, 'w')
            for dataset in ['labels',
                            'baseline',
                            'EFlowTrack_Eta',
                            'EFlowTrack_Phi',
                            'EFlowTrack_PT',
                            'EFlowPhoton_Eta',
                            'EFlowPhoton_Phi',
                            'EFlowPhoton_ET',
                            'EFlowNeutralHadron_Eta',
                            'EFlowNeutralHadron_Phi',
                            'EFlowNeutralHadron_ET']:

                sub_dataset = hdf5_dataset.create_dataset(
                    name=dataset,
                    shape=(size,),
                    maxshape=(None,),
                    dtype=h5py.special_dtype(vlen=np.float32))

                s, e, limit = 0, 0, int(size / 3)
                for i, jet in enumerate(['_HH_', '_tt_', '_WW_', '_ZZ_']):

                    if i == 2:
                        limit = int(limit / 2)

                    f = h5py.File('{}/RSGraviton{}NARROW_PU50_PF_{}.h5'.format(
                        config['dataset_misc']['src_folder'], jet, x),
                        'r')
                    v = f.get(dataset)[:limit]
                    s = e
                    e += limit
                    try:
                        sub_dataset[s:e] = v
                    except TypeError:
                        v = [list(i) for i in v]
                        sub_dataset[s:e] = v

import argparse
import h5py
import numpy as np
import os
import simplejson as json
import uproot
import warnings

from uproot_methods import TLorentzVectorArray
from utils import IsReadableDir
from tqdm import tqdm


class PhysicsConstants():

    def __init__(self, example_file):

        # Define jet constants
        self.delta_r = .4
        self.delphes = uproot.open(example_file)['Delphes']
        self.min_eta = -3
        self.max_eta = 3
        self.min_pt = {'q': 30., 'h': 200., 't': 200., 'W': 200}
        self.settings = {'t': {'id': 0, 'pid': [6],
                               'cut_m': [105., 210.]},
                         'V': {'id': 1, 'pid': [23, 24],
                               'cut_m': [65., 105.]},
                         'H': {'id': 2, 'pid': [25],
                               'cut_m': [105., 140.]}}

    def get_edges_ecal(self, x, sample_events=1000):

        all_edges = np.array([], dtype=np.float32)
        edge_arr = self.delphes['EFlowPhoton']['EFlowPhoton.Edges[4]'].array()

        for i in range(sample_events):
            all_edges = np.append(all_edges, edge_arr[i][:, [x, x+1]])
            all_edges = np.unique(all_edges)

        if x == 0:
            all_edges = all_edges[(all_edges > self.min_eta) &
                                  (all_edges < self.max_eta)]

        return all_edges


class HDF5Generator:

    def __init__(self, hdf5_dataset_path, hdf5_dataset_size, files_details,
                 verbose=True):

        self.constants = PhysicsConstants(list(files_details[0])[0])
        self.edges_eta = self.constants.get_edges_ecal(0)
        self.edges_phi = self.constants.get_edges_ecal(2)

        self.hdf5_dataset_path = hdf5_dataset_path
        self.hdf5_dataset_size = hdf5_dataset_size
        self.files_details = files_details

        self.verbose = verbose

    def create_hdf5_dataset(self, progress_bar):

        # Create the HDF5 file.
        hdf5_dataset = h5py.File(self.hdf5_dataset_path, 'w')

        hdf5_labels = hdf5_dataset.create_dataset(
                name='labels',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.float32))

        hdf5_baseline = hdf5_dataset.create_dataset(
                name='baseline',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.float32))

        hdf5_EFlowTrack_Eta = hdf5_dataset.create_dataset(
                name='EFlowTrack_Eta',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.int16))

        hdf5_EFlowTrack_Phi = hdf5_dataset.create_dataset(
                name='EFlowTrack_Phi',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.int16))

        hdf5_EFlowTrack_PT = hdf5_dataset.create_dataset(
                name='EFlowTrack_PT',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.float32))

        hdf5_EFlowPhoton_Eta = hdf5_dataset.create_dataset(
                name='EFlowPhoton_Eta',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.int16))

        hdf5_EFlowPhoton_Phi = hdf5_dataset.create_dataset(
                name='EFlowPhoton_Phi',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.int16))

        hdf5_EFlowPhoton_ET = hdf5_dataset.create_dataset(
                name='EFlowPhoton_ET',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.float32))

        hdf5_EFlowNeutralHadron_Eta = hdf5_dataset.create_dataset(
                name='EFlowNeutralHadron_Eta',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.int16))

        hdf5_EFlowNeutralHadron_Phi = hdf5_dataset.create_dataset(
                name='EFlowNeutralHadron_Phi',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.int16))

        hdf5_EFlowNeutralHadron_ET = hdf5_dataset.create_dataset(
                name='EFlowNeutralHadron_ET',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.float32))

        i = 0

        for file_details in self.files_details:
            file_path = next(iter(file_details.keys()))

            events = file_details[file_path]

            file = uproot.open(file_path)

            eFlowTrack = file['Delphes']['EFlowTrack']
            eFlowPhoton = file['Delphes']['EFlowPhoton']
            eFlowNH = file['Delphes']['EFlowNeutralHadron']

            eFlowTrack_Eta_full = eFlowTrack['EFlowTrack.Eta'].array()
            eFlowTrack_Phi_full = eFlowTrack['EFlowTrack.Phi'].array()
            eFlowTrack_PT_full = eFlowTrack['EFlowTrack.PT'].array()

            eFlowPhoton_Eta_full = eFlowPhoton['EFlowPhoton.Eta'].array()
            eFlowPhoton_Phi_full = eFlowPhoton['EFlowPhoton.Phi'].array()
            eFlowPhoton_ET_full = eFlowPhoton['EFlowPhoton.ET'].array()

            eFlowNH_Eta_full = eFlowNH['EFlowNeutralHadron.Eta'].array()
            eFlowNH_Phi_full = eFlowNH['EFlowNeutralHadron.Phi'].array()
            eFlowNH_ET_full = eFlowNH['EFlowNeutralHadron.ET'].array()

            particle = file['Delphes']['Particle']
            particle_Status_full = particle['Particle.Status'].array()
            particle_PID_full = particle['Particle.PID'].array()
            particle_Eta_full = particle['Particle.Eta'].array()
            particle_Phi_full = particle['Particle.Phi'].array()
            particle_PT_full = particle['Particle.PT'].array()

            jet = file['Delphes']['JetPUPPIAK8']
            jet_SoftDroppedJet_full = jet['JetPUPPIAK8.SoftDroppedJet'].array()
            jet_Taus_full = jet['JetPUPPIAK8.Tau[5]'].array()
            jet_Etas_full = jet['JetPUPPIAK8.Eta'].array()
            jet_Phis_full = jet['JetPUPPIAK8.Phi'].array()
            jet_PT_full = jet['JetPUPPIAK8.PT'].array()

            for event_number in np.arange(events[0], events[1], dtype=int):

                # Get jet labels
                particle_Status = particle_Status_full[event_number]
                particle_PID = particle_PID_full[event_number]
                particle_Eta = particle_Eta_full[event_number]
                particle_Phi = particle_Phi_full[event_number]
                particle_PT = particle_PT_full[event_number]

                labels = self.get_labels(['t', 'H', 'V'],
                                         particle_Status,
                                         particle_PID,
                                         particle_Eta,
                                         particle_Phi,
                                         particle_PT)

                # Flatten the labels array and write it to the dataset
                hdf5_labels[i] = labels.reshape(-1)

                # Get baseline
                jet_SDJ = jet_SoftDroppedJet_full[event_number]
                jet_Tau = jet_Taus_full[event_number]
                jet_Eta = jet_Etas_full[event_number]
                jet_Phi = jet_Phis_full[event_number]
                jet_PT = jet_PT_full[event_number]

                baseline = self.get_baseline(['t', 'H', 'V'],
                                             jet_SDJ,
                                             jet_Tau,
                                             jet_Eta,
                                             jet_Phi,
                                             jet_PT)

                # Flatten the baseline array and write it to the dataset
                hdf5_baseline[i] = baseline.reshape(-1)

                # Get EFlowTrack
                e = eFlowTrack_Eta_full[event_number]
                p = eFlowTrack_Phi_full[event_number]
                v = eFlowTrack_PT_full[event_number]
                mask = ((e > self.edges_eta[0]) & (e < self.edges_eta[-1]))
                e, p, v = e[mask], p[mask], v[mask]
                e, p, v = self.get_energy_map(e, p, v)
                hdf5_EFlowTrack_Eta[i] = e
                hdf5_EFlowTrack_Phi[i] = p
                hdf5_EFlowTrack_PT[i] = v

                # Get EFlowPhoton
                e = eFlowPhoton_Eta_full[event_number]
                p = eFlowPhoton_Phi_full[event_number]
                v = eFlowPhoton_ET_full[event_number]
                mask = ((e > self.edges_eta[0]) & (e < self.edges_eta[-1]))
                e, p, v = e[mask], p[mask], v[mask]
                e, p, v = self.get_energy_map(e, p, v)
                hdf5_EFlowPhoton_Eta[i] = e
                hdf5_EFlowPhoton_Phi[i] = p
                hdf5_EFlowPhoton_ET[i] = v

                # Get EFlowNeutralHadron
                e = eFlowNH_Eta_full[event_number]
                p = eFlowNH_Phi_full[event_number]
                v = eFlowNH_ET_full[event_number]
                mask = ((e > self.edges_eta[0]) & (e < self.edges_eta[-1]))
                e, p, v = e[mask], p[mask], v[mask]
                e, p, v = self.get_energy_map(e, p, v)
                hdf5_EFlowNeutralHadron_Eta[i] = e
                hdf5_EFlowNeutralHadron_Phi[i] = p
                hdf5_EFlowNeutralHadron_ET[i] = v

                i += 1

                if self.verbose:
                    progress_bar.update(1)

        hdf5_dataset.close()

    def get_baseline(self, check_labels, j, taus, etas, phis, pts):
        baselines = np.empty((0, 5))
        m = TLorentzVectorArray.from_cartesian(j.fX, j.fY, j.fZ, j.fE).mass
        m = np.nan_to_num(m)
        taus = np.nan_to_num(taus)
        taus = np.where(taus == 0, 10**-6, taus)
        tau21 = taus[:, 1] / taus[:, 0]
        tau32 = taus[:, 2] / taus[:, 1]
        for label in check_labels:
            jid = self.constants.settings[label]['id']
            cuts_m = self.constants.settings[label]['cut_m']
            mask = (m > cuts_m[0]) & (m < cuts_m[1])
            scores = tau32 if label == 't' else tau21
            for e, p, pt, s in zip(etas[mask],
                                   phis[mask],
                                   pts[mask],
                                   scores[mask]):
                if e < self.edges_eta[0] or e > self.edges_eta[-1]:
                    continue
                e = np.argmax(self.edges_eta >= e) - 1
                p = np.argmax(self.edges_phi >= p) - 1
                baselines = np.vstack((baselines, [jid, e, p, pt, s]))
        return baselines

    def get_labels(self, check_labels, status, pids, etas, phis, pts):
        labels = np.empty((0, 4))
        pids = np.abs(pids)
        for label in check_labels:
            jid = self.constants.settings[label]['id']
            pid = self.constants.settings[label]['pid']
            for s, e, p, pt in zip(status[np.isin(pids, pid)],
                                   etas[np.isin(pids, pid)],
                                   phis[np.isin(pids, pid)],
                                   pts[np.isin(pids, pid)]):
                if s != 22 or e < self.edges_eta[0] or e > self.edges_eta[-1]:
                    continue
                e = np.argmax(self.edges_eta >= e) - 1
                p = np.argmax(self.edges_phi >= p) - 1
                labels = np.vstack((labels, [jid, e, p, pt]))
        return labels

    def get_energy_map(self, etas, phis, values):
        h, _, _ = np.histogram2d(etas,
                                 phis,
                                 bins=[self.edges_eta,
                                       self.edges_phi],
                                 weights=values)
        bins = np.argwhere(h)
        indices_eta = bins[:, 0]
        indices_phi = bins[:, 1]
        values = h[indices_eta, indices_phi]
        return indices_eta, indices_phi, values


class Utils():

    def parse_config(self, folder, nofiles, config_path):

        # Laod configuration
        with open(config_path, 'r') as f:
            config = json.loads(f.read())

        # Total number of events
        total = config[folder]['events']
        files_list = list(config[folder]['files'])
        files_details, files_batch = [], []
        gtotal, fid, event_min_next = 0, 0, 0
        batch_id = 1
        batch_size = total / float(nofiles)
        jtype = folder.split('/')[-1]

        while gtotal < total:

            file = files_list[fid]

            # Set FROM and TO indexes
            event_min = event_min_next
            event_max = config[folder]['files'][file]

            # Fix nominal target of events
            gtotal_target = gtotal + event_max - event_min

            # Save filenames with indexes
            # Fraction of the file
            if batch_id*batch_size <= gtotal_target:
                max_in_this_batch = int(batch_id*batch_size)
                event_max = event_max - (gtotal_target - max_in_this_batch)
                event_min_next = event_max

                # Prevent saving files with no events
                if event_max != event_min:
                    files_batch.append({file: (event_min, event_max)})

                # Push to file details
                files_details.append(files_batch)
                files_batch = []
                batch_id = batch_id + 1
            # Otherwise: full file
            else:
                files_batch.append({file: (event_min, event_max)})
                event_min_next = 0
                fid += 1

            gtotal = gtotal + event_max - event_min

        return files_details, batch_size, gtotal, jtype


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Convert root file data to h5')
    parser.add_argument('src_folder', type=str, help='Folder to convert')
    parser.add_argument('-n', '--number-of-files', type=int, default=10,
                        help='Target number of output files', dest='nfiles')
    parser.add_argument('-o', '--save-path', type=str, action=IsReadableDir,
                        default='.', help='Output directory', dest='save_dir')
    parser.add_argument('-c', '--config', type=str, action=IsReadableDir,
                        default='./data/file-configuration.json',
                        help='Configuration file path', dest='config')
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='Output verbosity')
    args = parser.parse_args()

    utils = Utils()

    files_details, batch_size, total_events, jtype = utils.parse_config(
        args.src_folder, args.nfiles, args.config)

    pb = None
    if args.verbose:
        pb = tqdm(total=total_events, desc=('Processing %s' % jtype))

    for index, file_dict in enumerate(files_details):

        dataset_size = int((index+1)*batch_size)-int((index)*batch_size)
        generator = HDF5Generator(
                hdf5_dataset_path='{0}/{1}_{2}.h5'.format(
                        args.save_dir, jtype, index),
                hdf5_dataset_size=dataset_size,
                files_details=file_dict,
                verbose=args.verbose)
        generator.create_hdf5_dataset(pb)

    if args.verbose:
        pb.close()

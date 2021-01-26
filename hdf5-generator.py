import argparse
import h5py
import numpy as np
import os
import simplejson as json
import uproot
import warnings

from uproot_methods import PtEtaPhiMassLorentzVector, TLorentzVector
from utils import IsReadableDir
from tqdm import tqdm


class PhysicsConstants():

    def __init__(self, example_file):

        # Define jet constants
        self.classes = {'q': 0, 'h': 1, 't': 2, 'W': 1}
        self.delta_r = .4
        self.delphes = uproot.open(example_file)['Delphes']
        self.min_eta = -3
        self.max_eta = 3
        self.min_pt = {'q': 30., 'h': 200., 't': 200., 'W': 200}

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

            genjet = file['Delphes']['GenJet']
            genjet_pt_full_file = genjet['GenJet.PT'].array()  # Jet PT
            genjet_eta_full_file = genjet['GenJet.Eta'].array()  # Jet x
            genjet_phi_full_file = genjet['GenJet.Phi'].array()  # Jet y
            genjet_mass_full_file = genjet['GenJet.Mass'].array()  # Jet mass

            particle = file['Delphes']['Particle']
            particle_pid_full_file = particle['Particle.PID'].array()
            particle_px_full_file = particle['Particle.Px'].array()
            particle_py_full_file = particle['Particle.Py'].array()
            particle_pz_full_file = particle['Particle.Pz'].array()
            particle_e_full_file = particle['Particle.E'].array()

            for event_number in np.arange(events[0], events[1], dtype=int):

                # Get jet labels
                jet_pt = genjet_pt_full_file[event_number]
                jet_eta = genjet_eta_full_file[event_number]
                jet_phi = genjet_phi_full_file[event_number]
                jet_mass = genjet_mass_full_file[event_number]

                etas_mask = ((jet_eta > self.edges_eta[0]) &
                             (jet_eta < self.edges_eta[-1]))

                jet_pt = jet_pt[etas_mask]
                jet_eta = jet_eta[etas_mask]
                jet_phi = jet_phi[etas_mask]
                jet_mass = jet_mass[etas_mask]

                particle_pid = particle_pid_full_file[event_number]
                particle_px = particle_px_full_file[event_number]
                particle_py = particle_py_full_file[event_number]
                particle_pz = particle_pz_full_file[event_number]
                particle_e = particle_e_full_file[event_number]

                labels = self.get_labels(jet_pt, jet_eta, jet_phi, jet_mass,
                                         particle_pid, particle_px,
                                         particle_py, particle_pz, particle_e)

                # Flatten the labels array and write it to the labels dataset.
                hdf5_labels[i] = labels.reshape(-1)

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

    def get_labels(self, j_pt, j_eta, j_phi, j_mass,
                   p_pid, p_px, p_py, p_pz, p_e):

        plv = np.array([])
        labels = np.empty((0, 5))
        p_pid = np.abs(p_pid)

        for x, y, z, e in zip(p_px, p_py, p_pz, p_e):
            plv = np.append(plv, TLorentzVector(x, y, z, e))

        for pt, eta, phi, mass in zip(j_pt, j_eta, j_phi, j_mass):
            label = None

            if mass < 1.:
                continue

            jlv = PtEtaPhiMassLorentzVector(pt, eta, phi, mass)

            # Order is important
            if not label and pt > self.constants.min_pt['t']:
                for lv in plv[p_pid == 6]:
                    if lv.delta_r(jlv) < self.constants.delta_r:
                        label = self.constants.classes['t']
                        break
            if not label and pt > self.constants.min_pt['W']:
                for lv in plv[(p_pid == 23) | (p_pid == 24)]:
                    if lv.delta_r(jlv) < self.constants.delta_r:
                        label = self.constants.classes['W']
                        break
            if not label and pt > self.constants.min_pt['h']:
                for lv in plv[p_pid == 25]:
                    if lv.delta_r(jlv) < self.constants.delta_r:
                        label = self.constants.classes['h']
                        break
            if not label and pt > self.constants.min_pt['q']:
                for lv in plv[p_pid <= 5]:
                    if lv.delta_r(jlv) < self.constants.delta_r:
                        label = self.constants.classes['q']
                        break

            if label is not None:
                e = np.argmax(self.edges_eta >= eta) - 1
                p = np.argmax(self.edges_phi >= phi) - 1
                labels = np.vstack((labels, [label, e, p, pt, mass]))
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

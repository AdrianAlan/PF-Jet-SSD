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
        self.classes = {'b': 0, 'h': 1, 'q': 3, 't': 2, 'W': 1}
        self.delta_r = .4
        self.delphes = uproot.open(example_file)['Delphes']
        self.min_eta = -3
        self.max_eta = 3
        self.min_pt = {'b': 21., 'h': 312., 'q': 1., 't': 433., 'W': 201}
        self.tower = self.delphes['Tower']

    def get_edges_ecal(self, edge_index, tower, sample_events=1000):
        tower_mask_full_file = self.tower[tower].array()
        edges_full_file = self.tower['Tower.Edges[4]'].array()

        global_edges = np.array([], dtype=np.float32)

        for i in range(sample_events):

            edges_event = edges_full_file[i][tower_mask_full_file[i] > 0]
            global_edges = np.append(global_edges,
                                     edges_event[:, edge_index])
            global_edges = np.append(global_edges,
                                     edges_event[:, edge_index+1])
            global_edges = np.unique(global_edges)

        if edge_index == 0:
            global_edges = global_edges[(global_edges > self.min_eta) &
                                        (global_edges < self.max_eta)]

        return global_edges


class HDF5Generator:

    def __init__(self, hdf5_dataset_path, hdf5_dataset_size, files_details,
                 verbose=True):

        self.constants = PhysicsConstants(list(files_details[0])[0])

        self.edges_eta_ecal = self.constants.get_edges_ecal(edge_index=0,
                                                            tower='Tower.Eem')
        self.edges_phi_ecal = self.constants.get_edges_ecal(edge_index=2,
                                                            tower='Tower.Eem')
        self.edges_eta_hcal = self.constants.get_edges_ecal(edge_index=0,
                                                            tower='Tower.Ehad')
        self.edges_phi_hcal = self.constants.get_edges_ecal(edge_index=2,
                                                            tower='Tower.Ehad')

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

        hdf5_ecal_energy = hdf5_dataset.create_dataset(
                name='ecal_energy',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.float32))

        hdf5_ecal_phi = hdf5_dataset.create_dataset(
                name='ecal_phi',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.int16))

        hdf5_ecal_eta = hdf5_dataset.create_dataset(
                name='ecal_eta',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.int16))

        hdf5_hcal_energy = hdf5_dataset.create_dataset(
                name='hcal_energy',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.float32))

        hdf5_hcal_phi = hdf5_dataset.create_dataset(
                name='hcal_phi',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.int16))

        hdf5_hcal_eta = hdf5_dataset.create_dataset(
                name='hcal_eta',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.int16))

        i = 0

        for file_details in self.files_details:
            file_path = next(iter(file_details.keys()))

            events = file_details[file_path]

            # First load values from file
            file = uproot.open(file_path)

            towers = file['Delphes']['Tower']
            ecal_energy_full_file = towers['Tower.Eem'].array()  # ECAL E
            hcal_energy_full_file = towers['Tower.Ehad'].array()  # HCAL E
            edges_full_file = towers['Tower.Edges[4]'].array()  # Crystal edge

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

                etas_mask = ((jet_eta > self.edges_eta_ecal[0]) &
                             (jet_eta < self.edges_eta_ecal[-1]))

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

                # Get ECAL info
                ecal_mask = ecal_energy_full_file[event_number] > 0

                etas = edges_full_file[event_number][ecal_mask][:, 0]
                phis = edges_full_file[event_number][ecal_mask][:, 2]
                energy = ecal_energy_full_file[event_number][ecal_mask]

                etas_mask = ((etas > self.edges_eta_ecal[0]) &
                             (etas < self.edges_eta_ecal[-1]))

                etas = etas[etas_mask]
                phis = phis[etas_mask]
                ecal_phis, ecal_etas = self.get_energy_map(etas, phis,
                                                           cal='ECAL')
                ecal_energy = energy[etas_mask]

                # Get HCAL info
                hcal_mask = hcal_energy_full_file[event_number] > 0

                etas = edges_full_file[event_number][hcal_mask][:, 0]
                phis = edges_full_file[event_number][hcal_mask][:, 2]
                energy = hcal_energy_full_file[event_number][hcal_mask]

                etas_mask = ((etas > self.edges_eta_hcal[0]) &
                             (etas < self.edges_eta_hcal[-1]))

                etas = etas[etas_mask]
                phis = phis[etas_mask]
                hcal_phis, hcal_etas = self.get_energy_map(etas, phis,
                                                           cal='HCAL')
                hcal_energy = energy[etas_mask]
                hcal_phis, hcal_etas, hcal_energy = self.hcal_resize(
                    hcal_phis, hcal_etas, hcal_energy)

                # Push the data to hdf5
                hdf5_ecal_energy[i] = ecal_energy
                hdf5_ecal_phi[i] = ecal_phis
                hdf5_ecal_eta[i] = ecal_etas

                hdf5_hcal_energy[i] = hcal_energy
                hdf5_hcal_phi[i] = hcal_phis
                hdf5_hcal_eta[i] = hcal_etas

                # Flatten the labels array and write it to the labels dataset.
                hdf5_labels[i] = labels.reshape(-1)

                i = i + 1

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
            if not label and pt > self.constants.min_pt['b']:
                for lv in plv[p_pid == 5]:
                    if lv.delta_r(jlv) < self.constants.delta_r:
                        label = self.constants.classes['b']
                        break

            if label is not None:
                e = np.argmax(self.edges_eta_ecal >= eta) - 1
                p = np.argmax(self.edges_phi_ecal >= phi) - 1
                labels = np.vstack((labels, [label, e, p, pt, mass]))
        return labels

    def hcal_resize(self, indices_phi, indices_eta, energy):
        mask = (indices_eta >= 85) & (indices_eta <= 118)

        energy = np.concatenate(
            [np.repeat(x, 5) if mask[i] else np.array([x]) for i, x
             in enumerate(energy)])
        indices_phi = np.concatenate(
            [np.repeat(x, 5) if mask[i] else np.array([x]) for i, x
                in enumerate(indices_phi)])
        indices_eta = np.concatenate(
            [np.array([e]) if e < 85 else np.array([e + 136])
                if e > 118 else 5*e-340 + np.arange(5) for e in indices_eta])

        return indices_phi, indices_eta, energy

    def get_energy_map(self, etas, phis, cal='ECAL'):

        if cal == 'ECAL':
            edges_phi = self.edges_phi_ecal
            edges_eta = self.edges_eta_ecal
        if cal == 'HCAL':
            edges_phi = self.edges_phi_hcal
            edges_eta = self.edges_eta_hcal

        indices_phi = np.squeeze([np.where(edges_phi == i) for i in phis])
        indices_eta = np.squeeze([np.where(edges_eta == i) for i in etas])

        return indices_phi, indices_eta


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

        return files_details, batch_size, gtotal


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

    files_details, batch_size, total_events = utils.parse_config(
        args.src_folder, args.nfiles, args.config)

    pb = None
    if args.verbose:
        pb = tqdm(total=total_events, desc=('Processing %s' % args.src_folder))

    for index, file_dict in enumerate(files_details):

        dataset_size = int((index+1)*batch_size)-int((index)*batch_size)
        generator = HDF5Generator(
                hdf5_dataset_path='{0}/{1}_{2}.h5'.format(
                        args.save_dir, args.src_folder, index),
                hdf5_dataset_size=dataset_size,
                files_details=file_dict,
                verbose=args.verbose)
        generator.create_hdf5_dataset(pb)

    if args.verbose:
        pb.close()

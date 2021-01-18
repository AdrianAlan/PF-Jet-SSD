import h5py
import torch

from ssd import qutils


class CalorimeterJetDataset(torch.utils.data.Dataset):

    def __init__(self, rank, hdf5_source_path, input_dimensions, jet_size,
                 qbits=None, return_pt=False):
        """Generator for calorimeter and jet data"""

        self.rank = rank
        self.source = hdf5_source_path
        self.channels = input_dimensions[0]  # Number of channels
        self.width = input_dimensions[1]  # Width of input
        self.height = input_dimensions[2]  # Height of input
        self.size = jet_size / 2
        self.qbits = qbits
        self.return_pt = return_pt

    def __getitem__(self, index):

        if not hasattr(self, 'hdf5_dataset'):
            self.open_hdf5()

        # Load calorimeter
        indices_ecal_phi = torch.cuda.LongTensor([self.ecal_phi[index]],
                                                 device=self.rank)
        indices_ecal_eta = torch.cuda.LongTensor([self.ecal_eta[index]],
                                                 device=self.rank)
        ecal_energy = torch.cuda.FloatTensor(self.ecal_energy[index],
                                             device=self.rank)

        indices_hcal_phi = torch.cuda.LongTensor([self.hcal_phi[index]],
                                                 device=self.rank)
        indices_hcal_eta = torch.cuda.LongTensor([self.hcal_eta[index]],
                                                 device=self.rank)
        hcal_energy = torch.cuda.FloatTensor(self.hcal_energy[index],
                                             device=self.rank)

        calorimeter, scaler = self.process_images(indices_ecal_phi,
                                                  indices_hcal_phi,
                                                  indices_ecal_eta,
                                                  indices_hcal_eta,
                                                  ecal_energy,
                                                  hcal_energy)

        # Set labels
        labels_raw = torch.cuda.FloatTensor(self.labels[index],
                                            device=self.rank)
        labels_processed = self.process_labels(labels_raw, scaler)

        return calorimeter, labels_processed

    def __len__(self):

        if not hasattr(self, 'hdf5_dataset'):
            self.open_hdf5()

        return self.dataset_size

    def open_hdf5(self):
        self.hdf5_dataset = h5py.File(self.source, 'r')

        self.ecal_energy = self.hdf5_dataset['ecal_energy']
        self.ecal_phi = self.hdf5_dataset['ecal_phi']
        self.ecal_eta = self.hdf5_dataset['ecal_eta']

        self.hcal_energy = self.hdf5_dataset['hcal_energy']
        self.hcal_phi = self.hdf5_dataset['hcal_phi']
        self.hcal_eta = self.hdf5_dataset['hcal_eta']

        self.labels = self.hdf5_dataset['labels']

        self.dataset_size = len(self.labels)

    def process_images(self,
                       indices_ecal_phi, indices_hcal_phi,
                       indices_ecal_eta, indices_hcal_eta,
                       ecal_energy, hcal_energy):

        c_ecal = torch.zeros(indices_ecal_phi.size(1),
                             dtype=torch.long).cuda(self.rank)
        c_hcal = torch.ones(indices_hcal_phi.size(1),
                            dtype=torch.long).cuda(self.rank)
        indices_channels = torch.unsqueeze(torch.cat((c_ecal, c_hcal)), 0)
        indices_phi = torch.cat((indices_ecal_phi, indices_hcal_phi), 1)
        indices_eta = torch.cat((indices_ecal_eta, indices_hcal_eta), 1)
        energy = torch.cat((ecal_energy, hcal_energy))
        scaler = torch.max(energy) / 10

        i = torch.cat((indices_channels, indices_eta, indices_phi), 0)
        v = energy / scaler

        if self.qbits is not None:
            v = qutils.uniform_quantization(v, self.qbits)

        pixels = torch.sparse.FloatTensor(i, v, torch.Size([self.channels,
                                                            self.width,
                                                            self.height]))
        return pixels.to_dense(), scaler

    def process_labels(self, labels_raw, scaler_mass):
        labels_reshaped = labels_raw.reshape(-1, 5)
        labels = torch.empty_like(labels_reshaped)

        # Set fractional coordinates
        labels[:, 0] = (labels_reshaped[:, 1] - self.size) / float(self.width)
        labels[:, 1] = (labels_reshaped[:, 2] - self.size) / float(self.height)
        labels[:, 2] = (labels_reshaped[:, 1] + self.size) / float(self.width)
        labels[:, 3] = (labels_reshaped[:, 2] + self.size) / float(self.height)

        # Set class label
        labels[:, 4] = labels_reshaped[:, 0]

        # Concatinate auxilary labels
        labels_reshaped[:, 4] = labels_reshaped[:, 4] / scaler_mass
        labels = torch.cat((labels, labels_reshaped[:, 4].unsqueeze(1)), 1)

        if self.return_pt:
            labels = torch.cat((labels, labels_reshaped[:, 3].unsqueeze(1)), 1)

        return labels

import h5py
import torch
import torch.cuda as tcuda

from ssd import qutils


class CalorimeterJetDataset(torch.utils.data.Dataset):

    def __init__(self,
                 rank,
                 hdf5_source_path,
                 input_dimensions,
                 jet_size,
                 cpu=False,
                 flip_prob=None,
                 raw=False,
                 return_baseline=False,
                 return_pt=False):
        """Generator for calorimeter and jet data"""

        self.rank = rank
        self.source = hdf5_source_path
        self.channels = input_dimensions[0]  # Number of channels
        self.width = input_dimensions[1]  # Width of input
        self.height = input_dimensions[2]  # Height of input
        self.size = jet_size / 2
        self.flip_prob = flip_prob
        self.cpu = cpu
        self.raw = raw
        self.return_baseline = return_baseline
        self.return_pt = return_pt

    def __getitem__(self, index):

        if not hasattr(self, 'hdf5_dataset'):
            self.open_hdf5()

        idx_eFTrack_Eta = tcuda.LongTensor([self.eFTrack_Eta[index]],
                                           device=self.rank)
        idx_eFTrack_Phi = tcuda.LongTensor([self.eFTrack_Phi[index]],
                                           device=self.rank)
        val_eFTrack_PT = tcuda.FloatTensor(self.eFTrack_PT[index],
                                           device=self.rank)

        idx_eFPhoton_Eta = tcuda.LongTensor([self.eFPhoton_Eta[index]],
                                            device=self.rank)
        idx_eFPhoton_Phi = tcuda.LongTensor([self.eFPhoton_Phi[index]],
                                            device=self.rank)
        val_eFPhoton_ET = tcuda.FloatTensor(self.eFPhoton_ET[index],
                                            device=self.rank)

        idx_eFNHadron_Eta = tcuda.LongTensor([self.eFNHadron_Eta[index]],
                                             device=self.rank)
        idx_eFNHadron_Phi = tcuda.LongTensor([self.eFNHadron_Phi[index]],
                                             device=self.rank)
        val_eFNHadron_ET = tcuda.FloatTensor(self.eFNHadron_ET[index],
                                             device=self.rank)

        calorimeter, scaler = self.process_images(idx_eFTrack_Eta,
                                                  idx_eFPhoton_Eta,
                                                  idx_eFNHadron_Eta,
                                                  idx_eFTrack_Phi,
                                                  idx_eFPhoton_Phi,
                                                  idx_eFNHadron_Phi,
                                                  val_eFTrack_PT,
                                                  val_eFPhoton_ET,
                                                  val_eFNHadron_ET)
        # Set labels
        labels = tcuda.FloatTensor(self.labels[index], device=self.rank)
        labels = self.process_labels(labels, scaler)

        if self.flip_prob:
            if torch.rand(1) < self.flip_prob:
                calorimeter, labels = self.flip_image(calorimeter,
                                                      labels,
                                                      vertical=True)
            if torch.rand(1) < self.flip_prob:
                calorimeter, labels = self.flip_image(calorimeter,
                                                      labels,
                                                      vertical=False)

        if self.cpu:
            calorimeter = calorimeter.cpu()
            labels = labels.cpu()
            scaler = scaler.cpu()

        if self.return_baseline:
            base = tcuda.FloatTensor(self.base[index], device=self.rank)
            base = self.process_baseline(base)
            return calorimeter, labels, base, scaler

        return calorimeter, labels

    def __len__(self):

        if not hasattr(self, 'hdf5_dataset'):
            self.open_hdf5()

        return self.dataset_size

    def flip_image(self, image, labels, vertical=True):
        if vertical:
            axis = 1
            labels[:, [0, 2]] = 1 - labels[:, [0, 2]]
            labels = labels[:, [2, 1, 0, 3, 4, 5]]
        else:
            axis = 2
            labels[:, [1, 3]] = 1 - labels[:, [1, 3]]
            labels = labels[:, [0, 3, 2, 1, 4, 5]]
        image = torch.flip(image, [axis])
        return image, labels

    def normalize(self, tensor):
        m = torch.mean(tensor)
        s = torch.std(tensor)
        return tensor.sub(m).div(s)

    def open_hdf5(self):
        self.hdf5_dataset = h5py.File(self.source, 'r')

        self.eFTrack_Eta = self.hdf5_dataset['EFlowTrack_Eta']
        self.eFTrack_Phi = self.hdf5_dataset['EFlowTrack_Phi']
        self.eFTrack_PT = self.hdf5_dataset['EFlowTrack_PT']

        self.eFPhoton_Eta = self.hdf5_dataset['EFlowPhoton_Eta']
        self.eFPhoton_Phi = self.hdf5_dataset['EFlowPhoton_Phi']
        self.eFPhoton_ET = self.hdf5_dataset['EFlowPhoton_ET']

        self.eFNHadron_Eta = self.hdf5_dataset['EFlowNeutralHadron_Eta']
        self.eFNHadron_Phi = self.hdf5_dataset['EFlowNeutralHadron_Phi']
        self.eFNHadron_ET = self.hdf5_dataset['EFlowNeutralHadron_ET']

        self.labels = self.hdf5_dataset['labels']
        self.base = self.hdf5_dataset['baseline']

        self.dataset_size = len(self.labels)

    def process_images(self,
                       idx_eFTrack_Eta, idx_eFPhoton_Eta, idx_eFNHadron_Eta,
                       idx_eFTrack_Phi, idx_eFPhoton_Phi, idx_eFNHadron_Phi,
                       val_eFTrack_PT, val_eFPhoton_ET, val_eFNHadron_ET):

        v0 = 0*torch.ones(idx_eFTrack_Eta.size(1),
                          dtype=torch.long).cuda(self.rank)
        v1 = 1*torch.ones(idx_eFPhoton_Eta.size(1),
                          dtype=torch.long).cuda(self.rank)
        v2 = 2*torch.ones(idx_eFNHadron_Eta.size(1),
                          dtype=torch.long).cuda(self.rank)

        idx_channels = torch.cat((v0, v1, v2)).unsqueeze(0)
        idx_eta = torch.cat((idx_eFTrack_Eta,
                             idx_eFPhoton_Eta,
                             idx_eFNHadron_Eta), 1)
        idx_phi = torch.cat((idx_eFTrack_Phi,
                             idx_eFPhoton_Phi,
                             idx_eFNHadron_Phi), 1)

        i = torch.cat((idx_channels, idx_eta, idx_phi), 0)
        v = torch.cat((val_eFTrack_PT, val_eFPhoton_ET, val_eFNHadron_ET))

        scaler = torch.max(v)
        pixels = torch.sparse.FloatTensor(i, v, torch.Size([self.channels,
                                                            self.width,
                                                            self.height]))
        pixels = pixels.to_dense()
        pixels = self.normalize(pixels)
        return pixels, scaler

    def process_baseline(self, base_raw):
        base_reshaped = base_raw.reshape(-1, 5)
        base = torch.empty_like(base_reshaped)

        # Set fractional coordinates
        base[:, 0] = base_reshaped[:, 1] / float(self.width)
        base[:, 1] = base_reshaped[:, 2] / float(self.height)

        # Set class label
        base[:, 2] = base_reshaped[:, 0] + 1

        # Add score
        base[:, 3] = 1 - base_reshaped[:, 4]

        # Add truth
        base[:, 4] = 0

        # Add pT
        base = torch.cat((base, base_reshaped[:, 3].unsqueeze(1)), 1)

        return base

    def process_labels(self, labels_raw, scaler):
        labels_reshaped = labels_raw.reshape(-1, 4)
        labels = torch.empty_like(labels_reshaped)

        # Set fractional coordinates
        labels[:, 0] = (labels_reshaped[:, 1] - self.size) / float(self.width)
        labels[:, 1] = (labels_reshaped[:, 2] - self.size) / float(self.height)
        labels[:, 2] = (labels_reshaped[:, 1] + self.size) / float(self.width)
        labels[:, 3] = (labels_reshaped[:, 2] + self.size) / float(self.height)

        # Set class label
        labels = torch.cat((labels, labels_reshaped[:, 0].unsqueeze(1) + 1), 1)

        if self.return_pt:
            pts = labels_reshaped[:, 3].unsqueeze(1)
            if not self.raw:
                pts = pts / scaler
            labels = torch.cat((labels, pts), 1)

        return labels

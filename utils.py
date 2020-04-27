import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import simplejson as json

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

class Plotting():

    def __init__(self, save_path):

        self.save_path = save_path
        self.line_styles = [(0, ()), (0, (3, 2))]

        plt.style.use('plots/ssdjet.mplstyle')
        matplotlib.rcParams["figure.figsize"] = (8.0, 6.0)

        with open('./data/palette.json') as json_file:
            self.color_palette = json.load(json_file)
        self.colors = [self.color_palette['indigo'],
                       self.color_palette['red'],
                       self.color_palette['green']]

    def draw_loss(self, data_train, data_val, keys):
        """Plots the training and validation loss"""

        fig, ax = plt.subplots()
        plt.xlabel("Epoch", horizontalalignment='right', x=1.0)
        plt.ylabel("Loss", horizontalalignment='right', y=1.0)
        plt.yscale("log")

        for x, (train, val, key) in enumerate(zip(data_train, data_val, keys)):
            color = self.colors[x]
            plt.plot(train,
                     linestyle=self.line_styles[0],
                     color=color['shade_800'],
                     label=key)
            plt.plot(val,
                     linestyle=self.line_styles[1],
                     color=color['shade_400'])

        ax.legend()
        ax.text(0, 1.02, 'CMS',
                weight='bold',
                transform=ax.transAxes,
                color=self.color_palette['grey']['shade_900'],
                fontsize=14)

        fig.savefig(self.save_path)
        plt.close(fig)

    def draw_precision_recall(self, data):
        """Plots the precision recall curve"""

        fig, ax = plt.subplots()
        plt.xlabel("Recall (TPR)", horizontalalignment='right', x=1.0)
        plt.ylabel("Precision (PPV)", horizontalalignment='right', y=1.0)

        for x, (recall, precision, jet, ap) in enumerate(data):
            recall = np.append(1, recall)
            precision = np.append(0, precision)
            plt.plot(recall, precision,
                     linestyle=self.line_styles[0],
                     color=self.colors[x]['shade_800'],
                     label='{0}, AP: {1:.3f}'.format(jet, ap))

        ax.legend()
        ax.text(0, 1.02, 'CMS',
                weight='bold',
                transform=ax.transAxes,
                color=self.color_palette['grey']['shade_900'],
                fontsize=14)
               
        logo = OffsetImage(plt.imread('./plots/hls4mllogo.jpg', format='jpg'), zoom=0.08)
        ab = AnnotationBbox(logo, [0, 1], xybox=(0.12, 1.085), frameon=False)
        ax.add_artist(ab)

        fig.savefig(self.save_path)
        plt.close(fig)
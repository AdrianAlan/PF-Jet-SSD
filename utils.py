import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import simplejson as json

from matplotlib.offsetbox import OffsetImage, AnnotationBbox


class Plotting():

    def __init__(self, save_path, ref_recall=0.3):

        self.save_path = save_path
        self.line_styles = [(0, ()), (0, (5, 2)), (0, (2, 2))]
        self.legend = ['Full Precision Network', 'Ternary Weight Network']
        self.ref_recall = ref_recall

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
                fontsize=13)

        fig.savefig(self.save_path)
        plt.close(fig)

    def draw_precision_recall(self, data):
        """Plots the precision recall curve"""

        fig, ax = plt.subplots()
        plt.xlabel("Recall (TPR)", horizontalalignment='right', x=1.0)
        plt.ylabel("Precision (PPV)", horizontalalignment='right', y=1.0)
        ref_precisions = []

        for q, data_model in enumerate(data):
            shade = 'shade_800' if q else 'shade_500'
            for x, (recall, precision, jet, ap) in enumerate(data_model):
                # Helper line
                ref_precision = np.round(
                    precision[(np.abs(recall - self.ref_recall)).argmin()], 2)
                ref_precisions.append(ref_precision)
                ax.plot([0, 0.3], [ref_precision, ref_precision],
                        linestyle=self.line_styles[2],
                        linewidth=0.8,
                        alpha=0.5,
                        color=self.color_palette['grey']['shade_500'])

                recall = np.append(1, recall)
                precision = np.append(0, precision)
                plt.plot(recall, precision,
                         linestyle=self.line_styles[q],
                         color=self.colors[x][shade],
                         label='{0}: {1} jets, AP: {2:.3f}'.format(
                                 self.legend[q], jet, ap))

        # Helper line c.d.
        plt.xticks(list(plt.xticks()[0]) + [self.ref_recall])
        plt.yticks([0, 1] + ref_precisions)
        ax.plot([0.3, 0.3], [0, np.max(ref_precisions)],
                linestyle=self.line_styles[2],
                alpha=0.5,
                color=self.color_palette['grey']['shade_500'])

        ax.legend(loc='upper center', bbox_to_anchor=(0.25, -0.1))

        ax.text(0, 1.02, 'CMS',
                weight='bold',
                transform=ax.transAxes,
                color=self.color_palette['grey']['shade_900'],
                fontsize=13)

        logo = OffsetImage(plt.imread('./plots/hls4mllogo.jpg', format='jpg'),
                           zoom=0.08)
        ab = AnnotationBbox(logo, [0, 1], xybox=(0.12, 1.085), frameon=False)
        ax.add_artist(ab)

        fig.savefig(self.save_path)
        plt.close(fig)

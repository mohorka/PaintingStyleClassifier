import matplotlib.pyplot as plt

from core.config.config import RESULTS_PATH

def plot_mask(features, nameing):
    square = 2
    ix = 1
    for _ in range(square):
        for _ in range(square):
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(features[0,:,:,ix-1])
            ix += 1


    plt.savefig(''.join([RESULTS_PATH,nameing,'_','result.png']))
    plt.figure().clear()
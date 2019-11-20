from utilities.training.logger import transformer_output_log
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
from models.vggpt2.model import gpt2_tokenizer
from utilities.vqa.dataset import *
import io


def softmap_visualize(softmaps, sequence, image, show_plot=True):
    """
    Visualize softmaps
    :param softmaps: The softmaps. Expected shape (N, 7*7)
    :param sequence: The sequence. Expected shape (N+1)
    :param show_plot: Whether or not to show the plt plot
    :return: The plt figure
    """

    softmaps = softmaps.detach().to('cpu')

    # image = image.resize([7 * 32, 7 * 32], Image.LANCZOS)
    words_tokenized = sequence.tolist()
    words = [gpt2_tokenizer.convert_ids_to_tokens(w) for w in words_tokenized]
    alphas = []
    softmaps = softmaps.view(softmaps.size(0), 7, 7)
    if '<pad>' in words:
        words = words[:words.index('<pad>')]
    assert words[0] == '<bos>'
    for t in range(len(words)):
        if t >= 25:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        if words[t][0] == 'Ġ':  # Remove new word indicator of gpt2 tokenizer
            words[t] = words[t][1:]

        plt.text(0, 0, '%s' % words[t], color='white', backgroundcolor='black', fontsize=10)
        plt.imshow(image)
        # We do not have a softmap for the bos token
        if words[t] != '<bos>':
            current_alpha = softmaps[t - 1, :]
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=32, sigma=8)
            # alpha = skimage.transform.resize(current_alpha.numpy(), [7 * 32, 7 * 32])
            if words[t] in ['<eos>', '<sep>']:
                plt.imshow(alpha, alpha=0)
            else:
                plt.imshow(alpha, alpha=0.7)
            # plt.set_cmap(cm.Greys_r)
            alphas.append(alpha)
        plt.axis('off')

    fig = plt.gcf()
    if show_plot:
        plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    im = Image.open(buf)
    plt.clf()
    # Generate alphas images

    return im

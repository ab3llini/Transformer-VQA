import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)


from utilities import paths
from models.vggpt2.model import VGGPT2
from utilities.visualization.softmap import *
from utilities.evaluation.evaluate import *
from utilities.evaluation.beam_search import *


init = False
checkpoint = None
model = None


def rm_eos_token(a):
    if a[-1] == gpt2_tokenizer.eos_token_id:
        return a[:-1]
    return a


def do_beam_search(__model, question, image, beam_search_input, device='cuda:1', beam_size=1, maxlen=20):
    cs, rs, cs_out, rs_out = beam_search_with_softmaps(__model, beam_search_input, len(gpt2_tokenizer), beam_size,
                                                       gpt2_tokenizer.eos_token_id, maxlen, device=device)

    if cs is not None:
        seq = torch.cat([question, torch.tensor(cs).to(device)])
        return gpt2_tokenizer.decode(rm_eos_token(cs)), softmap_visualize(cs_out, seq, image, 7, False)
    elif rs is not None:
        seq = torch.cat([question, torch.tensor(rs).to(device)])
        return gpt2_tokenizer.decode(rm_eos_token(rs)), softmap_visualize(rs_out, seq, image, 7, False)


def init_singletons():
    global init
    if not init:
        global checkpoint
        global model

        vggpt2_path = paths.resources_path('models', 'vggpt2')
        checkpoint = torch.load(os.path.join(vggpt2_path, 'checkpoints', 'latest', 'B_20_LR_5e-05_CHKP_EPOCH_19.pth'))

        model = VGGPT2()
        model.to('cuda:1').set_train_on(False)
        model.load_state_dict(checkpoint)

        init = True


def answer(question, image, args=None):
    global model

    init_singletons()

    # Resize and convert image to tensor
    torch.manual_seed(0)
    resized_image = resize_image(image)
    tensor_image = normalized_tensor_image(resized_image).to('cuda:1')

    # Encode question
    question_tkn = gpt2_tokenizer.encode(question)
    question_tkn = [gpt2_tokenizer.bos_token_id] + question_tkn + [gpt2_tokenizer.sep_token_id]
    tensor_question = torch.tensor(question_tkn).long().to('cuda:1')

    # Prepare Beam search input
    beam_input = BeamSearchInput(0, 0, tensor_question, tensor_image)

    # Predict
    ans, softmaps = do_beam_search(model, tensor_question, resized_image, beam_input)

    return ans, [resized_image, softmaps]


if __name__ == '__main__':
    with open(paths.resources_path('tmp', 'image.png'), 'rb') as fp:
        img = Image.open(fp)
        q = 'What do you see?'
        ans, _, _ = answer(q, img)
        print(ans)

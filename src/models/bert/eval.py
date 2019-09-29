import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from models.bert import model as bert_model
from utilities.paths import *
from utilities.vqa.dataset import *
from torch.utils.data import DataLoader
from models.bert import loss as bert_loss
import torch
from pytorch_transformers import BertTokenizer

if __name__ == '__main__':

    # Load model
    print('Loading model..')
    model = bert_model.Model()
    model.load_state_dict(
        torch.load(resources_path('models', 'bert', 'checkpoints', 'bert_vgg_1M_B_64_LR_5e-05_CHKP_EPOCH_3.pth')))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    decode = lambda text: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    # Hardware
    device = torch.cuda.current_device()
    model.to(device)

    # Data
    batch_size = 64
    ts_dataset = BertDataset(directory=resources_path('models', 'bert', 'data'), name='ts_bert_1M.pk')
    ts_loader = DataLoader(dataset=ts_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=2)

    mode = input('Automatic or Interactive mode? [a/i]: ')
    if mode == 'a':
        # Automatic evaluation
        iterations = tqdm(ts_loader)
        total_loss = 0

        # Deactivate back propagation
        with torch.no_grad():
            for it, batch in enumerate(iterations):
                # Accessing batch objects and moving them to the computing device
                sequences = batch[1].to(device)
                images = batch[2].to(device)
                token_type_ids = batch[3].to(device)
                attention_masks = batch[4].to(device)

                # Computing model output
                out = model(sequences, token_type_ids, attention_masks, images)

                # Compute the loss
                loss = bert_loss.loss_fn(out, sequences)
                total_loss += loss

                iterations.set_description('Test loss: {}'.format(loss.item()))

            print('Total test loss (averaged) = {}'.format(total_loss / len(iterations)))

    else:

        selected_image = None
        while selected_image is None:
            random_idx = random.randint(0, len(ts_dataset))
            random_sample = ts_dataset.data[random_idx]
            print('Randomly selected this sample:\nSequence={}'.format(tokenizer.decode(random_sample[1])))
            open_fp = ts_dataset.get_image(random_sample[2])
            open_fp.show()

            # Check for good sample
            ok = input('Keep the current sample? [y/n]: ')
            if ok == 'y':
                stop = 'n'
                while stop == 'n':
                    question = input('Enter your question: ')
                    question = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(
                        tokenizer.tokenize(question)) + [tokenizer.sep_token_id]
                    input_type_ids = [0] * len(question)
                    att_mask = [1] * len(question)
                    selected_image = ts_dataset[random_idx][2].unsqueeze(0).to(device)

                    generated_tokens = []
                    last_full_output = None
                    limit = 15
                    last_token = -1

                    with torch.no_grad():
                        # Iterative model pooling
                        while last_token not in tokenizer.all_special_ids and len(generated_tokens) <= limit:
                            sequence_in = torch.tensor(question + generated_tokens).unsqueeze(0).to(device)
                            input_type_ids_in = torch.tensor(input_type_ids).unsqueeze(0).to(device)
                            att_mask_in = torch.tensor(att_mask).unsqueeze(0).to(device)
                            out = model(sequence_in, input_type_ids_in, att_mask_in, selected_image)

                            # Get next predicted tokens. At the beginning is only one, then two, three..
                            generated_tokens = torch.argmax(out[0, -len(generated_tokens) - 1:], dim=1).tolist()

                            # Save last full output
                            last_full_output = torch.argmax(out[0], dim=1).tolist()

                            print('Iteration {} -> Output = {} \n Generated tokens = {}'.format(len(generated_tokens),
                                                                                                tokenizer.decode(
                                                                                                    last_full_output),
                                                                                                tokenizer.decode(
                                                                                                    generated_tokens)))

                            print(
                                'Undecoded answer tokens {}'.format(tokenizer.convert_ids_to_tokens(generated_tokens)))

                            input_type_ids.append(1)
                            att_mask.append(1)

                            last_token = generated_tokens[-1]

                        print('Model full output: {}'.format(tokenizer.decode(last_full_output)))
                        print('Answer tokens: {}'.format(tokenizer.decode(generated_tokens)))

                    stop = input('Change image? [y/n (default)]: ')
                    if stop != 'y':
                        stop = 'n'
                    else:
                        open_fp.close()
                        selected_image = None
            else:
                open_fp.close()
                continue

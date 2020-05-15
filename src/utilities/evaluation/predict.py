import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

import torch


def answer(model, stop_words, max_len, device, tokenizer, *args):
    model.to(device)

    # Prepare inputs (always question, eventual image)
    tensors = [convert_q(prepare_q(args[0], tokenizer))]
    tensors = list(map(lambda arg: arg.to(device), tensors))

    with torch.no_grad():
        answer = []
        stop_condition = False
        its = 0

        while not stop_condition:
            out = model(*tensors)
            # Get predicted words in this beam batch
            pred = torch.argmax(out[0, -1, :])

            eos = (pred.item() in stop_words)
            its += 1

            stop_condition = eos or its > max_len

            # Append the predicted token to the question
            tensors[0] = torch.cat([tensors[0], pred.unsqueeze(0).unsqueeze(0)], dim=1)
            # Append the predicted token to the answer
            answer.append(pred.item())

        return tokenizer.decode(answer)


def prepare_q(question, tokenizer):
    return [tokenizer.bos_token_id] + tokenizer.encode(question) + [tokenizer.sep_token_id]


def convert_q(question):
    return torch.tensor(question).unsqueeze(0)


def pretty_format(question, answer, tokenizer):
    return f'{tokenizer.bos_token} {question} {tokenizer.sep_token} {answer}'

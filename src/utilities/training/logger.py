import torch

def transformer_output_log(out, sequences, description, tokenizer, log_size=3):
    ret = ''
    for s in range(log_size):
        ret += '*' * 25 + '\n'
        ret += '{}'.format(description) + '\n'
        ret += 'Input = {}\n'.format(tokenizer.decode(sequences[0][s].tolist()))
        ret += 'Output = {}\n'.format(tokenizer.decode(torch.argmax(outputs[s], dim=1).tolist()))
    return ret

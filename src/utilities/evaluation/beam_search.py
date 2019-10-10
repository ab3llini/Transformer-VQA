import torch
from torch.functional import F
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class BeamSearchInput:
    def __init__(self, model, seq_id, logits_id, *args):
        self.model = model
        self.args = args
        self.seq_id = seq_id
        self.logits_id = logits_id

    def get_inputs(self):
        return list(self.args), self.seq_id

    def get_logits(self, running_args):
        return self.model(*running_args)[self.logits_id]


class BertBeamSearchInput(BeamSearchInput):
    def __init__(self, model, seq_id, seg_id, logits_id, *args):
        super().__init__(model, seq_id, logits_id, *args)
        self.seg_id = seg_id

    def get_logits(self, running_args):
        out = self.model(*running_args)
        running_args[self.seg_id] = torch.cat(
            [running_args[self.seg_id], torch.ones(running_args[self.seg_id].shape[0], 1).long().to('cuda')], dim=1)
        return out[self.logits_id]


def beam_search(beam_search_input, vocab_size, beam_size, stop_word, max_len):
    # ----------------------------------------------
    # INITIALIZATION
    # ----------------------------------------------
    args, seq_idx = beam_search_input.get_inputs()
    running_args = args[:]
    running_sequences = args[seq_idx].expand(beam_size, -1)  # (beam_size, seq_len + step)
    complete_sequences = []
    step = 1
    k = beam_size  # Copy because we need to lively change this parameter

    # Args expansion to create a batch of size beam_size
    for idx in range(len(running_args)):
        if idx != seq_idx:
            # Expansion
            running_args[idx] = args[idx].expand([k] + list(args[idx].shape))
        else:
            running_args[idx] = running_sequences

    # ----------------------------------------------
    # BEAM SEARCH
    # ----------------------------------------------
    while True:
        # Get output logits
        with torch.no_grad():
            logits = beam_search_input.get_logits(running_args)  # (beam_size, seq_len + step, voc_size)
        # Get predicted words in this beam batch
        preds = logits[:, -1]
        # Apply a softmax in order to have all values between 0 and 1
        preds = F.softmax(preds)
        # Compute top k logits
        if step == 1:
            top_k_scores, top_k_words = preds[0].topk(k, dim=0)
        else:
            top_k_scores, top_k_words = preds.view(-1).topk(k, dim=0)

        # Select the most probable beams and update running sequences
        source_beam_ids = top_k_words / vocab_size
        next_most_probable_words = top_k_words % vocab_size
        selected_source_beams = running_sequences[source_beam_ids]  # (beam_size, seq_len + step)
        next_most_probable_words = next_most_probable_words.unsqueeze(1)  # (beam_size, 1)
        # Update running sequences
        running_sequences = torch.cat([selected_source_beams, next_most_probable_words], dim=1)

        # Check if any sequence reached the end
        incomplete_sequence_ids = [i for i, next_word in enumerate(next_most_probable_words.squeeze(1)) if
                                   next_word.item() != stop_word]

        complete_sequence_ids = list(
            set(range(len(next_most_probable_words.squeeze(1)))) - set(incomplete_sequence_ids))

        # If new sequences have been found, save them and decrease the beam size
        n_completed = len(complete_sequence_ids)
        if n_completed > 0:
            complete_sequences.extend(running_sequences[complete_sequence_ids].tolist())
            k -= n_completed
            # Reduce batch size
            for idx in range(len(running_args)):
                if idx != seq_idx:
                    # Expansion
                    running_args[idx] = args[idx].expand([k] + list(args[idx].shape))
                else:
                    running_args[idx] = running_sequences[incomplete_sequence_ids]
        else:
            running_args[seq_idx] = running_sequences
        # Check if we need to stop
        print('Current step = {}, Max = {}'.format(step, max_len))
        if step == max_len:
            break
        else:
            step += 1

    return complete_sequences, running_sequences.tolist()


if __name__ == '__main__':
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

    # Encode a text inputs
    text = "Who is tom cruise? "
    indexed_tokens = tokenizer.encode(text)

    # Convert indexed tokens in a PyTorch tensor
    tokens_tensor = torch.tensor([indexed_tokens]).squeeze(0)

    # Load pre-trained model (weights)
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model.eval()

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    model.to('cuda')

    cs, rs = beam_search(BeamSearchInput(model, 0, 0, tokens_tensor), vocab_size=len(tokenizer),
                         beam_size=100, stop_word=tokenizer.sep_token_id, max_len=50)

    for seq in cs:
        print(tokenizer.decode(seq))

    for seq in rs:
        print(tokenizer.decode(seq))

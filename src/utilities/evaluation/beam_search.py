import torch
from torch.functional import F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset
import numpy as np


class BeamSearchInput:
    def __init__(self, seq_idx, logits_idx, *args):
        """
        Beam search input. Use this class to provide input data to the beam search utility
        :param model: The instance of the model in use
        :param seq_idx: The index of the input sequence in the model input args
        :param logits_idx: The index of the logits in the model output args. If none we assume single output model.
        :param args: *Input arguments to the model
        """
        self.args = list(args)
        self.seq_idx = seq_idx
        self.logits_idx = logits_idx

    def get_inputs(self, device):
        """
        This method provides input for the model
        :return: Returns the list of arguments followed by the index of the sequence in the args
        """
        for i, arg in enumerate(self.args):
            self.args[i] = arg.to(device)
        return self.args, self.seq_idx

    def update_args(self, running_args, initial_args):
        """
        Returns the logits from the model output
        :param running_args: These are the dynamic arguments coming from beam search.
        It is a batch of variable size equal to the beam with at this specific time step
        :param initial_args: Initial beam search arguments. Might need modification
        :return: the modified args
        """
        return running_args, initial_args


def beam_search(model, beam_search_input, vocab_size, beam_size, stop_word, max_len, device='cuda'):
    """
    Performs beam search over a generative language model
    Please make sure the tensors stored in the input compel with the provided computing device.
    We do not check that here.
    :param model: An instance of the considered model
    :param beam_search_input: Instance of BeamSearchInput.
    Mandatory to work with MIMO models
    :param vocab_size: The size of the vocabulary in use. Usually len(tokenizer)
    :param beam_size: The beam size. The higher, the better, but slower
    :param stop_word: We save as completed a
    sentence that hits this stop word
    :param max_len: We will stop if the output reaches this size with no enough
    completed sequences
    :param device: The device to use. Defaults to CUDA
    :return: Returns a tuple consisting of the
    best completed sequence and the best running sequence, according to their scores. Note that each element in the
    tuple might be None if no sequences are found.
    """

    # ----------------------------------------------
    # INITIALIZATION
    # ----------------------------------------------
    args, seq_idx = beam_search_input.get_inputs(device=device)
    running_args = args[:]
    seq_len = len(args[seq_idx])
    running_sequences = args[seq_idx].expand(beam_size, -1)  # (beam_size, seq_len + step)
    running_sequence_scores = torch.zeros(beam_size, 1).to(device)
    complete_sequences = []
    complete_sequence_scores = []
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
            out = model(*running_args)
            logits = out[beam_search_input.logits_idx]  # (beam_size, seq_len + step, voc_size)

        running_args, args = beam_search_input.update_args(running_args, args)

        # Get predicted words in this beam batch
        preds = logits[:, -1]
        # Apply a softmax in order to have all values between 0 and 1
        preds = F.softmax(preds, dim=1)
        # Compute top k logits
        if step == 1:
            running_sequence_scores, top_k_words = preds[0].topk(k, dim=0)
        else:
            running_sequence_scores, top_k_words = preds.view(-1).topk(k, dim=0)

        # Select the most probable beams and update running sequences
        source_beam_ids = top_k_words / vocab_size
        next_most_probable_words = top_k_words % vocab_size
        selected_source_beams = running_sequences[source_beam_ids]  # (beam_size, seq_len + step)
        next_most_probable_words = next_most_probable_words.unsqueeze(1)  # (beam_size, 1)
        # Update running sequences
        running_sequences = torch.cat([selected_source_beams, next_most_probable_words], dim=1)

        # Check if any sequence reached the end
        if type(stop_word) != list:
            incomplete_sequence_ids = [i for i, next_word in enumerate(next_most_probable_words.squeeze(1)) if
                                       next_word.item() != stop_word]
        else:
            incomplete_sequence_ids = [i for i, next_word in enumerate(next_most_probable_words.squeeze(1)) if
                                       next_word.item() not in stop_word]

        complete_sequence_ids = list(
            set(range(len(next_most_probable_words.squeeze(1)))) - set(incomplete_sequence_ids))

        # If new sequences have been found, save them and decrease the beam size
        n_completed = len(complete_sequence_ids)
        if n_completed > 0:
            complete_sequences.extend(running_sequences[complete_sequence_ids].tolist())
            complete_sequence_scores.extend(running_sequence_scores[complete_sequence_ids])
            k -= n_completed
            running_sequences = running_sequences[incomplete_sequence_ids]

            # Reduce batch size
            for idx in range(len(running_args)):
                if idx != seq_idx:
                    # Expansion
                    running_args[idx] = args[idx].expand([k] + list(args[idx].shape))
                else:
                    running_args[idx] = running_sequences

        else:
            running_args[seq_idx] = running_sequences

        # Check if we need to stop
        # print('Current step = {}, Max = {}'.format(step, max_len))
        if step == max_len or k == 0:
            break
        else:
            step += 1

    # ----------------------------------------------
    # SELECTION
    # ----------------------------------------------

    if len(complete_sequences) > 0:
        best_complete_idx = complete_sequence_scores.index(max(complete_sequence_scores))
        best_complete = complete_sequences[best_complete_idx]
        best_complete = best_complete[seq_len:-1]  # Remove <eos> token and question

    else:
        best_complete = None

    running_sequences = running_sequences.tolist()
    running_sequence_scores = running_sequence_scores.tolist()

    if len(running_sequences) > 0:
        best_running_idx = running_sequence_scores.index(max(running_sequence_scores))
        best_running = running_sequences[best_running_idx]
        best_running = best_running[seq_len:]  # Remove question

    else:
        best_running = None

    return best_complete, best_running


def beam_search_with_softmaps(model, beam_search_input, vocab_size, beam_size, stop_word, max_len, softmap_batch_idx=1,
                              device='cuda'):
    """
    Performs beam search over a generative language model
    Please make sure the tensors stored in the input compel with the provided computing device.
    We do not check that here.
    :param beam_search_input: Instance of BeamSearchInput.
    Mandatory to work with MIMO models
    :param vocab_size: The size of the vocabulary in use. Usually len(tokenizer)
    :param beam_size: The beam size. The higher, the better, but slower
    :param stop_word: We save as completed a
    sentence that hits this stop word
    :param max_len: We will stop if the output reaches this size with no enough
    completed sequences
    :param softmap_batch_idx: Which is the index of the softmaps in the full model output tuple?
    :param device: The device to use. Defaults to CUDA
    :return: Returns a tuple consisting of the
    best completed sequence and the best running sequence, according to their scores. Note that each element in the
    tuple might be None if no sequences are found.
    Even the softmaps are returned here. Given a sequence of N elements, N-1 softmaps are generated.
    We don't know which was the softmap of the first element in the given sequence.
    """

    # ----------------------------------------------
    # INITIALIZATION
    # ----------------------------------------------
    args, seq_idx = beam_search_input.get_inputs(device=device)
    running_args = args[:]
    seq_len = len(args[seq_idx])
    running_sequences = args[seq_idx].expand(beam_size, -1)  # (beam_size, seq_len + step)
    running_sequences_outs = []
    running_sequence_scores = torch.zeros(beam_size, 1).to(device)
    complete_sequences = []
    complete_sequences_outs = []
    complete_sequence_scores = []
    step = 1
    k = beam_size  # Copy because we need to lively change this parameter

    # Args expansion to create a batch of size beam_size
    for idx in range(len(running_args)):
        if idx != seq_idx:
            # Expansion
            running_args[idx] = args[idx].expand([k] + list(args[idx].shape))
        else:
            running_args[idx] = running_sequences

    while True:
        # Get output logits
        with torch.no_grad():
            out = model(*running_args)
            logits = out[beam_search_input.logits_idx]  # (beam_size, seq_len + step, voc_size)
            softmaps = out[softmap_batch_idx]

        # Get predicted words in this beam batch
        preds = logits[:, -1]
        # Apply a softmax in order to have all values between 0 and 1
        preds = F.softmax(preds, dim=1)
        # Compute top k logits
        if step == 1:
            running_sequence_scores, top_k_words = preds[0].topk(k, dim=0)
        else:
            running_sequence_scores, top_k_words = preds.view(-1).topk(k, dim=0)

        # Select the most probable beams and update running sequences
        source_beam_ids = top_k_words / vocab_size
        next_most_probable_words = top_k_words % vocab_size
        selected_source_beams = running_sequences[source_beam_ids]  # (beam_size, seq_len + step)
        next_most_probable_words = next_most_probable_words.unsqueeze(1)  # (beam_size, 1)
        # Update running sequences
        running_sequences = torch.cat([selected_source_beams, next_most_probable_words], dim=1)
        running_sequences_outs = softmaps

        # Check if any sequence reached the end
        if type(stop_word) != list:
            incomplete_sequence_ids = [i for i, next_word in enumerate(next_most_probable_words.squeeze(1)) if
                                       next_word.item() != stop_word]
        else:
            incomplete_sequence_ids = [i for i, next_word in enumerate(next_most_probable_words.squeeze(1)) if
                                       next_word.item() not in stop_word]

        complete_sequence_ids = list(
            set(range(len(next_most_probable_words.squeeze(1)))) - set(incomplete_sequence_ids))

        # If new sequences have been found, save them and decrease the beam size
        n_completed = len(complete_sequence_ids)
        if n_completed > 0:
            complete_sequences.extend(running_sequences[complete_sequence_ids].tolist())
            complete_sequence_scores.extend(running_sequence_scores[complete_sequence_ids])
            complete_sequences_outs.extend(softmaps[complete_sequence_ids])
            k -= n_completed
            running_sequences = running_sequences[incomplete_sequence_ids]
            running_sequences_outs = softmaps[incomplete_sequence_ids]
            # Reduce batch size
            for idx in range(len(running_args)):
                if idx != seq_idx:
                    # Expansion
                    running_args[idx] = args[idx].expand([k] + list(args[idx].shape))
                else:
                    running_args[idx] = running_sequences


        else:
            running_args[seq_idx] = running_sequences

        # Check if we need to stop
        # print('Current step = {}, Max = {}'.format(step, max_len))
        if step == max_len or k == 0:
            break
        else:
            step += 1

    # ----------------------------------------------
    # SELECTION
    # ----------------------------------------------

    if len(complete_sequences) > 0:
        best_complete_idx = complete_sequence_scores.index(max(complete_sequence_scores))
        best_complete = complete_sequences[best_complete_idx]
        best_complete = best_complete[seq_len:]
        best_complete_output = complete_sequences_outs[best_complete_idx]

    else:
        best_complete = None
        best_complete_output = None

    running_sequences = running_sequences.tolist()
    running_sequence_scores = running_sequence_scores.tolist()

    if len(running_sequences) > 0:
        best_running_idx = running_sequence_scores.index(max(running_sequence_scores))
        best_running = running_sequences[best_running_idx]
        best_running = best_running[seq_len:]  # Remove question
        best_running_output = running_sequences_outs[best_running_idx]

    else:
        best_running = None
        best_running_output = None

    return best_complete, best_running, best_complete_output, best_running_output


if __name__ == '__main__':
    """
    Sample usage
    """

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

    # Encode a text inputs
    text = "What color is the car?"
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

    bc, br = beam_search(BeamSearchInput(model, 0, 0, tokens_tensor), vocab_size=len(tokenizer),
                         beam_size=25, stop_word=tokenizer.sep_token_id, max_len=50)

    print('Best completed:', tokenizer.decode(bc) if bc is not None else 'None')
    print('Best running:', tokenizer.decode(br) if br is not None else 'None')

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
from torch.utils.data import DataLoader
from utilities.evaluation.beam_search import beam_search
from datasets.creator import MultiPurposeDataset
from tqdm import tqdm
import random

def compute_corpus_bleu(model, dataset: MultiPurposeDataset, decode_fn, vocab_size, beam_size, stop_word, max_len,
                        device='cuda'):

    # Set the model in evaluation mode
    model.eval()
    model.to(device)

    # Prepare references and predictions
    references = []
    predictions = []

    # Prepare sequential batch loader
    loader = DataLoader(dataset=dataset, collate_fn=dataset.collate_fn, num_workers=4,
                        shuffle=True, pin_memory=True)

    print('Beam searching with size = {}'.format(beam_size))
    for batch in tqdm(loader):

        # Make beam search
        bc, br = beam_search(model, batch[0][0], vocab_size, beam_size, stop_word, max_len, device)
        # Append references
        references.append(batch[1][0])
        # Append best completed or best running
        predictions.append(bc) if bc is not None else predictions.append(br)


    print('Decoding & NLTK encoding predictions with the provided tokenizer..')
    predictions = list(map(decode_fn, predictions))

    # Compute BLEU score
    print('Computing BLEU score..')
    smf = SmoothingFunction()

    # BLEU-1
    score = corpus_bleu(references, predictions, smoothing_function=smf.method1, weights=(1, 0, 0, 0))

    print('BLEU = {}'.format(score))

    print('Random prediction & truths:')
    i = random.randint(0, len(predictions))
    print('Prediction', predictions[i])
    for t in references[i]:
        print('Reference', t)

    return score, predictions, references


def compute_sentences_bleu(model, dataset: MultiPurposeDataset, vocab_size, beam_size, stop_word, max_len,
                           device='cuda'):
    # Set the model in evaluation mode
    model.eval()
    model.to(device)

    # Prepare references and predictions
    bleu = []
    references = []
    predictions = []

    # Prepare sequential batch loader
    loader = DataLoader(dataset=dataset, batch_size=10, num_workers=4, shuffle=False, pin_memory=True)

    # Compute BLEU score
    print('Computing BLEU score..')
    smf = SmoothingFunction()

    print('Beam searching with size = {}'.format(beam_size))
    for batch in tqdm(loader):
        beam_search_input, ground_truths = dataset.get_bleu_inputs(model, batch, device)
        # Make beam search
        bc, br = beam_search(beam_search_input, vocab_size, beam_size, stop_word, max_len, device)
        # Append references
        references.append(ground_truths)
        # Append best completed or best running
        pred = bc if bc is not None else br
        predictions.append(pred)

        # BLEU-1
        score = sentence_bleu(ground_truths, pred, smoothing_function=smf.method1, weights=(1, 0, 0, 0))

        bleu.append(score)

    return bleu, predictions, references

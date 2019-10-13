from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from torch.utils.data import DataLoader
from utilities.evaluation.beam_search import BeamSearchDataset
from utilities.evaluation.beam_search import beam_search
from tqdm import tqdm
from nltk.util import ngrams


def compute_bleu(model, dataset: BeamSearchDataset, vocab_size, beam_size, stop_word, max_len, device='cuda'):
    # Set the model in evaluation mode
    model.eval()
    model.to(device)

    # Prepare references and predictions
    references = []
    predictions = []

    # Prepare sequential batch loader
    loader = DataLoader(dataset=dataset, batch_size=10, num_workers=4, shuffle=False, pin_memory=True)

    print('Beam searching with size = {}'.format(beam_size))
    for batch in tqdm(loader):
        beam_search_input, ground_truths = dataset.get_bleu_inputs(model, batch, device)
        # Make beam search
        bc, br = beam_search(beam_search_input, vocab_size, beam_size, stop_word, max_len, device)
        # Append references
        references.append(ground_truths)
        # Append best completed or best running
        predictions.append(bc) if bc is not None else predictions.append(br)

    # Compute BLEU score
    print('Computing BLEU score..')
    smf = SmoothingFunction()
    score = corpus_bleu(references, predictions, smoothing_function=smf.method1)

    print('BLEU = {}'.format(score))

    return score, predictions, references

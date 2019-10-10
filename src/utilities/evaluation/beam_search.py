import torch
from torch.functional import F
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def beam_search(model, out_filter, context, vocab_size, beam_size, stop_token_id):
    """
    This function applies beam search to extract the most likely sentence from a generative model.
    :param model: Instance of the model. Loaded from checkpoint.
    :param out_filter: Every model has its own output tuple, please extract only the last FC values with a lambda
    :param context: This will be the initial input to the model. QA / IA / VQA. This tuple will be passed straight ahead
            make sure the each element in the context tuple has already been unsqueezed
    :param vocab_size: We need this number to flatten the output tensors and compute the top_k probabilities
    :param beam_size: Size of the beam search. Search is exponential in the size of the beam.
    :param stop_token_id: When do we stop ? <stop> / <SEP> / <end> ? Pass the relative ID (i.e. an integer)
    :return: The searched sentence. List of ids. Use relative tokenizer / word map to decode it
    """

    # Step 1, get the model output logits
    output = out_filter(model(*context))
    # Remove batch fake dimension
    output = output.squeeze(0)
    # Get next word
    next_logits = output[-1]
    # Softmax these values
    next_logits = F.softmax(next_logits)
    # Compute top k logits
    top_k_scores, top_k_words = next_logits.topk(beam_size, 0, True, True)
    # From these logits create the initial batch for the next prediction

    return top_k_words.tolist()


if __name__ == '__main__':
    # Load pre-trained model (weights)
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model.eval()

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Encode a text inputs
    text = "Who was Jim Henson ? Jim Henson was a"
    indexed_tokens = tokenizer.encode(text)

    # Convert indexed tokens in a PyTorch tensor
    tokens_tensor = torch.tensor([indexed_tokens])
    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda').unsqueeze(0)
    model.to('cuda')

    out_filter = lambda out: out[0]

    preds = beam_search(model, out_filter, tokens_tensor, len(tokenizer), 5, tokenizer.eos_token_id)

    print(tokenizer.decode(preds))

    """
    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    # get the predicted next sub-word (in our case, the word 'man')
    predicted_index = torch.argmax(predictions[0, -1, :]).item()
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    assert predicted_text == 'Who was Jim Henson? Jim Henson was a man'
    """

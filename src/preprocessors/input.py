from pytorch_transformers import GPT2Tokenizer
from torchvision import transforms
import torch

# Load pre-trained model tokenizer (vocabulary)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def gpt2_tokenize(question):
    # Encode the question
    indexed_question = gpt2_tokenizer.encode(question)
    # Convert indexed tokens in a PyTorch tensor
    return torch.tensor([indexed_question])


def resnet50_normalize(image, size=224):
    normalization_pipe = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return normalization_pipe(image)


# This function prepares the inputs for out model
# It tokenizes the questions using the specified tokenizer
# It normalizes the images using the provided normalizer
def prepare(inputs, tokenize_fn=gpt2_tokenize, normalize_fn=resnet50_normalize):
    q, i = [], []
    for o in inputs:
        q.append(tokenize_fn(o.question))
        i.append(normalize_fn(o.load_image()))
    return q, i

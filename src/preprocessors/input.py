from pytorch_transformers import GPT2Tokenizer
from torchvision import transforms
import torch

# Load pre-trained model tokenizer (vocabulary)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def q_tokenizer_gpt2(question):
    # Encode the question
    indexed_question = gpt2_tokenizer.encode(question)
    # Convert indexed tokens in a PyTorch tensor
    return torch.tensor([indexed_question])


def i_normalizer_resnet50(image, size=224):
    tr = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return tr(image)


def prepare(inputs, q_tokenizer=q_tokenizer_gpt2, i_normalizer=i_normalizer_resnet50):
    q, i = [], []
    for o in inputs:
        q.append(q_tokenizer(o.question))
        i.append(i_normalizer(o.load_image()))
    return q, i

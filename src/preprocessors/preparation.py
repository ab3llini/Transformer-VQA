from pytorch_transformers import GPT2Tokenizer
from torchvision import transforms
import torch

# Load pre-trained model tokenizer (vocabulary)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def gpt2_tokenize(text):
    # Encode the question
    indexed_question = gpt2_tokenizer.encode(text, add_special_tokens=True)
    # Convert indexed tokens in a PyTorch tensor
    return torch.tensor([indexed_question])


def resnet50_normalize(image, size=224):
    normalization_pipe = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return normalization_pipe(image)


if __name__ == '__main__':
    print(gpt2_tokenizer.all_special_tokens)

    tk = gpt2_tokenizer.encode('hey how are you brother ?', add_special_tokens=False)
    print(tk)

    for t in tk:
        print(gpt2_tokenizer.decode(t))
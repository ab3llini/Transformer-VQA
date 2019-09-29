from torch.nn import CrossEntropyLoss

cross_entropy = CrossEntropyLoss(ignore_index=0)  # 0 = pad token in bert


def loss_fn(output, labels):
    """
    Loss function calculator
    :param output:
    :param labels:
    :return:
    """
    # Flatten the tensors (shift-align)
    # Remove last token from output
    output = output[..., :-1, :].contiguous().view(-1, output.size(-1))

    # Remove the first token from labels e do not care for question
    labels = (labels[..., 1:].contiguous()).view(-1)

    # Compute the actual loss
    return cross_entropy(output, labels)
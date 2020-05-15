from torch.nn import CrossEntropyLoss


class GPT2Loss(CrossEntropyLoss):
    def __init__(self, pad_token_id):
        super(GPT2Loss, self).__init__(ignore_index=pad_token_id)

    def forward(self, output, labels):
        """
        Loss function for gpt2
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
        return super(GPT2Loss, self).forward(output, labels)


class VisualGPT2Loss(GPT2Loss):
    def __init__(self, pad_token_id, extract=None):
        super(VisualGPT2Loss, self).__init__(pad_token_id=pad_token_id)
        if extract is not None:
            assert type(extract) == int, 'Extract value MUST be integer'

        self.extract = extract

    def forward(self, output, labels):
        if self.extract is not None:
            output = output[self.extract]

        # Compute the actual loss
        return super(VisualGPT2Loss, self).forward(output, labels[0])


class BERTLoss(CrossEntropyLoss):
    def __init__(self, pad_token_id):
        super(BERTLoss, self).__init__(ignore_index=pad_token_id)

    def forward(self, output, labels):
        """
        Loss function for gpt2
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
        return super(BERTLoss, self).forward(output, labels)

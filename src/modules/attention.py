import torch
from torch import nn


class CoAttention(nn.Module):
    def __init__(self, map_dim, hidden_dim, attention_dim=512):
        """
        :param map_dim: Number of maps (i.e. channels) in the used CNN. Usually map_dim = maps.shape[-1]
        :param hidden_dim: Size of hidden layer of the language model. Usually hidden_size = hiddens.shape[-1]
        :param attention_dim: Dimension of the attention layers
        """
        super(CoAttention, self).__init__()
        # Set parameters
        self.map_dim = map_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        # Prepare linear layers
        self.map_linear = nn.Linear(in_features=self.map_dim, out_features=self.attention_dim)
        self.hidden_linear = nn.Linear(in_features=self.hidden_dim, out_features=self.attention_dim)
        self.co_att_linear = nn.Linear(in_features=self.map_dim, out_features=self.hidden_dim)
        self.rectified_linear = nn.Linear(in_features=self.attention_dim, out_features=1)
        self.pixel_softmax = nn.Softmax(dim=2)  # Make sure this is 2 and not 1. We softmax over the pixels!

        # Prepare the rectifier
        self.rectify = nn.ReLU()

    def forward(self, maps, hiddens):
        """
        Where the magic happens :)
        In this method we will use the hidden values of the used language model to
        generate masks (softmaps) over the maps coming from the used CNN.
        The procedure involves summations, pointwise multiplications and linear transformations.
        :param maps: The maps coming from the last layer of the used CNN. (batch_size, x, y, map_dim)
        :param hiddens: The hidden vector coming from the last transformer layer (batch_size, n_tokens, hidden_size)
        :return: co-attended vector (batch_size, n_tokens, hidden_size), softmap (batch_size, x * y, n_maps)
        Note: For performance reasons we do not reshape the softmap to a 2D representation. We leave it flattened.
        In case you need to plot it, first reshape it.
        """

        # Obtain basic information
        batch_size = maps.size(0)

        # Flatten each map from x, y to x * y, where x = n_rows, y = n_cols, [x, y] = pixel
        flattened_maps = maps.view(batch_size, -1, self.map_dim)

        # Go to attention space with linear layers
        map_linear_out = self.map_linear(flattened_maps)
        hidden_linear_out = self.hidden_linear(hiddens)

        # Make the two resulting vectors ready for pointwise multiplication
        # Add a fake dimension right after the batch size (accounts for tokens)
        map_linear_out = map_linear_out.unsqueeze(1)
        # Add a fake dimension after the number of tokens (accounts for pixels)
        hidden_linear_out = hidden_linear_out.unsqueeze(2)

        # Sum the two vectors -> (batch_size, n_tokens, x * y, map_dim)
        pointwise_sum = map_linear_out + hidden_linear_out

        # Rectify the result and get rid of negative values + help vanishing gradients
        rectified_sum = self.rectify(pointwise_sum)

        # Combine the maps channels together and prepare them for the softmax (batch_size, n_tokens, x * y)
        rectified_linear_out = self.rectified_linear(rectified_sum).squeeze(3)

        # Create the softmap using a softmax function over all the pixels (batch_size, n_tokens, x*y), x*y in [0,1]
        pixel_softmax_out = self.pixel_softmax(rectified_linear_out)

        # Pointwise multiply the softmap with the original maps (batch_size, n_tokens, x*y, map_dim)
        # Unsqueeze(1) to add a fake dimension to sum the maps over all the tokens
        # Unsqueeze(3) to add a fake dimension to sum the same vector to all the original maps
        pointwise_mul = flattened_maps.unsqueeze(1) * pixel_softmax_out.unsqueeze(3)

        # Sum the pixels together, in each map, to get a unified representation (batch_size, n_tokens, map_dim)
        pointwise_mul = pointwise_mul.sum(dim=2)

        # At this point we expand (or reduce) the resulting tensor to match the hidden one
        co_att_linear_out = self.co_att_linear(pointwise_mul)  # (batch_size, n_tokens, hidden)

        # We finally pointwise multiply the latter output with the original hidden tensor
        co_att_out = co_att_linear_out * hiddens

        # Return the computed tensors
        return co_att_out, pixel_softmax_out


class LightAttention(nn.Module):
    def __init__(self, map_dim, hidden_dim):
        """
        :param map_dim: Number of maps (i.e. channels) in the used CNN. Usually map_dim = maps.shape[-1]
        :param hidden_dim: Size of hidden layer of the language model. Usually hidden_size = hiddens.shape[-1]
        """
        super(LightAttention, self).__init__()
        # Set parameters
        self.map_dim = map_dim
        self.hidden_dim = hidden_dim

        # Prepare linear layers
        self.map_linear = nn.Linear(in_features=self.map_dim, out_features=self.hidden_dim)
        self.final_linear = nn.Linear(in_features=self.map_dim, out_features=self.hidden_dim)
        self.rectified_linear = nn.Linear(in_features=self.hidden_dim, out_features=1)
        self.pixel_softmax = nn.Softmax(dim=2)  # Make sure this is 2 and not 1. We softmax over the pixels!

        with torch.no_grad():
            self.final_linear.weight = torch.zeros((self.hidden_dim, self.map_dim))

        # Prepare the rectifier
        self.rectify = nn.ReLU()

    def forward(self, maps, hiddens):
        """
        Where the magic happens :)
        In this method we will use the hidden values of the used language model to
        generate masks (softmaps) over the maps coming from the used CNN.
        The procedure involves summations, pointwise multiplications and linear transformations.
        :param maps: The maps coming from the last layer of the used CNN. (batch_size, x, y, map_dim)
        :param hiddens: The hidden vector coming from the last transformer layer (batch_size, n_tokens, hidden_size)
        :return: co-attended vector (batch_size, n_tokens, hidden_size), softmap (batch_size, x * y, n_maps)
        Note: For performance reasons we do not reshape the softmap to a 2D representation. We leave it flattened.
        In case you need to plot it, first reshape it.
        """

        # Obtain basic information
        batch_size = maps.size(0)

        # Flatten each map from x, y to x * y, where x = n_rows, y = n_cols, [x, y] = pixel
        flattened_maps = maps.view(batch_size, -1, self.map_dim)

        # Go to attention space with linear layers
        map_linear_out = self.map_linear(flattened_maps)

        # Make the two resulting vectors ready for pointwise multiplication
        # Add a fake dimension right after the batch size (accounts for tokens)
        map_linear_out = map_linear_out.unsqueeze(1)
        # Add a fake dimension after the number of tokens (accounts for pixels)
        hidden_linear_out = hiddens.unsqueeze(2)

        # Sum the two vectors -> (batch_size, n_tokens, x * y, map_dim)
        pointwise_sum = map_linear_out + hidden_linear_out

        # Rectify the result and get rid of negative values + help vanishing gradients
        rectified_sum = self.rectify(pointwise_sum)

        # Combine the maps channels together and prepare them for the softmax (batch_size, n_tokens, x * y)
        rectified_linear_out = self.rectified_linear(rectified_sum).squeeze(3)

        # Create the softmap using a softmax function over all the pixels (batch_size, n_tokens, x*y), x*y in [0,1]
        pixel_softmax_out = self.pixel_softmax(rectified_linear_out)

        # Pointwise multiply the softmap with the original maps (batch_size, n_tokens, x*y, map_dim)
        # Unsqueeze(1) to add a fake dimension to sum the maps over all the tokens
        # Unsqueeze(3) to add a fake dimension to sum the same vector to all the original maps
        pointwise_mul = flattened_maps.unsqueeze(1) * pixel_softmax_out.unsqueeze(3)

        # Sum the pixels together, in each map, to get a unified representation (batch_size, n_tokens, map_dim)
        pointwise_mul = pointwise_mul.sum(dim=2)

        # At this point we expand (or reduce) the resulting tensor to match the hidden one
        out = self.final_linear(pointwise_mul)  # (batch_size, n_tokens, hidden)

        # We finally pointwise multiply the latter output with the original hidden tensor
        out = out + hiddens

        # Return the computed tensors
        return out, pixel_softmax_out

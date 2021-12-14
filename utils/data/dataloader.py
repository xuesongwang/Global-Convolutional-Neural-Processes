import torch
import numpy as np
__all__ = ["cntxt_trgt_collate", "context_target_split", "context_target_split2d", "cntxt_trgt_collate_covid",
           "sequential_input_to_img"]


def cntxt_trgt_collate(get_cntxt_trgt, is_duplicate_batch=False, **kwargs):
    """Transformes and collates inputs to neural processes given the whole input.

    Parameters
    ----------
    get_cntxt_trgt : callable
        Function that takes as input the features and tagrets `X`, `y` and return
        the corresponding `X_cntxt, Y_cntxt, X_trgt, Y_trgt`.

    is_duplicate_batch : bool, optional
        Wether to repeat th`e batch to have 2 different context and target sets
        for every function. If so the batch will contain the concatenation of both.
    """

    def mycollate(batch):
        collated = torch.utils.data.dataloader.default_collate(batch)
        X = collated[0]
        y = collated[1]

        if is_duplicate_batch:
            X = torch.cat([X, X], dim=0)
            if y is not None:
                y = torch.cat([y, y], dim=0)
            y = torch.cat([y, y], dim=0)

        X_cntxt, Y_cntxt, X_trgt, Y_trgt = get_cntxt_trgt(X, y, **kwargs)
        inputs = dict(X_cntxt=X_cntxt, Y_cntxt=Y_cntxt, X_trgt=X_trgt, Y_trgt=Y_trgt)
        targets = Y_trgt

        return inputs, targets

    return mycollate

def cntxt_trgt_collate_covid(get_cntxt_trgt, is_duplicate_batch=False, **kwargs):
    """Transformes and collates inputs to neural processes given the whole input.

    Parameters
    ----------
    get_cntxt_trgt : callable
        Function that takes as input the features and tagrets `X`, `y` and return
        the corresponding `X_cntxt, Y_cntxt, X_trgt, Y_trgt`.

    is_duplicate_batch : bool, optional
        Wether to repeat th`e batch to have 2 different context and target sets
        for every function. If so the batch will contain the concatenation of both.
    """

    def mycollate(batch):
        collated = torch.utils.data.dataloader.default_collate(batch)
        X = collated[0]

        X_cntxt, Y_cntxt, X_trgt, Y_trgt = get_cntxt_trgt(X, None, **kwargs)
        inputs = dict(X_cntxt=X_cntxt, Y_cntxt=Y_cntxt, X_trgt=X_trgt, Y_trgt=Y_trgt)
        targets = Y_trgt

        return inputs, targets

    return mycollate

def context_target_split(dataset, num_context=50, num_extra_target=50):
    """
    dataset:  torch.utils.data.Dataset
    """
    import numpy as np
    data_index = np.random.randint(len(dataset))
    x, y = dataset[data_index]

    # num_context = np.random.randint(3, max_num_context)
    # num_extra_target = np.random.randint(3, max_num_extra_target)

    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)

    inds = np.random.choice(range(x.shape[1]), size=(num_context + num_extra_target), replace=False)
    context_x = x[:, inds][:, :num_context]
    context_y = y[:, inds][:, :num_context]

    target_x = x[:, inds][:, num_context:]
    target_y = y[:, inds][:, num_context:]
    return context_x, context_y, target_x, target_y


def context_target_split2d(dataset, prob_low, prob_high, batch_size = 1, p=None, convgrid=False):
    data_index = np.random.randint(0, high=len(dataset), size=batch_size)
    img = []
    for i in range(batch_size):
        img.append(dataset[data_index[i]][0])
    img = torch.stack(img)
    context_mask, target_mask = generate_mask(img, prob_low, prob_high, p)
    img_masked = img * context_mask.unsqueeze(1)
    if not convgrid:
        x_context, y_context, x_target, y_target = img_mask_to_np_input(img,context_mask, target_mask,include_context=True)
    else:
        x_context = context_mask.unsqueeze(1).permute(0, 2, 3, 1)
        y_context = img_masked.permute(0, 2, 3, 1)
        x_target = target_mask.unsqueeze(1).permute(0, 2, 3, 1)
        y_target = img.permute(0, 2, 3, 1)
    return x_context, y_context, x_target, y_target, img_masked



def img_mask_to_np_input(img, context_mask, target_mask, include_context = False, normalize=True):
    """
    Given an image and two masks, return x and y tensors expected by Neural
    Process. Specifically, x will contain indices of unmasked points, e.g.
    [[1, 0], [23, 14], [24, 19]] and y will contain the corresponding pixel
    intensities, e.g. [[0.2], [0.73], [0.12]] for grayscale or
    [[0.82, 0.71, 0.5], [0.42, 0.33, 0.81], [0.21, 0.23, 0.32]] for RGB.

    Parameters
    ----------
    img : torch.Tensor
        Shape (N, C, H, W). Pixel intensities should be in [0, 1]

    context_mask : torch.Tensor
        Context mask where 0 corresponds to masked pixel and 1 to a visible
        pixel. Shape (N, H, W). Note the number of unmasked pixels must be the
        SAME for every mask in batch.

    target_mask : torch.Tensor
        Target mask where 0 corresponds to masked pixel and 1 to a visible
        pixel. Shape (N, H, W). Note the number of unmasked pixels must be the
        SAME for every mask in batch.

    normalize : bool
        If true normalizes pixel locations x to [-1, 1]
    """
    batch_size, num_channels, height, width = img.size()
    # Create a mask which matches exactly with image size which will be used to
    # extract pixel intensities
    context_mask_img_size = context_mask.unsqueeze(1).repeat(1, num_channels, 1, 1)
    target_mask_img_size = target_mask.unsqueeze(1).repeat(1, num_channels, 1, 1)
    # Number of points corresponds to number of visible pixels in mask, i.e. sum
    # of non zero indices in a mask (here we assume every mask has same number
    # of visible pixels)
    num_context_points = context_mask[0].nonzero().size(0)
    num_target_points  = target_mask[0].nonzero().size(0)

    # Compute non zero indices
    # Shape (num_nonzeros, 3), where each row contains index of batch, height and width of nonzero
    context_nonzero_idx = context_mask.nonzero()
    target_nonzero_idx = target_mask.nonzero()

    # The x tensor for Neural Processes contains (height, width) indices, i.e.
    # 1st and 2nd indices of nonzero_idx (in zero based indexing)
    x_context = context_nonzero_idx[:, 1:].view(batch_size, num_context_points, 2).float()
    # The y tensor for Neural Processes contains the values of non zero pixels
    y_context = img[context_mask_img_size].view(batch_size, num_channels, num_context_points)
    # Ensure correct shape, i.e. (batch_size, num_points, num_channels)
    y_context = y_context.permute(0, 2, 1)

    x_target = target_nonzero_idx[:, 1:].view(batch_size, num_target_points, 2).float()
    y_target = img[target_mask_img_size].view(batch_size, num_channels, num_target_points)
    y_target = y_target.permute(0, 2, 1)

    if normalize:
        # TODO: make this separate for height and width for non square image
        # Normalize x to [-1, 1]
        x_context = (x_context - float(height) / 2) / (float(height) / 2)
        x_target = (x_target - float(height) / 2) / (float(height) / 2)

    if include_context:
        x_target = torch.cat([x_target, x_context], dim=1)
        y_target = torch.cat([y_target, y_context], dim=1)
    return x_context, y_context, x_target, y_target

def generate_mask(img, prob_low=0.01, prob_high=0.8, p = None):
    """
    return a context mask and a target mask
    Args:
        img: input images, shape: [B, C, H, W]
    Returns:
        context mask, shape: [B, H, W]
        target mask, shape: [B, H, W]
    """
    batch_size = img.size(0)
    prob = np.random.uniform(prob_low, prob_high)
    if p is not None:
        prob = p
    # num_extra_target = int(torch.empty(1).uniform_(n_total / 100, n_total / 2).item())
    context_mask = img.new_empty(img.size(2), img.size(3)).bernoulli_(p=prob).bool()
    target_mask = ~context_mask
    context_mask = context_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    target_mask = target_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    return context_mask, target_mask



def sequential_input_to_img(x, y, img_size, spatial_factor =1, return_plot = "z_samples"):
    """Given an x and y returned by a Neural Process, reconstruct image.
    Missing pixels will have a value of 0.

    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, 2) containing normalized indices.

    y : torch.Tensor
        Shape (n_z_samples, batch_size, num_points, num_channels) where num_channels = 1 for
        grayscale and 3 for RGB, containing normalized pixel intensities.

    img_size : tuple of ints
        [B, C, H, W].
    return_plot: "z_samples" / "batch", return a batch of images or one image with different z_samples

    Returns:
        a image with size
    """
    batch_size, channel, height, width = img_size
    n_z_samples = y.shape[0]
    # Unnormalize x and y
    x = x /spatial_factor * float(height / 2) + float(height / 2)
    x = x.long()

    # Permute y so it matches order expected by image
    if return_plot == 'batch':
        # (n_z_samples, batch_size, num_points, num_channels) -> (batch_size, num_channels, num_points)
        y = y[0]  # return the first z_sample element
        y = y.permute(0, 2, 1)
        # Initialize empty image
        img = torch.zeros(batch_size, channel, height+1, width+1)
        for i in range(batch_size):
            img[i, :, x[i, :, 0], x[i, :, 1]] = y[i, :, :].cpu()
        return img
    else:
        # (n_z_samples, batch_size, num_points, num_channels) -> (n_z_samples, num_channels, num_points)
        y = y[:,0,:,:] # return the first batch element
        y = y.permute(0, 2, 1)
        img = torch.zeros(n_z_samples, *img_size[1:])
        for i in range(n_z_samples):
            img[i, :, x[0, :, 0], x[0, :, 1]] = y[i, :, :].cpu()
        return img
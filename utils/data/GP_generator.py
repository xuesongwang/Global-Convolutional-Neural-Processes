import numpy as np
import collections
import torch
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, Matern
from sklearn.gaussian_process import GaussianProcessRegressor

# The NP takes as input a `NPRegressionDescription` namedtuple with fields:
#   `query`: a tuple containing ((context_x, context_y), target_x)
#   `target_y`: a tesor containing the ground truth for the targets to be
#     predicted
#   `num_total_points`: A vector containing a scalar that describes the total
#     number of datapoints used (context + target)
#   `num_context_points`: A vector containing a scalar that describes the number
#     of datapoints used as context
# The GPCurvesReader returns the newly sampled data in this format at each
# iteration

NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "y_target", "num_total_points", "num_context_points"))


class GPCurvesReader(object):
    """Generates curves using a Gaussian Process (GP).

    Supports vector inputs (x) and vector outputs (y). Kernel is
    mean-squared exponential, using the x-value l2 coordinate distance scaled by
    some factor chosen randomly in a range. Outputs are independent gaussian
    processes.
    """
    def __init__(self,
                 kernel,
                 batch_size,
                 max_num_context,
                 x_size=1,
                 y_size=1,
                 testing=False,
                 device = torch.device("cpu")):
        """Creates a regression dataset of functions sampled from a GP.

        Args:
        kernel: kernel type, "EQ" or "period"
        batch_size: An integer.
        max_num_context: The max number of observations in the context.
        x_size: Integer >= 1 for length of "x values" vector.
        y_size: Integer >= 1 for length of "y values" vector.
        l1_scale: Float; typical scale for kernel distance function.
        sigma_scale: Float; typical scale for variance.
        testing: Boolean that indicates whether we are testing. If so there are
        more targets for visualization.
        """
        # Pass the x_values through the Gaussian kernel
        if kernel == 'RBF_Kernel':
            self.kernel = RBF(length_scale=(0.2))
        elif kernel == 'Periodic_Kernel':
            self.kernel = ExpSineSquared(length_scale=0.5, periodicity=0.5)
        elif kernel == 'matern':
            self.kernel = Matern(length_scale=0.2, nu=1.5)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.005)
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._testing = testing
        self.device = device

    def _rbf_kernel(self, xdata, l1 = 0.4, sigma_f = 1.0, sigma_noise=2e-2):
        """Applies the Gaussian kernel to generate curve data
            we use the same kernel parameter for the whole training process
            instead of using dynamic parameters
        Args:
        xdata: Tensor with shape `[batch_size, num_total_points, x_size]` with
            the values of the x-axis data.
        l1: Tensor with shape `[batch_size, y_size, x_size]`, the scale
            parameter of the Gaussian kernel.
        sigma_f: Float tensor with shape `[batch_size, y_size]`; the magnitude
            of the std.
        sigma_noise: Float, std of the noise that we add for stability.

        Returns:
        The kernel, a float tensor with shape
        `[batch_size, y_size, num_total_points, num_total_points]`.
        """
        # Set kernel parameters
        l1 = torch.ones([self._batch_size, self._y_size, self._x_size]).to(self.device) * l1
        sigma_f = torch.ones([self._batch_size, self._y_size]).to(self.device) * sigma_f

        num_total_points = xdata.size(1)
        # Expand and take the difference
        xdata1 = torch.unsqueeze(xdata, dim=1)  # [B, 1, num_total_points, x_size]
        xdata2 = torch.unsqueeze(xdata, dim=2)  # [B, num_total_points, 1, x_size]
        diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]

        # [B, y_size, num_total_points, num_total_points, x_size]
        norm = (diff[:, None, :, :, :] / l1[:, :, None, None, :])**2

        norm = torch.sum(norm,  dim=-1)  # [B, y_size, num_total_points, num_total_points]

        # [B, y_size, num_total_points, num_total_points]
        kernel = (sigma_f**2)[:, :, None, None] * torch.exp(-0.5 * norm)

        # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise**2) * torch.eye(num_total_points).to(self.device)

        return kernel

    def _periodic_kernel(self, xdata, l1 = 1.0, p = 1.0, sigma_f = 1.0, sigma_noise=2e-2):
        """Applies the periodic kernel to generate curve data
            we use the same kernel parameter for the whole training process
            instead of using dynamic parameters
        Args:
        xdata: Tensor with shape `[batch_size, num_total_points, x_size]` with
            the values of the x-axis data.
        l1:Tensor with shape `[batch_size, y_size, x_size]`, the scale
            parameter of the Gaussian kernel.
        p:  Tensor with the shape `[batch_size, y_size, x_size]`, the distance between repetitions.
        sigma_f: Float tensor with shape `[batch_size, y_size]`; the magnitude
            of the std.
        sigma_noise: Float, std of the noise that we add for stability.

        Returns:
        The kernel, a float tensor with shape
        `[batch_size, y_size, num_total_points, num_total_points]`.
        """
        l1 = torch.ones([self._batch_size, self._y_size, self._x_size]).to(self.device) * l1
        sigma_f = torch.ones([self._batch_size, self._y_size]).to(self.device) * sigma_f

        num_total_points = xdata.size(1)
        # Expand and take the difference
        xdata1 = torch.unsqueeze(xdata, dim=1)  # [B, 1, num_total_points, x_size]
        xdata2 = torch.unsqueeze(xdata, dim=2)  # [B, num_total_points, 1, x_size]

        diff = np.pi*torch.abs(xdata1 - xdata2)/p  # [B, num_total_points, num_total_points, x_size]

        # [B, y_size, num_total_points, num_total_points, x_size]
        norm = (diff[:, None, :, :, :] / l1[:, :, None, None, :]) ** 2
        norm = torch.sum(norm, dim=-1) # [B, y_size, num_total_points, num_total_points]

        # [B, y_size, num_total_points, num_total_points]
        kernel = (sigma_f ** 2)[:, :, None, None] * torch.exp(-2 * norm)

        # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise ** 2) * torch.eye(num_total_points).to(self.device)

        return kernel

    def _matern_kernel(self):
        # num_total_points = xdata.size(1)
        # # Expand and take the difference
        # xdata1 = torch.unsqueeze(xdata, dim=1)  # [B, 1, num_total_points, x_size]
        # xdata2 = torch.unsqueeze(xdata, dim=2)  # [B, num_total_points, 1, x_size]
        #
        # d = 4 * torch.abs(xdata1 - xdata2) # [B, num_total_points, num_total_points, x_size]
        # d = torch.sum(d, dim=-1).unsqueeze(dim=1)  # [B, y_size, num_total_points, num_total_points]
        # kernel = (1 + 4*5**0.5*d + 5.0/3.0*d**2) * torch.exp(-5**(0.5)*d)
        # # kernel += (sigma_noise ** 2) * torch.eye(num_total_points).to(self.device)
        import stheno.torch as stheno
        import stheno as sth
        kernel = stheno.Matern52().stretch(0.25)
        gp = stheno.GP(kernel, graph=sth.Graph())
        return gp

    def generate_with_Matern(self, gp):
        num_points = 200
        x_all = np.linspace(-2., 2., num_points)
        y_all = gp(x_all).sample()
        #     y_all = gp_.sample(x_all)
        x_all = torch.tensor(x_all, dtype=torch.float)[None, :, None]
        y_all = torch.tensor(y_all, dtype=torch.float).unsqueeze(0)
        self._batch_size = 1
        return x_all, y_all

    def generate_curves(self, include_context = True, x_min = -2, x_max = 2, sort=False):
        """Builds the op delivering the data.

        Generated functions are `float32` with x values between -2 and 2.
        Args:
            include_context:  Whether to include context data to target data, useful for NP, CNP, ANP
        Returns:
            A `NPRegressionDescription` namedtuple.
        """
        num_context = np.random.randint(3, self._max_num_context)

        # If we are testing we want to have more targets and have them evenly
        # distributed in order to plot the function.
        if self._testing:
            num_total_points = 400
            num_target = num_total_points - num_context
            x_values = torch.arange(x_min, x_max, 0.01)[None, :, None].repeat([self._batch_size, 1, 1]).to(self.device)
        # During training the number of target points and their x-positions are
        # selected at random
        else:
            num_target = np.random.randint(3, self._max_num_context)
            num_total_points = num_context + num_target
            # shape: (n_points, x_dim)
            x_values = np.random.rand(num_total_points, self._x_size)*(x_max - x_min) + x_min

        # shape: (n_points, batch_size)
        y_values = self.gp.sample_y(x_values, n_samples=self._batch_size)
        if sort == True:
            index_sorted = np.argsort(x_values)
            x_values = x_values[index_sorted]
            y_values = y_values[index_sorted]

        # convert to tensor and match expected shape
        x_values = torch.tensor(x_values, dtype=torch.float, device=self.device)
        ## rescale X following the website
        x_values = self.rescale_range(x_values)
        # shape: (bs, n_points, x_dim)
        x_values = x_values.unsqueeze(0).repeat(self._batch_size, 1, 1)
        # shape: (bs, n_points, x_dim)
        y_values = torch.tensor(y_values.T, dtype=torch.float, device=self.device).unsqueeze(dim=-1)

        # # scale
        # # scale = 2*np.random.rand() + 1 #scale by a factor from [1, 3)
        # # bias = 3*np.random.rand() #bias by [0,3)
        # scale = 1
        # bias = 0
        # y_values = y_values * scale + bias

        if self._testing:
            # Select the observations
            idx = np.random.permutation(num_total_points)
            context_x = x_values[:, idx[:num_context], :]
            context_y = y_values[:, idx[:num_context], :]
            target_x = x_values[:, idx[num_context:], :]
            target_y = y_values[:, idx[num_context:], :]
        else:

            if include_context:
                # Select the targets which constitute the context points as well as
                # some new target points
                target_x = x_values[:, :(num_target + num_context), :]
                target_y = y_values[:, :(num_target + num_context), :]
            else:
                target_x = x_values[:, num_context :(num_target + num_context), :]
                target_y = y_values[:, num_context :(num_target + num_context), :]
            # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
            query=query,
            y_target=target_y,
            num_total_points= num_target + num_context,
            num_context_points=num_context)

    def generate_temporal_curves(self, max_num_context = 10, include_context = True):
        """Builds the op delivering the data.

                Generated functions are `float32` with x values between -2 and 0 for context data, and 0 to 2 for target data.
                Args:
                    max_num_context:  we set max_num_context to be smaller(25) than generate curve for sequential data
                    include_context:  Whether to include context data to target data, useful for NP, CNP, ANP
                Returns:
                    A `NPRegressionDescription` namedtuple.
                """
        self._max_num_context = max_num_context
        num_context =  self._max_num_context # fixed sequence

        # the number of target points and their x-positions are
        # selected at random
        num_target = self._max_num_context #fixed sequence

        num_total_points = num_context + num_target
        x_context = -2 * torch.rand([self._batch_size, num_context, self._x_size]).to(self.device)
        x_target = 2 * torch.rand([self._batch_size, num_target, self._x_size]).to(self.device)
        x_values = torch.cat([x_context, x_target], dim=1)

        # Pass the x_values through the Gaussian kernel
        # [batch_size, y_size, num_total_points, num_total_points]
        if self.kernel == 'EQ':
            kernel = self._rbf_kernel(x_values)  # [B, y_size, num_total_points, num_total_points]
        elif self.kernel == 'period':
            kernel = self._periodic_kernel(x_values)

        # Calculate Cholesky, using double precision for better stability:
        cholesky = torch.cholesky(kernel.type(torch.float64)).type(
            torch.float32)  # [B, y_size, num_total_points, num_total_points]

        # Sample a curve
        # [batch_size, y_size, num_total_points, 1]
        y_values = torch.matmul(cholesky,
                                torch.rand([self._batch_size, self._y_size, num_total_points, 1]).to(self.device))

        # [batch_size, num_total_points, y_size]
        y_values = y_values.squeeze(-1).permute([0, 2, 1])

        if include_context:
            # Select the targets which constitute the context points as well as
            # some new target points
            target_x = x_values[:, :num_target + num_context, :]
            target_y = y_values[:, :num_target + num_context, :]
        else:
            target_x = x_values[:, num_context:num_target + num_context, :]
            target_y = y_values[:, num_context:num_target + num_context, :]
        # Select the observations
        context_x = x_values[:, :num_context, :]
        context_y = y_values[:, :num_context, :]

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
            query=query,
            y_target=target_y,
            num_total_points=num_target + num_context,
            num_context_points=num_context)

    def rescale_range(self, X, old_range=(-2, 2), new_range=(-1, 1)):
        """Rescale X linearly to be in `new_range` rather than `old_range`."""
        old_min = old_range[0]
        new_min = new_range[0]
        old_delta = old_range[1] - old_min
        new_delta = new_range[1] - new_min
        return (((X - old_min) * new_delta) / old_delta) + new_min

import logging
from functools import partial

import numpy as np
import sklearn
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from torch.utils.data import Dataset

from npf.utils.helpers import rescale_range
from .helpers import DIR_DATA, NotLoadedError, load_chunk, save_chunk

class CustomizedGPDataset(Dataset):
    """Generates curves using a Gaussian Process (GP).

    Supports vector inputs (x) and vector outputs (y). Kernel is
    mean-squared exponential, using the x-value l2 coordinate distance scaled by
    some factor chosen randomly in a range. Outputs are independent gaussian
    processes.
    """
    def __init__(self,
                 kernel,
                 max_num_context,
                 x_size=1,
                 y_size=1,
                 testing=False,
                 n_samples=1000,
                 device = torch.device("cpu")):
        """Creates a regression dataset of functions sampled from a GP.

        Args:
        kernel: kernel type, "EQ" or "period"
        batch_size: An integer.
        max_num_context: The max number of observations in the context.
        x_size: Integer >= 1 for length of "x values" vector.
        y_size: Integer >= 1 for length of "y values" vector.
        l1_scale: Float; typical scale for kernel distance function.
        sigma_scale: Float; typical scale for variance.
        testing: Boolean that indicates whether we are testing. If so there are
        more targets for visualization.
        """
        # Pass the x_values through the Gaussian kernel
        if kernel == 'EQ':
            self.kernel = RBF(length_scale=(0.2))
        elif kernel == 'period':
            self.kernel = ExpSineSquared(length_scale=0.5, periodicity=0.5)
        elif kernel == 'matern':
            self.kernel = Matern(length_scale=0.2, nu=1.5)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.005)
        # self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._testing = testing
        self.n_samples = n_samples
        self.device = device
        self.sort = False
        # self.precompute_xy()

    def _rbf_kernel(self, xdata, l1 = 0.4, sigma_f = 1.0, sigma_noise=2e-2):
        """Applies the Gaussian kernel to generate curve data
            we use the same kernel parameter for the whole training process
            instead of using dynamic parameters
        Args:
        xdata: Tensor with shape `[batch_size, num_total_points, x_size]` with
            the values of the x-axis data.
        l1: Tensor with shape `[batch_size, y_size, x_size]`, the scale
            parameter of the Gaussian kernel.
        sigma_f: Float tensor with shape `[batch_size, y_size]`; the magnitude
            of the std.
        sigma_noise: Float, std of the noise that we add for stability.

        Returns:
        The kernel, a float tensor with shape
        `[batch_size, y_size, num_total_points, num_total_points]`.
        """
        # Set kernel parameters
        l1 = torch.ones([self._batch_size, self._y_size, self._x_size]).to(self.device) * l1
        sigma_f = torch.ones([self._batch_size, self._y_size]).to(self.device) * sigma_f

        num_total_points = xdata.size(1)
        # Expand and take the difference
        xdata1 = torch.unsqueeze(xdata, dim=1)  # [B, 1, num_total_points, x_size]
        xdata2 = torch.unsqueeze(xdata, dim=2)  # [B, num_total_points, 1, x_size]
        diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]

        # [B, y_size, num_total_points, num_total_points, x_size]
        norm = (diff[:, None, :, :, :] / l1[:, :, None, None, :])**2

        norm = torch.sum(norm,  dim=-1)  # [B, y_size, num_total_points, num_total_points]

        # [B, y_size, num_total_points, num_total_points]
        kernel = (sigma_f**2)[:, :, None, None] * torch.exp(-0.5 * norm)

        # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise**2) * torch.eye(num_total_points).to(self.device)

        return kernel

    def _periodic_kernel(self, xdata, l1 = 1.0, p = 1.0, sigma_f = 1.0, sigma_noise=2e-2):
        """Applies the periodic kernel to generate curve data
            we use the same kernel parameter for the whole training process
            instead of using dynamic parameters
        Args:
        xdata: Tensor with shape `[batch_size, num_total_points, x_size]` with
            the values of the x-axis data.
        l1:Tensor with shape `[batch_size, y_size, x_size]`, the scale
            parameter of the Gaussian kernel.
        p:  Tensor with the shape `[batch_size, y_size, x_size]`, the distance between repetitions.
        sigma_f: Float tensor with shape `[batch_size, y_size]`; the magnitude
            of the std.
        sigma_noise: Float, std of the noise that we add for stability.

        Returns:
        The kernel, a float tensor with shape
        `[batch_size, y_size, num_total_points, num_total_points]`.
        """
        l1 = torch.ones([self._batch_size, self._y_size, self._x_size]).to(self.device) * l1
        sigma_f = torch.ones([self._batch_size, self._y_size]).to(self.device) * sigma_f

        num_total_points = xdata.size(1)
        # Expand and take the difference
        xdata1 = torch.unsqueeze(xdata, dim=1)  # [B, 1, num_total_points, x_size]
        xdata2 = torch.unsqueeze(xdata, dim=2)  # [B, num_total_points, 1, x_size]

        diff = np.pi*torch.abs(xdata1 - xdata2)/p  # [B, num_total_points, num_total_points, x_size]

        # [B, y_size, num_total_points, num_total_points, x_size]
        norm = (diff[:, None, :, :, :] / l1[:, :, None, None, :]) ** 2
        norm = torch.sum(norm, dim=-1) # [B, y_size, num_total_points, num_total_points]

        # [B, y_size, num_total_points, num_total_points]
        kernel = (sigma_f ** 2)[:, :, None, None] * torch.exp(-2 * norm)

        # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise ** 2) * torch.eye(num_total_points).to(self.device)

        return kernel

    def _matern_kernel(self):
        # num_total_points = xdata.size(1)
        # # Expand and take the difference
        # xdata1 = torch.unsqueeze(xdata, dim=1)  # [B, 1, num_total_points, x_size]
        # xdata2 = torch.unsqueeze(xdata, dim=2)  # [B, num_total_points, 1, x_size]
        #
        # d = 4 * torch.abs(xdata1 - xdata2) # [B, num_total_points, num_total_points, x_size]
        # d = torch.sum(d, dim=-1).unsqueeze(dim=1)  # [B, y_size, num_total_points, num_total_points]
        # kernel = (1 + 4*5**0.5*d + 5.0/3.0*d**2) * torch.exp(-5**(0.5)*d)
        # # kernel += (sigma_noise ** 2) * torch.eye(num_total_points).to(self.device)
        import stheno.torch as stheno
        import stheno as sth
        kernel = stheno.Matern52().stretch(0.25)
        gp = stheno.GP(kernel, graph=sth.Graph())
        return gp

    def generate_with_Matern(self, gp):
        num_points = 200
        x_all = np.linspace(-2., 2., num_points)
        y_all = gp(x_all).sample()
        #     y_all = gp_.sample(x_all)
        x_all = torch.tensor(x_all, dtype=torch.float)[None, :, None]
        y_all = torch.tensor(y_all, dtype=torch.float).unsqueeze(0)
        self._batch_size = 1
        return x_all, y_all

    def generate_curves(self, include_context = True, x_min = -2, x_max = 2, sort=False):
        """Builds the op delivering the data.

        Generated functions are `float32` with x values between -2 and 2.
        Args:
            include_context:  Whether to include context data to target data, useful for NP, CNP, ANP
        Returns:
            A `NPRegressionDescription` namedtuple.
        """
        num_context = np.random.randint(3, self._max_num_context)

        # If we are testing we want to have more targets and have them evenly
        # distributed in order to plot the function.
        if self._testing:
            num_total_points = 400
            num_target = num_total_points - num_context
            x_values = torch.arange(x_min, x_max, 0.01)[None, :, None].repeat([self._batch_size, 1, 1]).to(self.device)
        # During training the number of target points and their x-positions are
        # selected at random
        else:
            num_target = np.random.randint(3, self._max_num_context)
            num_total_points = num_context + num_target
            # shape: (n_points, x_dim)
            x_values = np.random.rand(num_total_points, self._x_size)*(x_max - x_min) + x_min

        # shape: (n_points, batch_size)
        y_values = self.gp.sample_y(x_values, n_samples=self._batch_size)
        if sort == True:
            index_sorted = np.argsort(x_values)
            x_values = x_values[index_sorted]
            y_values = y_values[index_sorted]

        # convert to tensor and match expected shape
        x_values = torch.tensor(x_values, dtype=torch.float, device=self.device)
        ## rescale X following the website
        x_values = self.rescale_range(x_values)
        # shape: (bs, n_points, x_dim)
        x_values = x_values.unsqueeze(0).repeat(self._batch_size, 1, 1)
        # shape: (bs, n_points, x_dim)
        y_values = torch.tensor(y_values.T, dtype=torch.float, device=self.device).unsqueeze(dim=-1)

        # # scale
        # # scale = 2*np.random.rand() + 1 #scale by a factor from [1, 3)
        # # bias = 3*np.random.rand() #bias by [0,3)
        # scale = 1
        # bias = 0
        # y_values = y_values * scale + bias

        if self._testing:
            # Select the observations
            idx = np.random.permutation(num_total_points)
            context_x = x_values[:, idx[:num_context], :]
            context_y = y_values[:, idx[:num_context], :]
            target_x = x_values[:, idx[num_context:], :]
            target_y = y_values[:, idx[num_context:], :]
        else:

            if include_context:
                # Select the targets which constitute the context points as well as
                # some new target points
                target_x = x_values[:, :(num_target + num_context), :]
                target_y = y_values[:, :(num_target + num_context), :]
            else:
                target_x = x_values[:, num_context :(num_target + num_context), :]
                target_y = y_values[:, num_context :(num_target + num_context), :]
            # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
            query=query,
            y_target=target_y,
            num_total_points= num_target + num_context,
            num_context_points=num_context)

    def generate_temporal_curves(self, max_num_context = 10, include_context = True):
        """Builds the op delivering the data.

                Generated functions are `float32` with x values between -2 and 0 for context data, and 0 to 2 for target data.
                Args:
                    max_num_context:  we set max_num_context to be smaller(25) than generate curve for sequential data
                    include_context:  Whether to include context data to target data, useful for NP, CNP, ANP
                Returns:
                    A `NPRegressionDescription` namedtuple.
                """
        self._max_num_context = max_num_context
        num_context =  self._max_num_context # fixed sequence

        # the number of target points and their x-positions are
        # selected at random
        num_target = self._max_num_context #fixed sequence

        num_total_points = num_context + num_target
        x_context = -2 * torch.rand([self._batch_size, num_context, self._x_size]).to(self.device)
        x_target = 2 * torch.rand([self._batch_size, num_target, self._x_size]).to(self.device)
        x_values = torch.cat([x_context, x_target], dim=1)

        # Pass the x_values through the Gaussian kernel
        # [batch_size, y_size, num_total_points, num_total_points]
        if self.kernel == 'EQ':
            kernel = self._rbf_kernel(x_values)  # [B, y_size, num_total_points, num_total_points]
        elif self.kernel == 'period':
            kernel = self._periodic_kernel(x_values)

        # Calculate Cholesky, using double precision for better stability:
        cholesky = torch.cholesky(kernel.type(torch.float64)).type(
            torch.float32)  # [B, y_size, num_total_points, num_total_points]

        # Sample a curve
        # [batch_size, y_size, num_total_points, 1]
        y_values = torch.matmul(cholesky,
                                torch.rand([self._batch_size, self._y_size, num_total_points, 1]).to(self.device))

        # [batch_size, num_total_points, y_size]
        y_values = y_values.squeeze(-1).permute([0, 2, 1])

        if include_context:
            # Select the targets which constitute the context points as well as
            # some new target points
            target_x = x_values[:, :num_target + num_context, :]
            target_y = y_values[:, :num_target + num_context, :]
        else:
            target_x = x_values[:, num_context:num_target + num_context, :]
            target_y = y_values[:, num_context:num_target + num_context, :]
        # Select the observations
        context_x = x_values[:, :num_context, :]
        context_y = y_values[:, :num_context, :]

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
            query=query,
            y_target=target_y,
            num_total_points=num_target + num_context,
            num_context_points=num_context)

    def rescale_range(self, X, old_range=(-2, 2), new_range=(-1, 1)):
        """Rescale X linearly to be in `new_range` rather than `old_range`."""
        old_min = old_range[0]
        new_min = new_range[0]
        old_delta = old_range[1] - old_min
        new_delta = new_range[1] - new_min
        return (((X - old_min) * new_delta) / old_delta) + new_min

    def __len__(self):
        return self.n_samples

    def _sample_features(self, min_max, n_points, n_samples):
        """Sample X with non uniform intervals. """
        X = np.random.uniform(min_max[1], min_max[0], size=(n_samples, n_points))
        # sort which is convenient for plotting
        # X.sort(axis=-1)
        return X

    def _sample_targets(self, X):
        self.n_same_samples = 20
        targets = X.copy()
        n_samples, n_points = X.shape
        for i in range(0, n_samples, self.n_same_samples):


            for attempt in range(self.n_same_samples):
                # can have numerical issues => retry using a different X
                try:
                    # takes care of boundaries
                    n_same_samples = targets[i : i + self.n_same_samples, :].shape[0]
                    targets[i : i + self.n_same_samples, :] = self.gp.sample_y(
                        X[i + attempt, :, np.newaxis],
                        n_samples=n_same_samples,
                        random_state=None,
                    ).transpose(1, 0)
                    X[i : i + self.n_same_samples, :] = X[i + attempt, :]
                except np.linalg.LinAlgError:
                    continue  # try again
                else:
                    break  # success
            else:
                raise np.linalg.LinAlgError("SVD did not converge 10 times in a row.")

        # shuffle output to not have n_same_samples consecutive
        X, targets = sklearn.utils.shuffle(X, targets)
        targets = torch.from_numpy(targets)
        targets = targets.view(n_samples, n_points, 1).float()
        return X, targets

    def _postprocessing_features(self, X):
        """Convert the features to a tensor, rescale them to [-1,1] and expand."""
        self.min_max = (-2, 2)
        X = torch.from_numpy(X).unsqueeze(-1).float()
        X = rescale_range(X, self.min_max, (-1, 1))
        return X

    def precompute_xy(self):
        X = self._sample_features((-2, 2), self._max_num_context, self.n_samples)
        X, targets = self._sample_targets(X)
        X = self._postprocessing_features(X)
        self.x_values = X
        self.y_values = targets



    def __getitem__(self, index):
        # doesn't use index because randomly generated in any case => sample
        # in order which enables to know when epoch is finished and regenerate
        # new functions

        # shape: (n_points, x_dim)
        x_min = -2
        x_max = 2
        x_values = np.random.rand(self._max_num_context, self._x_size) * (x_max - x_min) + x_min
        x_values = x_values.astype('float32')

        # shape: (n_points, batch_size)
        y_values = self.gp.sample_y(x_values).astype('float32')
        if self.sort == True:
            index_sorted = np.argsort(x_values)
            x_values = x_values[index_sorted]
            y_values = y_values[index_sorted]

        x_values = self.rescale_range(x_values)
        # x_values = self.x_values[index]
        # y_values = self.y_values[index]
        return x_values, y_values

import numpy as np
import collections
import torch
import os
import pandas as pd
from torch.utils.data import Dataset



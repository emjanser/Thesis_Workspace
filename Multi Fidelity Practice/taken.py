import torch
import gpytorch

from gpytorch.kernels import Kernel
from gpytorch.lazy import DiagLazyTensor, InterpolatedLazyTensor, PsdSumLazyTensor, RootLazyTensor
from gpytorch.lazy import lazify
from gpytorch.priors import NormalPrior


class TwoFidelityIndexKernel(Kernel):
    """
    Separate kernel for each task based on the Hadamard Product between the task
    kernel and the data kernel. based on :
    https://github.com/cornellius-gp/gpytorch/blob/master/examples/03_Multitask_GP_Regression/Hadamard_Multitask_GP_Regression.ipynb
    
    The index identifier must start from 0, i.e. all task zero have index identifier 0 and so on.
    
    If noParams is set to `True` then the covar_factor doesn't include any parameters.
    This is needed to construct the 2nd matrix in the sum, as in (https://arxiv.org/pdf/1604.07484.pdf eq. 3.2) 
    where the kernel is treated as a sum of two kernels.
    
    k = [      k1, rho   * k1   + [0, 0
         rho * k1, rho^2 * k1]     0, k2]
    """
    def __init__(self,
                num_tasks,
                rank=1, # for two multifidelity always assumed to be 1
                prior=None,
                includeParams=True,
                **kwargs
                ):
        if rank > num_tasks:
            raise RuntimeError("Cannot create a task covariance matrix larger than the number of tasks")
        super().__init__(**kwargs)
        try:
            self.batch_shape
        except AttributeError as e:
            self.batch_shape = 1 #torch.Size([200])

        # we take a power of rho with the task index list (assuming all task 0 represented as 0, task 1 represented as 1 etc.)
        self.covar_factor = torch.arange(num_tasks).to(torch.float32)
        
        if includeParams:
            self.register_parameter(name="rho", parameter=torch.nn.Parameter(torch.randn(1)))
            print(f"Initial value : rho  {self.rho.item()}")
            self.covar_factor = torch.pow(self.rho.repeat(num_tasks), self.covar_factor)
            
        self.covar_factor = self.covar_factor.unsqueeze(0).unsqueeze(-1)
        self.covar_factor = self.covar_factor.repeat(self.batch_shape, 1, 1)

        if prior is not None and includeParams is True:
            self.register_prior("rho_prior", prior , self._rho)

    def _rho(self):
        return self.rho
    
    def _eval_covar_matrix(self):
        transp = self.covar_factor.transpose(-1, 0)
        ret = self.covar_factor.matmul(self.covar_factor.transpose(-1, -2)) #+ D
        return ret

    @property
    def covar_matrix(self):
        res = RootLazyTensor(self.covar_factor)
        return res

    def forward(self, i1, i2, **params):
        covar_matrix = self._eval_covar_matrix()
        res = InterpolatedLazyTensor(base_lazy_tensor=covar_matrix, left_interp_indices=i1, right_interp_indices=i2)
        return res

from pprint import pprint
from gpytorch.priors import NormalPrior


"""

and here's the code for the MultiTask Model
https://github.com/cornellius-gp/gpytorch/issues/594

"""

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)
        self.task_covar_module1 = TwoFidelityIndexKernel(num_tasks=2, rank=1)
        self.task_covar_module2 = TwoFidelityIndexKernel(num_tasks=2, rank=1, includeParams=False)
        print(f"Initial value : Covar 1, lengthscale {self.covar_module1.base_kernel.lengthscale.item()}, prefactor {self.covar_module1.outputscale.item()}")
        print(f"Initial value : Covar 2, lengthscale {self.covar_module2.base_kernel.lengthscale.item()}, prefactor {self.covar_module2.outputscale.item()}")
        
    def forward(self,x,i):
        mean_x = self.mean_module(x)
        
        # Get input-input covariance
        covar1_x = self.covar_module1(x)
        # Get task-task covariance
        covar1_i = self.task_covar_module1(i)
        
        # Get input-input covariance
        covar2_x = self.covar_module2(x)
        # Get task-task covariance
        covar2_i = self.task_covar_module2(i)
        
            
        # Multiply the two together to get the covariance we want
        covar1 = covar1_x.mul(covar1_i)
        covar2 = covar2_x.mul(covar2_i)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar1+covar2)
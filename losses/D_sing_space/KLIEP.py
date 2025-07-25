import torch
from torch import nn
from typing import Sequence, Union, List, Literal
from torch.nn import functional as F
from utils.registry import register_loss

@register_loss('DKLIEP')
class DKLIEP(nn.Module):
    """
    DKLIEP (KL Divergence-based Kernelized Importance Estimation Procedure) loss implementation.
    This loss is used for estimating good representations that span the D-singular space.
    Parameterizations:
    1. Naive version: K(x,a) = \Phi(x)^T \Psi(a) # This is the default option i.e., None
    2. Exponential parameterization: K(x,a) = \exp(\Phi(x)^T \Psi(a) / T)
    3. Squared exponential parameterization: K(x,a) = \exp(|| \Phi(x) - \Psi(a) ||^2 / T)   
    """
    def __init__(self, 
                 # kernel: nn.Module = None,
                 x_proj: nn.Module = None, 
                 a_proj: nn.Module = None,
                 exp_parameterization: Literal["inner_product", "squared"] = None,
                 temperature: float = 1.0,
                 normalization: float = 0.0,
                 ) -> None:
        """
        Initialize the DKLIEP loss module.
        Args:
        - kernel: a kernel module that computes the kernel between inputs and contexts.
        Args:
        - x_proj: a MLP module that further projects inputs x to embeddings. \Phi'(x) = x_proj(\Phi(x))
        - a_proj: a MLP module that further projects contexts a to embeddings. \Psi'(a) = a_proj(\Psi(a))
        - exp_paramerization:  whether to use exponential parameterization. 
        - temparature: float, temperature for exp_parameterization, default is 1.0.
        - temperature: float, temperature parameter for scaling the kernel values, default is 1.0.
        """
        super(DKLIEP, self).__init__()
        # self.kernel = kernel
        self.x_proj = x_proj
        self.a_proj = a_proj
        self.exp_parameterization = exp_parameterization
        self.temperature = temperature
        self.normalization = normalization

    def forward(self, x: torch.Tensor, a: torch.Tensor,
                reduction: Literal["mean", "none"] = "mean",
                ) -> torch.Tensor:
        """
        Inputs:
        - x: embedding of inputs x, torch tensor of shape (N, D)
        - a: embedding of contexts a, torch tensor of shape (N,D) or (N, r, D), representing single context or r contexts for each input 
        
        Outputs:
        - dkliep_loss: DKLIEP loss, torch tensor of shape (N,) or scalar if mean reduction is applied.
        - loss_dict: Dictionary containing verbose information like the loss of the positive pairs and negative pairs.
        """
        if self.x_proj is not None:
            x = self.x_proj(x)
        if self.a_proj is not None:
            if a.ndim == 2:
                a = self.a_proj(a)
            elif a.ndim == 3:
                N, r, D = a.shape
                a = self.a_proj(a.view(N*r, D)).view(N, r, -1)
        N, D = x.shape  # Batch size and embedding dimension

        # split a batch and corresponding augmentations into two halves
        x1, x2 = x.chunk(2, dim=0)
        a1, a2 = a.chunk(2, dim=0) # in both cases whether a is 2D or 3D
        # Check the dimensionality of a, and compute the kernel accordingly for both halves
        if a.ndim == 2:
            assert x.shape == a.shape, "Input tensors must have the same shape"
            
            # Dot products aka kernel evaluations between embeddings
            if self.exp_parameterization is None:
                k_xa_1 = x1 @ a1.T # (N, N)
                k_xa_2 = x2 @ a2.T # (N, N)
            elif self.exp_parameterization == "inner_product":
                k_xa_1 = torch.exp(x1 @ a1.T / self.temparature) # (N, N)
                k_xa_2 = torch.exp(x2 @ a2.T / self.temparature) # (N, N)
            elif self.exp_parameterization == "squared":
                # Z[i, j] = exp(|| X[i] - A[j] ||^2 / T)
                squared_diff_1 = ((x1[:, None, :] - a1[None, :, :]) ** 2).sum(dim=-1)
                squared_diff_2 = ((x2[:, None, :] - a2[None, :, :]) ** 2).sum(dim=-1)
                k_xa_1 = torch.exp(squared_diff_1 / self.temperature) # (N, N)
                k_xa_2 = torch.exp(squared_diff_2 / self.temperature) # (N, N)
            
            """
            # Compute the dot products for positive pairs
            sim_pos = dot_products.diag()  # Similarity of positive pairs, (N,)
            
            # Compute the dot products for negative pairs
            sim_neg = off_diagonal(dot_products)  # Similarity of negative pairs, (N, N-1)
            sim_neg_square = sim_neg.pow(2)  # Squared similarity of negative pairs, (N, N-1) 
            sim_neg_square = sim_neg_square.mean(dim=1)  # (N,)
            """
        elif a.ndim == 3:
            assert x.shape[0] == a.shape[0], "Batch size of x and a must match"
            assert x.shape[1] == a.shape[2], "Input tensors must have the same embedding dimension"
            N, r, D = a.shape
            
            # DKernel evaluations for all three parameterizations
            if self.exp_parameterization is None:
                # Z[i, j, k] = X[i]^T A[j, k]
                k_xa_1= torch.einsum('id,jkd->ijk', x1, a1)  # Shape: (N, N, r)
                k_xa_2= torch.einsum('id,jkd->ijk', x2, a2)  # Shape: (N, N, r)
            elif self.exp_parameterization == "inner_product":
                k_xa_1 = torch.einsum('id,jkd->ijk', x1, a1) / self.temparature  # Shape: (N, N, r)
                k_xa_1 = torch.exp(k_xa_1)
                k_xa_2 = torch.einsum('id,jkd->ijk', x2, a2) / self.temparature  # Shape: (N, N, r)
                k_xa_2 = torch.exp(k_xa_2)
            elif self.exp_parameterization == "squared":
                # Z[i, j, k] = exp(|| X[i] - A[j, k] ||^2 / T)
                squared_diff_1 = ((x1[:, None, None, :] - a1[None, :, :, :]) ** 2).sum(dim=-1)  # Shape: (N, N, r)
                k_xa_1 = torch.exp(squared_diff_1 / self.temperature) # (N, N, r)
                squared_diff_2 = ((x2[:, None, None, :] - a2[None, :, :, :]) ** 2).sum(dim=-1)  # Shape: (N, N, r)
                k_xa_2 = torch.exp(squared_diff_2 / self.temperature) # (N, N, r)

        # Now we compute the empirical loss according to the expression written in Overleaf
        # First term: -1/(Br) * sum_i sum_m log(K(x_i, a_{i,m}))
        # Extract diagonal elements K(x_i, a_{i,m}) for i=1..B, m=1..r
        diagonal_elements_1 = torch.diagonal(k_xa_1, dim1=0, dim2=1)  # Shape: (r, B)
        diagonal_elements_1 = diagonal_elements_1.T  # Shape: (B, r), where B is N.
        diagonal_elements_2 = torch.diagonal(k_xa_2, dim1=0, dim2=1)  # Shape: (r, B)
        diagonal_elements_2 = diagonal_elements_2.T  # Shape: (B, r)
    
        # Compute log and sum
        log_term = torch.log(diagonal_elements_1).sum()
        log_term_batch = torch.log(diagonal_elements_1).sum(dim=1)  # Shape: (B,)
        first_term = -log_term / (N * r)
        first_term_batch = -log_term_batch / r
        # Second term: λ * (1/(B²r) * sum_i sum_j sum_m K(x_i, a_{j,m}) - 1)²
        # Sum over all elements in the tensor
        total_sum = k_xa_2.sum()
        constraint_violation = total_sum / (N * N * r) - 1
        second_term = self.normalization * (constraint_violation ** 2)
    
    #return first_term + second_term
        # Normalization typically requires samples of the form (x_i, a_{j,m}) i.e., contexts sampled from different distributions.
        # Compute the loss based on the kernel values
        if reduction == "mean":
            dkliep_loss = first_term + second_term
        elif reduction == "none": # What do I return here?
            dkliep_loss = first_term_batch
        else:
            raise ValueError("Reduction must be 'mean' or 'none'.")
        
        # Prepare loss dictionary for verbose information
        loss_dict = {
            'dkliep_loss': (first_term + second_term).item(),
            'kernel_values': torch.mean(k_xa_1).item()
        }
        
        return dkliep_loss, loss_dict
                
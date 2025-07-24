import torch
from torch import nn
from typing import Sequence, Union, List, Literal
from torch.nn import functional as F
from utils.registry import register_loss

@register_loss('NTLogistic')
class NT_Logistic(nn.Module):
    """
    Info NWJ (Info Nguyen Weiss Jordan) loss implementation.
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
                 ) -> None:
        """
        Initialize the InfoNWJ loss module.
        Args:
        - kernel: a kernel module that computes the kernel between inputs and contexts.
        Args:
        - x_proj: a MLP module that further projects inputs x to embeddings. \Phi'(x) = x_proj(\Phi(x))
        - a_proj: a MLP module that further projects contexts a to embeddings. \Psi'(a) = a_proj(\Psi(a))
        - exp_paramerization:  whether to use exponential parameterization. 
        - temparature: float, temperature for exp_parameterization, default is 1.0.
        - temperature: float, temperature parameter for scaling the kernel values, default is 1.0.
        """
        super(NT_Logistic, self).__init__()
        # self.kernel = kernel
        self.x_proj = x_proj
        self.a_proj = a_proj
        self.exp_parameterization = exp_parameterization
        self.temperature = temperature
        # self.normalization = normalization

    def forward(self, x: torch.Tensor, a: torch.Tensor,
                reduction: Literal["mean", "none"] = "mean",
                ) -> torch.Tensor:
        """
        Inputs:
        - x: embedding of inputs x, torch tensor of shape (N, D)
        - a: embedding of contexts a, torch tensor of shape (N,D) or (N, r, D), representing single context or r contexts for each input 
        
        Outputs:
        - INWJ_loss: INWJ loss, torch tensor of shape (N,) or scalar if mean reduction is applied.
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


        if a.ndim == 2:
            assert x.shape == a.shape, "Input tensors must have the same shape"
            
            # Dot products between embeddings
            if self.exp_parameterization is None:
                k_xa = x @ a.T # (N, N)
            elif self.exp_parameterization == "inner_product":
                k_xa = torch.exp(x @ a.T / self.temparature) # (N, N)
            elif self.exp_parameterization == "squared":
                # Z[i, j] = exp(|| X[i] - A[j] ||^2 / T)
                squared_diff = ((x[:, None, :] - a[None, :, :]) ** 2).sum(dim=-1)
                k_xa = torch.exp(squared_diff / self.temperature) # (N, N)
            
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
                k_xa= torch.einsum('id,jkd->ijk', x, a)  # Shape: (N, N, r)
            elif self.exp_parameterization == "inner_product":
                k_xa = torch.einsum('id,jkd->ijk', x, a) / self.temparature  # Shape: (N, N, r)
                k_xa = torch.exp(k_xa)
            elif self.exp_parameterization == "squared":
                # Z[i, j, k] = exp(|| X[i] - A[j, k] ||^2 / T)
                squared_diff = ((x[:, None, None, :] - a[None, :, :, :]) ** 2).sum(dim=-1)  # Shape: (N, N, r)
                k_xa = torch.exp(squared_diff / self.temperature) # (N, N, r)

        # Now we compute the empirical loss according to the expression written in Overleaf
        # First term: 1/(Br) * sum_i sum_m [-log(σ(log(K(x_i, a_{i,m}))))]
        # Extract diagonal elements K(x_i, a_{i,m}) - shape: (B, r)
        diagonal_elements = torch.diagonal(k_xa, dim1=0, dim2=1).T  # Shape: (B, r)
    
        # Compute -log(σ(log(K))) = -log(sigmoid(log(K)))
        # Using log-sum-exp trick for numerical stability: -log(sigmoid(x)) = log(1 + exp(-x))
        log_diagonal = torch.log(diagonal_elements)
        first_term_sum = torch.log(1 + torch.exp(-log_diagonal)).sum() # because - log \sigma is just log (1/\sigma)
        first_term = first_term_sum / (N * r)
        first_term_row_average = torch.log(1 + torch.exp(-log_diagonal)).sum(dim=1)/r # Shape: (N,)
        # Second term: 1/(B²r) * sum_i sum_j sum_m [-log(σ(-log(K(x_i, a_{j,m}))))]
        # Compute -log(σ(-log(K))) = -log(sigmoid(-log(K)))
        # Using log-sum-exp trick: -log(sigmoid(x)) = log(1 + exp(-x))
        log_all_elements = torch.log(k_xa)
        neg_log_all_elements = -log_all_elements
        second_term_sum = torch.log(1 + torch.exp(-neg_log_all_elements)).sum()
        second_term_row_average = torch.log(1 + torch.exp(-neg_log_all_elements)).sum(dim=(1,2)) / (N * r) # Shape: (N,)
        second_term = second_term_sum / (N * N * r)
    
        # We just return the above INWJ loss which is the sum of first and second terms.
       
        if reduction == "mean":
            NTL_loss = first_term + second_term
        elif reduction == "none": # What do I return here?
            NTL_loss = first_term_row_average + second_term_row_average
        else:
            raise ValueError("Reduction must be 'mean' or 'none'.")
        
        # Prepare loss dictionary for verbose information
        loss_dict = {
            'NT_Logistic_loss': torch.mean(NTL_loss).item(),
            'mean of kernel_values': torch.mean(k_xa).item()
        }
        return NTL_loss, loss_dict
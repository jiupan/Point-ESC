import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinkhornKnopp(nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
    
    @torch.no_grad()
    def forward(self, scores):
        if scores.ndim not in (2, 3):
            raise ValueError(f"Unsupported scores shape: {scores.shape}")
        # Log-space evaluation of Q = exp(S / epsilon), without overflow.
        # Keep the paper's finite row-then-column normalization order.
        if scores.dtype in (torch.float16, torch.bfloat16):
            scores = scores.float()
        log_q = scores / self.epsilon
        for _ in range(self.num_iters):
            log_q = log_q - torch.logsumexp(log_q, dim=-1, keepdim=True)
            log_q = log_q - torch.logsumexp(log_q, dim=-2, keepdim=True)
        return log_q.exp()


class SimilarityComputer(nn.Module):
    def __init__(self, similarity_type='cosine', temperature=0.1):
        super().__init__()
        self.similarity_type = similarity_type
        self.temperature = temperature
    
    def forward(self, z1, z2):
        if self.similarity_type == 'cosine':
            return self._cosine_similarity(z1, z2)
        elif self.similarity_type == 'dot':
            return self._dot_similarity(z1, z2)
        elif self.similarity_type == 'euclidean':
            return self._euclidean_similarity(z1, z2)
        else:
            raise ValueError(f"Unsupported similarity type: {self.similarity_type}")
    
    def _cosine_similarity(self, z1, z2):
        z1_norm = F.normalize(z1, p=2, dim=-1)  # [B, N, D]
        z2_norm = F.normalize(z2, p=2, dim=-1)  # [B, N, D]
        
        similarity = torch.bmm(z1_norm, z2_norm.transpose(-2, -1))  # [B, N, N]
        
        return similarity / self.temperature
    
    def _dot_similarity(self, z1, z2):
        similarity = torch.bmm(z1, z2.transpose(-2, -1))
        return similarity / self.temperature
    
    def _euclidean_similarity(self, z1, z2):
        B, N, D = z1.shape
        
        z1_expanded = z1.unsqueeze(2)  # [B, N, 1, D]
        z2_expanded = z2.unsqueeze(1)  # [B, 1, N, D]
        
        distances = torch.norm(z1_expanded - z2_expanded, p=2, dim=-1)  # [B, N, N]
        
        similarity = -distances / self.temperature
        
        return similarity


class ReSALoss(nn.Module):
    def __init__(self, 
                 num_clusters=64,
                 similarity_type='cosine',
                 temperature=0.1,
                 sinkhorn_iters=3,
                 sinkhorn_epsilon=0.05,
                 distance_type='kl'):
        super().__init__()
        
        self.num_clusters = num_clusters
        self.similarity_computer = SimilarityComputer(similarity_type, temperature)
        self.sinkhorn = SinkhornKnopp(sinkhorn_iters, sinkhorn_epsilon)
        self.distance_type = distance_type
    
    def forward(self, h1, h2, z1, z2):
        # H generates Q; Z predicts correspondence and retains its encoder graph.
        with torch.no_grad():
            encoder_affinity = torch.bmm(h1, h2.transpose(-2, -1))
            assignment = self.sinkhorn(encoder_affinity)
        similarity = self.similarity_computer(z1, z2)  # [B, N, N]
        
        if self.distance_type == 'kl':
            loss = self._kl_divergence_loss(assignment, similarity)
        elif self.distance_type == 'cosine':
            loss = self._cosine_distance_loss(assignment, z1, z2)
        elif self.distance_type == 'cross_entropy':
            loss = self._cross_entropy_loss(assignment, similarity)
        else:
            raise ValueError(f"Unsupported distance type: {self.distance_type}")
        
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite ESC loss")
        
        return loss
    
    def _kl_divergence_loss(self, assignment, similarity):
        log_similarity = F.log_softmax(similarity, dim=-1)
        loss = F.kl_div(log_similarity, assignment, reduction='none').mean()
        return loss
    
    def _cosine_distance_loss(self, assignment, z1, z2):
        weighted_similarity = torch.sum(assignment * self.similarity_computer(z1, z2), dim=-1)
        loss = -torch.mean(weighted_similarity)
        return loss
    
    def _cross_entropy_loss(self, assignment, similarity):
        targets = torch.argmax(assignment, dim=-1)  # [B, N]
        
        loss = F.cross_entropy(similarity.view(-1, similarity.size(-1)), 
                              targets.view(-1), 
                              reduction='mean')
        return loss


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=256, num_layers=2):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.projection = nn.Sequential(*layers)
    
    def forward(self, x):
        original_shape = x.shape
        
        if len(original_shape) == 3:
            B, N, D = original_shape
            x = x.view(B * N, D)
            projected = self.projection(x)
            projected = projected.view(B, N, -1)
        else:
            projected = self.projection(x)
        
        return projected


def compute_assignment_accuracy(assignment1, assignment2, threshold=0.5):
    hard_assign1 = (assignment1 > threshold).float()
    hard_assign2 = (assignment2 > threshold).float()
    
    agreement = (hard_assign1 == hard_assign2).float().mean()
    
    return agreement


def visualize_assignment_matrix(assignment, save_path=None):
    import matplotlib.pyplot as plt
    
    if len(assignment.shape) == 3:
        assignment = assignment[0]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(assignment.cpu().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Assignment Matrix')
    plt.xlabel('Patches (View 2)')
    plt.ylabel('Patches (View 1)')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class result_logger():
    def __init__(self, root):
        self.root = root

    def write_content(self, *arg):
        print(*arg)
        f = open(os.path.join(self.root, 'log.txt'), 'a')
        print(*arg, file=f)
        f.close()

def sum_reverse(Q, num):
    """
    Calculates the sum of integers from Q down to num+1.
    Used for calculating offsets in universal task indexing.
    
    Args:
        Q (int): Total number of queries/tasks.
        num (int): Lower bound (exclusive).
        
    Returns:
        int: Sum of the sequence.
    """
    return np.sum([n for n in range(Q, num, -1)])


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron (Feed-Forward Network).
    Structure: Linear -> ReLU -> ... -> Linear
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        
        # Create list of layers: [Input -> Hidden, Hidden -> Hidden, ..., Hidden -> Output]
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # Apply ReLU for all layers except the last one
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def task_onehot2task_index_universal_v2(onehot):
    """
    Converts a batch of task one-hot encodings into task indices.
    This is used to map specific task combinations to a unique index
    for retrieving task-specific tokens or parameters.

    Args:
        onehot (Tensor): Task codes [B, C, Q].
                         B: Batch size
                         C: Number of channels/tasks
                         Q: Total query number

    Returns:
        Tensor: Task indices [B, C]
    """
    B, C, Q = onehot.shape
    
    # Calculate the number of active tasks per channel (sum along Q dim)
    # s_t shape: [B, C]
    s_t = torch.sum(onehot, dim=2).cpu().numpy()

    # Find the index of the first active task (where value is 1)
    # This replaces the triple nested loop for finding 'q'
    # argmax returns the first index of the maximum value (1)
    sep_index = torch.argmax(onehot, dim=2).cpu().numpy() # Shape: [B, C]

    # Calculate the global task index
    # Logic: Based on a triangular indexing scheme (sum_reverse calculates the offset)
    task_index = []
    for b in range(B):
        batch_indices = []
        for c in range(C):
            # Calculate offset based on how many tasks are active (s_t)
            # sum_reverse calculates the cumulative sum of previous 'levels'
            level_offset = sum_reverse(Q, int(Q - (s_t[b, c] - 1)))
            
            # Final index = Level Offset + Position Index
            current_idx = level_offset + sep_index[b, c]
            batch_indices.append(current_idx)
        task_index.append(np.stack(batch_indices, axis=0))

    # Convert back to tensor on the correct device
    task_index = np.array(task_index).astype(np.int64) # Use int64 for indices
    return torch.from_numpy(task_index).to(onehot.device)
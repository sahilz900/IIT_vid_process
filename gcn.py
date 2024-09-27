import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch_geometric
import numpy as np

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        return x

def create_edges(num_nodes):
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]], dtype=np.int64)
    return torch.tensor(edges.T, dtype=torch.long)

def refine_landmarks(landmarks):
    input_dim = 2
    hidden_dim = 16
    output_dim = 2

    model = GCN(input_dim, hidden_dim, output_dim)
    landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)
    edge_index = create_edges(landmarks_tensor.shape[0])
    refined_landmarks = model(landmarks_tensor, edge_index)

    return refined_landmarks.detach().numpy()

if __name__ == "__main__":
    sample_landmarks = np.array([
        [30, 40],
        [70, 40],
        [50, 60],
        [40, 80],
        [60, 80]
    ], dtype=np.float32)
    refined = refine_landmarks(sample_landmarks)

    print("Original Landmarks:", sample_landmarks)
    print("Refined Landmarks:", refined)

import torch_geometric
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
import scipy
from scipy import io
import numpy as np
class QM7(InMemoryDataset) :

    url = 'http://www.quantum-machine.org/data/qm7.mat'

    def __init__(self, root, transform=None, pre_transform = None, pre_filter=None) :
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self) :
        return 'qm7.mat'
    
    @property
    def processed_file_names(self) :
        return 'data.pt'
    
    def download(self) :
        download_url(self.url, self.raw_dir)
    
    def process(self) :
        data = scipy.io.loadmat(self.raw_paths[0])
        X = torch.from_numpy(data['X'])
        num_samples = X.shape[0]
        labels = np.transpose(data['T']).reshape(num_samples, -1)
        label_scaled_factor = np.max(np.abs(labels))
        labels = torch.from_numpy(labels/label_scaled_factor).to(torch.float)
        data_list = []
        for i in range(X.shape[0]) :
            coulomb_matrix = X[i] # 23 x 23
            pre_edge_index = coulomb_matrix.nonzero()
            edge_index = coulomb_matrix.nonzero().t().contiguous() # num_edges, 2 => needed to transform to 2,num_edges
            edge_attr = coulomb_matrix[pre_edge_index[0], pre_edge_index[1]]
            edge_attr = edge_attr.reshape(-1, 1) # num_edges, num_edge_features
            y = labels[i].view(1,-1)  # (1, ) for graph level prediction
            diag = torch.diagonal(coulomb_matrix, offset=0) # > 0 : half above diagonal < 0 : half under diagonal
            x = diag[diag.nonzero()]
            x = x.reshape(x.shape[0], -1)
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            
            graph.num_nodes = x.shape[0]
            graph.num_edge_features = 1
            graph.num_node_features = 1
            data_list.append(graph)
        
        if self.pre_filter is not None :
            data_list = [d for d in data_list if self.pre_filter[d]]
        
        if self.pre_transform is not None :
            data_list = [self.pre_transform(d) for d in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data,slices), self.processed_paths[0])

dataset = QM7(root='qm7')
print("Num graphs", len(dataset))
print("Num features", (dataset[0].num_features))
#print("Num classes", dataset.num_classes)










    




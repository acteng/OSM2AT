# import torch
# import torch.nn as nn
# import torch.optim as optim
#from torch_geometric.nn import GCNConv
# import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import tqdm

# class GCN(torch.nn.Module):
#     def __init__(self, input_feats, hidden1, num_classes):
#         super().__init__()
#         self.conv1 = GCNConv(input_feats, hidden1)
#         self.conv2 = GCNConv(hidden1, num_classes)

#     def forward(self, x, edge_index, edge_values = None,with_vals=False):
#         if with_vals:
#             x = self.conv1(x, edge_index, edge_values)
#         else:
#             x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         output = self.conv2(x, edge_index)

#         return output
    

# class Multiclass(nn.Module):
#     def __init__(self, input_size, hidden_size1, output_size):
#         super().__init__()
#         self.hidden = nn.Linear(input_size, hidden_size1)
#         self.act = nn.ReLU()
#         self.output = nn.Linear(hidden_size1, output_size)
        
#     def forward(self, x):
#         x = self.act(self.hidden(x))
#         x = self.output(x)
#         return x
    
    
#%%

def get_impute_masks(tag_to_impute,edge_attributes):

    edges_with_values = list(edge_attributes[edge_attributes[tag_to_impute].notna()]['edge_index'])
    
    # Split into test/train sets
    var_exists = [False] * edge_attributes.shape[0]
    var_to_impute = [True] * edge_attributes.shape[0]
    
    for i in edges_with_values:
        var_exists[i] = True
        var_to_impute[i] = False

    return var_exists, var_to_impute

#%%
def knn_dist_impute(edge_attributes,var_exists,var_to_impute,tag_to_impute):

    training_set = edge_attributes[['cent_x','cent_y']].values[var_exists]
    impute_set = edge_attributes[['cent_x','cent_y']].values[var_to_impute]
    target = edge_attributes[tag_to_impute].values[var_exists]
    
    k = 1
    
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(training_set,target)
    
    imputed_vars = neigh.predict(impute_set)
    
    return imputed_vars

#%%

def feature_learning_train_sets(edge_attributes, tag_to_impute, tags_to_add):
    tag_labels = list(edge_attributes[tag_to_impute].value_counts().index)
    tag_labels.sort()
    
    target_to_num = {}
    num_to_target = {}
    
    counter = 0
    
    for i in tag_labels:
        target_to_num[i] = counter
        num_to_target[counter] = i
        counter += 1
    
    #Train and impute sets with features
    
    # Generate Feature Vectors
    feat_list = list(set(edge_attributes.columns) & set(tags_to_add))
    feat_list.remove(tag_to_impute)
    
    #Generate Target Vectors
    target = edge_attributes[tag_to_impute]
    
    #Target at ints
    y_int = target.map(target_to_num).values
    
    #Target as one hot encoded
    y_onehot = pd.get_dummies(target).values
    
    # Create training set using one hot encoding
    features = edge_attributes[feat_list]
    one_hot_feats = pd.get_dummies(features.fillna('No tag'))
    x_hot = one_hot_feats.values
    
    return target_to_num, num_to_target, target, y_int, y_onehot, x_hot

#%%

def mode_rule(edge_attributes,var_exists,var_to_impute,tag_to_impute):
    mode_rule_dict = {}
    mode_ts = edge_attributes[var_exists]

    for t in list(edge_attributes[var_to_impute]['highway'].value_counts().index):
        if mode_ts[mode_ts['highway'] == t][tag_to_impute].shape[0] > 0:
            mode_rule_dict[t] = mode_ts[mode_ts['highway'] == t][tag_to_impute].mode()[0]
        else:
            mode_rule_dict[t] = mode_ts[tag_to_impute].mode()[0]
    
    return edge_attributes[var_to_impute]['highway'].map(mode_rule_dict).values
    
#%%

def knn_feats(x_hot,var_exists,target,var_to_impute,k = 3):
   
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x_hot[var_exists], target[var_exists])
    predicted = neigh.predict(x_hot[var_to_impute])
    return predicted

#%%

# def mlp_impute(y_onehot,x_hot,hidden_layer1,var_exists,var_to_impute,batch_size,n_epochs,num_to_target):

#     num_classes = y_onehot.shape[1]
#     num_features = x_hot.shape[1]
    
#     model = Multiclass(num_features, hidden_layer1, num_classes)
#     print(model)
    
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    
#     x_train_t = torch.tensor(x_hot[var_exists].astype(np.float16), dtype=torch.float32)
#     x_test_t = torch.tensor(x_hot[var_to_impute].astype(np.float16), dtype=torch.float32)
#     y_train_t = torch.tensor(y_onehot.astype(np.float16), dtype=torch.float32)
    
#     batches_per_epoch = len(x_train_t) // batch_size
    
#     losses = []
#     for epoch in range(n_epochs):
#         with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
#             bar.set_description(f"Epoch {epoch}")
#             for i in bar:
#                 # take a batch
#                 start = i * batch_size
#                 X_batch = x_train_t[start:start+batch_size]
#                 y_batch = y_train_t[start:start+batch_size]
#                 # forward pass
#                 y_pred = model(X_batch)
#                 loss = criterion(y_pred, y_batch)
#                 losses.append(loss.detach().numpy())
#                 # backward pass
#                 optimizer.zero_grad()
#                 loss.backward()
#                 # update weights
#                 optimizer.step()
    
#     y_pred = model(x_test_t)
    
#     predicted = []
#     for i in torch.argmax(y_pred, 1).cpu().detach().numpy():
#         predicted.append(num_to_target[i])
    
#     return np.array(predicted)


#%%

def get_adj_mx(edge_attributes,G):

    #Empty numpy matrix E*E
    adj_max_1 = np.zeros((len(edge_attributes),len(edge_attributes)))
    
    node_dict = dict.fromkeys(list(G.nodes()), [])
    
    for node in G.nodes():
    
        outgoing_edges = G.out_edges(node, data=True)  # Include edge data
        
        for (i, j, edge_data) in outgoing_edges:
            
            try:
                node_dict[i] = node_dict[i] + [edge_data['edge index']]
                node_dict[j] = node_dict[j] + [edge_data['edge index']]
            except:
                pass
    
    #Iterate through edges, get edges from node on either end
    for i,r in edge_attributes.iterrows():
        edges_i = node_dict[i[0]]
        edges_j = node_dict[i[1]]
        
        index_of_edges = edges_i + edges_j
        index_of_edges.append(r['edge_index'])
        adj_max_1[r['edge_index'],index_of_edges] = 1
    
    edge_index_1 = []
    
    for i in range(len(adj_max_1)):
        for j in range(len(adj_max_1)):
            if adj_max_1[i,j] > 0:
                edge_index_1.append([i,j])
    
    edge_index_1 = np.array(edge_index_1).T

    return edge_index_1

#%%

# def gnn_impute(x_hot,y_onehot,edge_index,device,hidden_layer1,n_epochs,var_exists,var_to_impute,num_to_target):

#     x_t = torch.tensor(x_hot.astype(np.float16), dtype=torch.float32)
#     y_t = torch.tensor(y_onehot.astype(np.float16), dtype=torch.float32)
    
#     edge_index_t = torch.tensor(edge_index).to(device).long()
    
#     num_classes = y_onehot.shape[1]
#     num_features = x_hot.shape[1]
    
#     gcn = GCN(num_features,hidden_layer1,num_classes).to(device)
#     optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
#     criterion = nn.CrossEntropyLoss()
    
#     gcn.train()
    
#     losses = []
    
#     for epoch in range(1, n_epochs + 1):
        
#         if epoch % 25 == 0:
#             print(epoch)
        
#         optimizer_gcn.zero_grad()
#         out = gcn(x_t,edge_index_t,None,with_vals=False)
#         loss = criterion(out[var_exists], y_t[var_exists])
#         losses.append(loss.detach().numpy())
#         loss.backward()
#         optimizer_gcn.step()    
        
#     out = gcn(x_t,edge_index_t,None,with_vals=False)
    
#     predicted = []
#     for i in out.argmax(dim=1).detach().numpy()[var_to_impute]:
#         predicted.append(num_to_target[i])
    
#     return np.array(predicted)


#%%

def ottawa_impute_speed(edge_attributes,var_to_impute):
    imputed_vals = []
    for i in list(edge_attributes[var_to_impute].highway):
        if i == 'motorway':
            imputed_vals.append(60)
        elif i == 'primary':
            imputed_vals.append(50)
        else:
            imputed_vals.append(30)
            
    return imputed_vals

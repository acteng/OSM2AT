from .Functions.imputation_missing_data import knn_dist_impute,get_impute_masks, mode_rule,ottawa_impute_speed
from .Functions.LTS import lts_ottawa
#from .Functions.self_learning import self_learn
from .Functions.helper_functions import dedupe_var_replace
import osmnx as ox
import csv
import shapely
import numpy as np
import pandas as pd
import os

class cyclist:
    def __init__(self, description, cycle_speed, risk_weights, risk_allowance, risk_decay):
        self.description = description
        self.cycle_speed = cycle_speed
        self.risk_weights = risk_weights
        self.risk_allowance = risk_allowance
        self.risk_decay = risk_decay
    def return_beta_linear(self,edge):
        beta = (edge[0] * self.risk_weights[0]) + (edge[1] * self.risk_weights[1]) + (edge[2] * self.risk_weights[2]) + (edge[3] * self.risk_weights[3]) + (edge[4] * self.risk_weights[4])
        return beta

#Import additional OSM tags for pulling data from OSMNX

# Get the path to the data directory within the package
tag_file = os.path.join(os.path.dirname(__file__), 'tags.txt')
with open (tag_file, 'r') as f:
    tags_to_add = [row[0] for row in csv.reader(f,delimiter=',')]

utw = ox.settings.useful_tags_way + tags_to_add
ox.config(use_cache=True, log_console=True, useful_tags_way=utw)

def get_cycle_network(bounding_box,impute_method,lts_method,pull_method,place):
    
    #Define weight matrices for different users
    weights_beginner = {0:0.1,1:0.2,2:2,3:4,4:10}
    weights_eager = {0:0.1,1:0.2,2:1.2,3:2,4:5}
    weights_experienced = {0:0.1,1:0.1,2:0.25,3:1,4:1.5}

    cyclist_types = {
        'beginner':cyclist(description = 'Beginner', cycle_speed=4.5, risk_weights=weights_beginner,risk_allowance = 3, risk_decay = 2),
        'eager':cyclist(description = 'Eager', cycle_speed=5.5, risk_weights=weights_eager,risk_allowance = 2, risk_decay = 2),
        'experienced':cyclist(description = 'Experienced', cycle_speed=6, risk_weights=weights_experienced,risk_allowance = 1.2, risk_decay = 2)
    }
    
    #Get data from OSMNX
    if pull_method == 'bb':
        G = ox.graph_from_bbox(bounding_box[3],bounding_box[1], bounding_box[0], bounding_box[2],network_type = 'bike',retain_all=True,simplify=False)
    elif pull_method == 'place':
        G = ox.graph.graph_from_place(place,network_type = 'bike', retain_all=True, simplify=False)
    
    #Get edge attributes
    edge_attributes = ox.graph_to_gdfs(G, nodes=True)[1]
    #Get edge centroids
    edge_attributes['cent_x'] = edge_attributes['geometry'].centroid.x
    edge_attributes['cent_y'] = edge_attributes['geometry'].centroid.y
    #Add edge index
    edge_attributes['edge_index'] = range(len(edge_attributes))
    
    #Impute Max Speed
    print('Imputing Max Speed')
    tag_to_impute = 'maxspeed'
    #Imputation masks
    #Manual work around to replace all instances of 'signal'
    #ToDo : more pythonic way to do this e.g., just search for strings which have a number in them.
    edge_attributes['maxspeed'] = edge_attributes['maxspeed'].replace('signals',np.nan)
    edge_attributes['maxspeed'] = edge_attributes['maxspeed'].replace('none',np.nan)
    var_exists, var_to_impute = get_impute_masks(tag_to_impute,edge_attributes)
    #Get ML training sets
    #target_to_num, num_to_target, target, y_int, y_onehot, x_hot = feature_learning_train_sets(edge_attributes, tag_to_impute, tags_to_add)
    #Impute missing data
    if impute_method == 'knn-dist':
        print('Imputing data using method - KNN Dist')
        imputed_vals = knn_dist_impute(edge_attributes,var_exists,var_to_impute,tag_to_impute)
    # elif impute_method == 'knn-feats':
    #     print('Imputing data using method - KNN Feats')
    #     #todo: default value for k
    #     imputed_vals = knn_feats(x_hot,var_exists,target,var_to_impute,k = 3)
    elif impute_method == 'mode-rule':
        print('Imputing data using method - Mode Rule')
        imputed_vals = mode_rule(edge_attributes,var_exists,var_to_impute,tag_to_impute)
        print('Data imputed')
    # elif impute_method == 'mlp':
    #     print('Imputing data using method - MLP')
    #     imputed_vals = mlp_impute(y_onehot,x_hot,mlp_train_params['hidden_layer'],var_exists,var_to_impute,mlp_train_params['batch_size'],mlp_train_params['n_epochs'],num_to_target)
    elif impute_method == 'ottawa':
        print('Imputing data using method - MLP')
        print('WARNING : This method has hardcoded values specific to a UK setting.')
        imputed_vals = ottawa_impute_speed(edge_attributes,var_to_impute)
    
    #Add imputed values to edge_attributes
    edge_attributes.loc[var_to_impute,tag_to_impute] = imputed_vals
    speed_num = []
    for i in list(edge_attributes[tag_to_impute].values):
        if type(i) != int:
            speed_num.append(int("".join(filter(str.isdigit, i))))
        else:
            speed_num.append(i)
    edge_attributes[tag_to_impute] = speed_num

    #Replace dupes on osmid with mode
    edge_attributes = dedupe_var_replace(edge_attributes,tag_to_impute)

    #Impute Lanes
    print('Imputing Number of Lanes')

    if 'lanes' in edge_attributes.columns:
        edge_attributes['lanes'] = edge_attributes['lanes'].replace('3;4;4',np.nan)
        edge_attributes['lanes'] = edge_attributes['lanes'].replace('1; 2',np.nan)
        tag_to_impute = 'lanes'
        #Imputation masks
        var_exists, var_to_impute = get_impute_masks(tag_to_impute,edge_attributes)
        #Get ML training sets
        #target_to_num, num_to_target, target, y_int, y_onehot, x_hot = feature_learning_train_sets(edge_attributes, tag_to_impute, tags_to_add)
        #Impute missing data
        if impute_method == 'knn-dist':
            print('Imputing data using method - KNN Dist')
            imputed_vals = knn_dist_impute(edge_attributes,var_exists,var_to_impute,tag_to_impute)
        # elif impute_method == 'knn-feats':
        #     print('Imputing data using method - KNN Feats')
        #     #todo: default value for k
        #     imputed_vals = knn_feats(x_hot,var_exists,target,var_to_impute,k = 3)
        elif impute_method == 'mode-rule':
            print('Imputing data using method - Mode Rule')
            imputed_vals = mode_rule(edge_attributes,var_exists,var_to_impute,tag_to_impute)
            print('Data imputed')
        # elif impute_method == 'mlp':
        #     print('Imputing data using method - MLP')
        #     imputed_vals = mlp_impute(y_onehot,x_hot,mlp_train_params['hidden_layer'],var_exists,var_to_impute,mlp_train_params['batch_size'],mlp_train_params['n_epochs'],num_to_target)
        
        if impute_method == 'ottawa':
            edge_attributes.loc[var_to_impute,tag_to_impute] = 2
            edge_attributes[tag_to_impute] = edge_attributes[tag_to_impute].astype(float)
        else:
            edge_attributes.loc[var_to_impute,tag_to_impute] = imputed_vals
            edge_attributes[tag_to_impute] = edge_attributes[tag_to_impute].astype(float)

        edge_attributes = dedupe_var_replace(edge_attributes,tag_to_impute)
        
    else:
        edge_attributes['lanes'] = 2

    #Impute Surface
    print('Imputing Surface')
    tag_to_impute = 'surface'
    print(edge_attributes.columns)
    if tag_to_impute in edge_attributes.columns:
        #Imputation masks
        var_exists, var_to_impute = get_impute_masks(tag_to_impute,edge_attributes)
        #Get ML training sets
        #target_to_num, num_to_target, target, y_int, y_onehot, x_hot = feature_learning_train_sets(edge_attributes, tag_to_impute, tags_to_add)

        #Impute missing data
        if impute_method == 'knn-dist':
            print('Imputing data using method - KNN Dist')
            imputed_vals = knn_dist_impute(edge_attributes,var_exists,var_to_impute,tag_to_impute)
        # elif impute_method == 'knn-feats':
        #     print('Imputing data using method - KNN Feats')
        #     #todo: default value for k
        #     imputed_vals = knn_feats(x_hot,var_exists,target,var_to_impute,k = 3)
        elif impute_method == 'mode-rule':
            print('Imputing data using method - mode rule')
            imputed_vals = mode_rule(edge_attributes,var_exists,var_to_impute,tag_to_impute)
            print('Data imputed')
        # elif impute_method == 'mlp':
        #     print('Imputing data using method - MLP')
        #     imputed_vals = mlp_impute(y_onehot,x_hot,mlp_train_params['hidden_layer'],var_exists,var_to_impute,mlp_train_params['batch_size'],mlp_train_params['n_epochs'],num_to_target)

        if impute_method != 'ottawa':
            edge_attributes.loc[var_to_impute,tag_to_impute] = imputed_vals
            edge_attributes = dedupe_var_replace(edge_attributes,tag_to_impute)
    else:
        edge_attributes[tag_to_impute] = None
    
    #Add in Access and Footway tags if missing for LTS classification
    if 'access' not in edge_attributes.columns:
        edge_attributes['access'] = 'NAN'
    if 'footway' not in edge_attributes.columns:
        edge_attributes['footway'] = 'NAN'
    if 'bicycle' not in edge_attributes.columns:
        edge_attributes['bicycle'] = 'NAN'
    if 'motor_vehicle' not in edge_attributes.columns:
        edge_attributes['motor_vehicle'] = 'NAN'
                    
    #Compute Edge-Level LTS 
    if lts_method == 'ottawa':
        print('Calculating LTS using Ottawa Advocacy Group method')
        lts = lts_ottawa(edge_attributes)
        edge_attributes['LTS'] = lts['LTS_ottawa']
        edge_attributes = pd.concat([edge_attributes, pd.get_dummies(lts['LTS_ottawa'])], axis=1)
    # elif lts_method == 'self-learn':
    #     print('Calculating LTS using Self-Learning Approach')
    #     print('WARNING : this approach is under development, please check your results carefully')
    #     lts = self_learn(edge_attributes,self_learn_k)
    #     edge_attributes['LTS'] = lts['cluster']
    #     edge_attributes = pd.concat([edge_attributes, pd.get_dummies(lts['cluster'])], axis=1)
    
    for col in [0,1,2,3,4]:
        if col not in edge_attributes.columns:
            edge_attributes[col] = 0
    
    #Compute edge-level access cost for each user type
    risk_vectors = np.zeros((len(edge_attributes),3))
    it = 0
    for i,r in edge_attributes.iterrows():
        risk_vectors[it,0] = cyclist_types['beginner'].return_beta_linear(r)
        risk_vectors[it,1] = cyclist_types['eager'].return_beta_linear(r)
        risk_vectors[it,2] = cyclist_types['experienced'].return_beta_linear(r)
        it += 1
        
    normalized_risk_vectors = 1 + (risk_vectors - risk_vectors.min()) / (risk_vectors.max() - risk_vectors.min())

    it = 0
    for i,r in edge_attributes.iterrows():
        G[i[0]][i[1]][i[2]]['ac_beginner'] = normalized_risk_vectors[it,0] * r['length']
        G[i[0]][i[1]][i[2]]['ac_eager'] = normalized_risk_vectors[it,1] * r['length']
        G[i[0]][i[1]][i[2]]['ac_expert'] = normalized_risk_vectors[it,2] * r['length']
        G[i[0]][i[1]][i[2]]['time_beginner'] = r['length']/cyclist_types['beginner'].cycle_speed
        G[i[0]][i[1]][i[2]]['time_eager'] = r['length']/cyclist_types['eager'].cycle_speed
        G[i[0]][i[1]][i[2]]['time_expert'] = r['length']/cyclist_types['experienced'].cycle_speed
        G[i[0]][i[1]][i[2]]['LTS'] = r['LTS']
        
    return G, edge_attributes


def measure_LTS_from_network(G,impute_method,lts_method):
    
    #Get edge attributes
    edge_attributes = ox.graph_to_gdfs(G, nodes=True)[1]
    
    #Get edge centroids
    edge_attributes['cent_x'] = edge_attributes['geometry'].centroid.x
    edge_attributes['cent_y'] = edge_attributes['geometry'].centroid.y
    #Add edge index
    edge_attributes['edge_index'] = range(len(edge_attributes))
    
    #Impute Max Speed
    print('Imputing Max Speed')
    tag_to_impute = 'maxspeed'
    #Imputation masks
    #Manual work around to replace all instances of 'signal'
    #ToDo : more pythonic way to do this e.g., just search for strings which have a number in them.
    edge_attributes['maxspeed'] = edge_attributes['maxspeed'].replace('signals',np.nan)
    edge_attributes['maxspeed'] = edge_attributes['maxspeed'].replace('none',np.nan)
    var_exists, var_to_impute = get_impute_masks(tag_to_impute,edge_attributes)
    #Get ML training sets
    # target_to_num, num_to_target, target, y_int, y_onehot, x_hot = feature_learning_train_sets(edge_attributes, tag_to_impute, tags_to_add)
    #Impute missing data
    if impute_method == 'knn-dist':
        print('Imputing data using method - KNN Dist')
        imputed_vals = knn_dist_impute(edge_attributes,var_exists,var_to_impute,tag_to_impute)
    # elif impute_method == 'knn-feats':
    #     print('Imputing data using method - KNN Feats')
    #     #todo: default value for k
    #     imputed_vals = knn_feats(x_hot,var_exists,target,var_to_impute,k = 3)
    elif impute_method == 'mode-rule':
        print('Imputing data using method - Mode Rule')
        imputed_vals = mode_rule(edge_attributes,var_exists,var_to_impute,tag_to_impute)
        print('Data imputed')
    # elif impute_method == 'mlp':
    #     print('Imputing data using method - MLP')
    #     imputed_vals = mlp_impute(y_onehot,x_hot,mlp_train_params['hidden_layer'],var_exists,var_to_impute,mlp_train_params['batch_size'],mlp_train_params['n_epochs'],num_to_target)
    elif impute_method == 'ottawa':
        print('Imputing data using method - MLP')
        print('WARNING : This method has hardcoded values specific to a UK setting.')
        imputed_vals = ottawa_impute_speed(edge_attributes,var_to_impute)
    
    #Add imputed values to edge_attributes
    edge_attributes.loc[var_to_impute,tag_to_impute] = imputed_vals
    speed_num = []
    for i in list(edge_attributes[tag_to_impute].values):
        if type(i) != int:
            speed_num.append(int("".join(filter(str.isdigit, i))))
        else:
            speed_num.append(i)
    edge_attributes[tag_to_impute] = speed_num

    #Replace dupes on osmid with mode
    edge_attributes = dedupe_var_replace(edge_attributes,tag_to_impute)

    #Impute Lanes
    print('Imputing Number of Lanes')

    if 'lanes' in edge_attributes.columns:
        edge_attributes['lanes'] = edge_attributes['lanes'].replace('3;4;4',np.nan)
        edge_attributes['lanes'] = edge_attributes['lanes'].replace('1; 2',np.nan)
        tag_to_impute = 'lanes'
        #Imputation masks
        var_exists, var_to_impute = get_impute_masks(tag_to_impute,edge_attributes)
        #Get ML training sets
        # target_to_num, num_to_target, target, y_int, y_onehot, x_hot = feature_learning_train_sets(edge_attributes, tag_to_impute, tags_to_add)
        #Impute missing data
        if impute_method == 'knn-dist':
            print('Imputing data using method - KNN Dist')
            imputed_vals = knn_dist_impute(edge_attributes,var_exists,var_to_impute,tag_to_impute)
        # elif impute_method == 'knn-feats':
        #     print('Imputing data using method - KNN Feats')
        #     #todo: default value for k
        #     imputed_vals = knn_feats(x_hot,var_exists,target,var_to_impute,k = 3)
        elif impute_method == 'mode-rule':
            print('Imputing data using method - Mode Rule')
            imputed_vals = mode_rule(edge_attributes,var_exists,var_to_impute,tag_to_impute)
            print('Data imputed')
        # elif impute_method == 'mlp':
        #     print('Imputing data using method - MLP')
        #     imputed_vals = mlp_impute(y_onehot,x_hot,mlp_train_params['hidden_layer'],var_exists,var_to_impute,mlp_train_params['batch_size'],mlp_train_params['n_epochs'],num_to_target)
        
        if impute_method == 'ottawa':
            edge_attributes.loc[var_to_impute,tag_to_impute] = 2
            edge_attributes[tag_to_impute] = edge_attributes[tag_to_impute].astype(float)
        else:
            edge_attributes.loc[var_to_impute,tag_to_impute] = imputed_vals
            edge_attributes[tag_to_impute] = edge_attributes[tag_to_impute].astype(float)

        edge_attributes = dedupe_var_replace(edge_attributes,tag_to_impute)
        
    else:
        edge_attributes['lanes'] = 2

    #Impute Surface
    print('Imputing Surface')
    tag_to_impute = 'surface'
    print(edge_attributes.columns)
    if tag_to_impute in edge_attributes.columns:
        #Imputation masks
        var_exists, var_to_impute = get_impute_masks(tag_to_impute,edge_attributes)
        #Get ML training sets
        # target_to_num, num_to_target, target, y_int, y_onehot, x_hot = feature_learning_train_sets(edge_attributes, tag_to_impute, tags_to_add)

        #Impute missing data
        if impute_method == 'knn-dist':
            print('Imputing data using method - KNN Dist')
            imputed_vals = knn_dist_impute(edge_attributes,var_exists,var_to_impute,tag_to_impute)
        # elif impute_method == 'knn-feats':
        #     print('Imputing data using method - KNN Feats')
        #     #todo: default value for k
        #     imputed_vals = knn_feats(x_hot,var_exists,target,var_to_impute,k = 3)
        elif impute_method == 'mode-rule':
            print('Imputing data using method - mode rule')
            imputed_vals = mode_rule(edge_attributes,var_exists,var_to_impute,tag_to_impute)
            print('Data imputed')
        # elif impute_method == 'mlp':
        #     print('Imputing data using method - MLP')
        #     imputed_vals = mlp_impute(y_onehot,x_hot,mlp_train_params['hidden_layer'],var_exists,var_to_impute,mlp_train_params['batch_size'],mlp_train_params['n_epochs'],num_to_target)

        if impute_method != 'ottawa':
            edge_attributes.loc[var_to_impute,tag_to_impute] = imputed_vals
            edge_attributes = dedupe_var_replace(edge_attributes,tag_to_impute)
    else:
        edge_attributes[tag_to_impute] = None
    
    #Add in Access and Footway tags if missing for LTS classification
    if 'access' not in edge_attributes.columns:
        edge_attributes['access'] = 'NAN'
    if 'footway' not in edge_attributes.columns:
        edge_attributes['footway'] = 'NAN'
    if 'bicycle' not in edge_attributes.columns:
        edge_attributes['bicycle'] = 'NAN'
    if 'motor_vehicle' not in edge_attributes.columns:
        edge_attributes['motor_vehicle'] = 'NAN'
    if 'service' not in edge_attributes.columns:
        edge_attributes['service'] = 'NAN'
        
    #Compute Edge-Level LTS 
    if lts_method == 'ottawa':
        print('Calculating LTS using Ottawa Advocacy Group method')
        lts = lts_ottawa(edge_attributes)
        edge_attributes['LTS'] = lts['LTS_ottawa']
        edge_attributes = pd.concat([edge_attributes, pd.get_dummies(lts['LTS_ottawa'])], axis=1)
    # elif lts_method == 'self-learn':
    #     print('Calculating LTS using Self-Learning Approach')
    #     print('WARNING : this approach is under development, please check your results carefully')
    #     lts = self_learn(edge_attributes,self_learn_k)
    #     edge_attributes['LTS'] = lts['cluster']
    #     edge_attributes = pd.concat([edge_attributes, pd.get_dummies(lts['cluster'])], axis=1)
    
    for col in [0,1,2,3,4]:
        if col not in edge_attributes.columns:
            edge_attributes[col] = 0
            
    return G, edge_attributes
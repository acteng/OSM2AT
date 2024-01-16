from .helper_functions import parking_present, biking_permited, separate_path
from scipy.stats import zscore
import gower
import kmedoids
import pandas as pd

#%%

def self_learn(training_data,k):
    
    cycle_way_columns = []
    for i in list(training_data.columns):
        if i[:8] == 'cycleway':
            cycle_way_columns.append(i)
    
    cycleway_values = ['crossing','lane','left','opposite','opposite_lane','right','yes']
    
    parking_columns = []
    
    for i in list(training_data.columns):
        if i[:7] == 'parking':
            parking_columns.append(i)
    
    parking_values = ['parallel','perpendicular','diagonal','yes','marked']
    
    #Add new features
    
    biking_permitted = []
    sep_paths = []
    bike_lane = []
    parking_list = []
    
    for i,r  in training_data.iterrows():
        
        cycling_permitted,lts,message,rule = biking_permited(r)
        biking_permitted.append(cycling_permitted)
        sep_path,lts,message,rule = separate_path(r,cycle_way_columns)
        sep_paths.append(sep_path)
        
        bike_append = False
        if len(cycle_way_columns) > 0:
            for col in cycle_way_columns:
                for val in cycleway_values:
                    if r[col] == val:
                        bike_append = True
        bike_lane.append(bike_append)
    
        parking = parking_present(r,parking_columns,parking_values)
        parking_list.append(parking)
    
    training_data['biking_permitted'] = biking_permitted
    training_data['sep_paths'] = sep_paths
    training_data['bike_lane'] = bike_lane
    training_data['parking_list'] = parking_list
    
    #Clustering
    
    #Prepare data for clustering
    cols_for_clustering = ['maxspeed', 'lanes','surface','biking_permitted','sep_paths', 'bike_lane','lit', 'sidewalk','bicycle','segregated','highway']
    numeric_cols = ['maxspeed', 'lanes']
    cluster_data = training_data[cols_for_clustering]
    
    #Nromalise using z-scaling (what about categorical? One hot)
    for col in numeric_cols:
        cluster_data['{}_zscore'.format(col)] = zscore(cluster_data[col])
        cluster_data = cluster_data.drop(col, axis=1)
    
    # #Training set as one hot encoded
    # cluster_onehot = pd.get_dummies(cluster_data).values
    
    #Compute Gower's similarity matrix
    cluster_onehot_gower = gower.gower_matrix(cluster_data)
    
    #Cluster data
    km = kmedoids.KMedoids(k, method='fasterpam')
    c = km.fit(cluster_onehot_gower)    
    
    clusters = c.labels_
    
    training_data['cluster'] = clusters
    
    #For each cluster, print mean speed, mean lanes and mode road type
    
    cluster_summaries = pd.DataFrame(index = list(training_data['cluster'].value_counts().index), columns = ['Avg Speed','Avg Lanes','Percent Res','Percent Prim'])
    
    for i in list(training_data['cluster'].value_counts().index):
        data_filt = training_data[training_data['cluster'] == i]
        cluster_summaries.loc[i,'Avg Speed'] = data_filt['maxspeed'].mean()
        cluster_summaries.loc[i,'Avg Lanes'] = data_filt['lanes'].mean()
        cluster_summaries.loc[i,'Percent Res'] = (data_filt['highway'] == 'residential').sum() / data_filt.shape[0]
        cluster_summaries.loc[i,'Percent Prim'] = (data_filt['highway'] == 'primary').sum() / data_filt.shape[0]
    
    clusters_ranked = cluster_summaries[['Avg Speed','Avg Lanes','Percent Prim']].rank(axis=0, method='min', ascending=True)
    clusters_ranked['Percent Res'] = cluster_summaries['Percent Res'].rank(axis=0, method='min', ascending=False)
    clusters_ranked['sum'] = clusters_ranked.sum(axis = 1)
    clusters_ranked = clusters_ranked.sort_values(by=['sum'])
    clusters_ranked['cluster_lts'] = range(len(clusters_ranked))
    
    cluster_lts_dict = {}
    for i,r in clusters_ranked.iterrows():
        cluster_lts_dict[i] = r['cluster_lts']
    
    training_data['cluster_lts'] = training_data['cluster'].map(cluster_lts_dict)
    
    return training_data
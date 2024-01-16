from .helper_functions import parking_present, biking_permited, separate_path, bikeLaneAnalysisNoParking, bikeLaneAnalysisParking, mixed_traffic

#%%

def lts_two_rule(edge_attributes):
    
    lts = []
    for i,r in edge_attributes.iterrows():
        if r['lanes'] >= 3:
            lts.append(4)
        elif r['maxspeed'] >= 35:
            lts.append(4)
        elif r['lanes'] <= 1.5:
            lts.append(2)
        elif r['maxspeed'] >= 30:
            lts.append(4)
        else:
            lts.append(3)
    
    edge_attributes['LTS_2r']  = lts
    
    return edge_attributes[['osmid','LTS_2r']]


def lts_ottawa(edge_attributes, width = 99999.9,output_vars = ['osmid','LTS_ottawa','LTS_ottawa_rule']):
    
    #ToDo - this requires a neater solution for pulling through cycleway and parking tags from OSM
    cycle_way_columns = []
    
    for i in list(edge_attributes.columns):
        if i[:8] == 'cycleway':
            cycle_way_columns.append(i)
    
    cycleway_values = ['crossing','lane','left','opposite','opposite_lane','right','yes']
    
    parking_columns = []
    
    for i in list(edge_attributes.columns):
        if i[:7] == 'parking':
            parking_columns.append(i)
    
    parking_values = ['parallel','perpendicular','diagonal','yes','marked']
    
    lts_scores = []
    rules = []

    count_cycle_checks = 0
    count_sep_path_checks = 0
    count_bike_lane_checks = 0
    count_mixed_traffic_checks = 0
    
    edge_attributes = edge_attributes.fillna('NAN')
    
    lts_scores = []
    rules = []
    
    for i,r  in edge_attributes.iterrows():
        
        count_cycle_checks += 1
        cycling_permitted,lts,message,rule = biking_permited(r)
        
        if lts is None:
            count_sep_path_checks += 1
            sep_path,lts,message,rule = separate_path(r,cycle_way_columns)
    
    
        if lts is None:
            count_bike_lane_checks += 1
            analyse = False
            if len(cycle_way_columns) > 0:
                for col in cycle_way_columns:
                    for val in cycleway_values:
                        if r[col] == val:
                            analyse = True
            try:
                if r['shoulder:access:bicycle'] == 'yes':
                    analyse = True
            except:
                pass    
            
            if analyse:
                parking = parking_present(r,parking_columns,parking_values)
                if parking:
                    lts,message,rule = bikeLaneAnalysisParking(r, width)
                else:
                    lts,message,rule = bikeLaneAnalysisNoParking(r, width)
    
        if lts is None:
            count_mixed_traffic_checks += 1
            lts,message,rule = mixed_traffic(r)
    
        lts_scores.append(lts)
        rules.append(rule)
    
    edge_attributes['LTS_ottawa'] = lts_scores
    edge_attributes['LTS_ottawa_rule'] = rules

    return edge_attributes[output_vars]
from sklearn.neighbors import KNeighborsClassifier

#%% Parking Present on an Edge

def parking_present(r,parking_columns,parking_values):

    parking_present = False
    
    try:
        if r['parking'] == 'yes':
            parking_present = True
    except:
        pass    
    
    if len(parking_columns) > 0:
        for col in parking_columns:
            for val in parking_values:
                if r[col] == val:
                    parking_present = True
                    
    return parking_present

#%%
def biking_permited(r):
    
    
    if r['highway'] != 'NAN' or r['bicycle'] != 'NAN':
        cycling_permitted = True
        lts = None
        message = None
        rule = None
        
        if r['bicycle'] == 'no':
            lts = 0
            message = "Cycling not permitted due to bicycle='no' tag."
            rule = "p2"
            cycling_permitted = False
        elif r['access'] == 'no':
            lts = 0
            message = "Cycling not permitted due to access='no' tag."
            rule = "p6"
            cycling_permitted = False
        elif r['highway'] == 'motorway':
            lts = 0
            message = "Cycling not permitted due to highway='motorway' tag."
            rule = "p3"
            cycling_permitted = False
        elif r['highway'] == 'motorway_link':
            lts = 0
            message = "Cycling not permitted due to highway='motorway_link' tag."
            rule = "p4"
            cycling_permitted = False
        elif r['highway'] == 'proposed':
            lts = 0
            message = "Cycling not permitted due to highway='proposed' tag."
            rule = "p7"
            cycling_permitted = False
        elif r['footway'] == 'sidewalk':
            if r['bicycle'] != 'yes':
                if (r['highway'] == 'footway') | (r['highway'] == 'path'):
                    lts = 0
                    message = "Cycling not permitted. When footway=\'sidewalk\' is present, there must be a bicycle=\'yes\' when the highway is \'footway\' or \'path\'."
                    rule = "p5"
                    cycling_permitted = False
    else:
        lts = 0
        message = "Way has neither a highway tag nor a bicycle=yes tag. The way is not a highway."
        rule = "p1"
        cycling_permitted = False

    return [cycling_permitted,lts,message,rule]

#%%
def separate_path(r,cycle_way_columns):
        
    seperated_path = False
    lts = None
    message = None
    rule = None
    
    if len(cycle_way_columns) > 0:
        for col in cycle_way_columns:
            if r[col] == 'track':
                lts = 1
                message = "This way is a separated path because cycleway* is defined as 'track'."
                rule = "s7"
                seperated_path = True
            elif r[col] == 'opposite_track':
                lts = 1
                message = "This way is a separated path because cycleway* is defined as 'opposite_track'."
                rule = "s8"
                seperated_path = True

     
    if r['highway'] == 'path':
        lts = 1
        message = "This way is a separated path because highway='path'."
        rule = "s1"
        seperated_path = True
    elif (r['highway'] == 'path') & (r['footway'] != 'crossing'):
        lts = 1
        message = "This way is a separated path because highway='footway' but it is not a crossing."
        rule = "s2"
        seperated_path = True
    elif r['highway'] == 'cycleway':
        lts = 1
        message = "This way is a separated path because highway='cycleway'."
        rule = "s3"
        seperated_path = True
    elif r['highway'] == 'construction':
        if r['construction'] == 'path':
            lts = 1
            message = "This way is a separated path because highway='construction' and construction='path'."
            rule = "s4"
            seperated_path = True
        elif r['construction'] == 'footway':
            lts = 1
            message = "This way is a separated path because highway='construction' and construction='footway'."
            rule = "s5"
            seperated_path = True
        elif r['construction'] == 'cycleway':
            lts = 1
            message = "This way is a separated path because highway='construction' and construction='cycleway'."
            rule = "s6"
            seperated_path = True
        
    return [seperated_path,lts,message,rule]

#%% Bike Lane

def bikeLaneAnalysisNoParking(r, width):
    
    lts = 1
    message = None
    rule = None
    
    if r['highway'] == 'residential':
        residential = True
    else:
        residential = False
    
    if r['lanes'] == 3:
        if lts < 2:
            lts = 2
            message = "Increasing LTS to 2 because there are 3 lanes with a separating median and no parking."
            rule = "c2"
    elif r['lanes'] >= 3:
        if lts < 3:
            lts = 3
            message = "Increasing LTS to 3 because there are 3 or more lanes and no parking."
            rule = "c3"
    
    if width <= 1.7:
        if lts < 2:
            lts = 2
            message = "Increasing LTS to 2 because the bike lane width is less than 1.7 metres and no parking."
            rule = "c4"
    
    if r['maxspeed'] > 30:
        if r['maxspeed'] < 40:
            if lts < 3:
                lts = 3
                message = "Increasing LTS to 3 because the maxspeed is between 30 and 40 mph and no parking."
                rule = "c5"
        else:
            if lts < 4:
                lts = 4
                message = "Increasing LTS to 4 because the maxspeed is over 40 mph and no parking."
                rule = "c6"
    
    if residential == False:
        if lts < 3:
            lts = 3
            message = "Increasing LTS to 3 because highway with bike lane is not 'residential' and no parking."
            rule = "c7"
    
    if lts == 1:
        message = "LTS is 1 because there is no parking, maxspeed is less than or equal to 50, highway='residential', and there are 2 lanes or less."
        rule = "c1"
        
    return [lts,message,rule]

#%%

def bikeLaneAnalysisParking(r, width): 

    lts  = 1
    message = None
    rule = None

    if r['highway'] == 'residential':
        residential = True
    else:
        residential = False    

    if r['lanes'] >= 3:
        if lts < 3:
            lts = 3
            message = "Increasing LTS to 3 because there are 3 or more lanes and parking present."
            rule = "b2"
    
    if width <= 4.1:
        if lts < 3:
            lts = 3
            message = "Increasing LTS to 3 because the bike lane width is less than 4.1m and parking present."
            rule = "b3"
            
        elif width <= 4.25:
            if lts < 2:
                lts = 2
                message = "Increasing LTS to 2 because the bike lane width is less than 4.25m and parking present."
                rule = "b4"
        
        elif (width <= 4.5) & (r['maxspeed'] <= 20 | residential):
            if lts < 2:
                lts = 2
                message = "Increasing LTS to 2 because the bike lane width is less than 4.5m. maxspeed is less than 30mph on a residential street and parking present."
                rule = "b5"        
    
    if r['maxspeed'] > 20:
        if r['maxspeed'] <= 30:
            if lts < 2:
                lts = 2
                message = "Increasing LTS to 2 because the maxspeed is between 20-30mph and parking present."
                rule = "b6"
        elif r['maxspeed'] < 40:
            if lts < 3:
                lts = 3
                message = "Increasing LTS to 3 because the maxspeed is between 30-40mph and parking present."
                rule = "b7"
        else:
            if lts < 4:
                lts = 4
                message = "Increasing LTS to 4 because the maxspeed is over 40mph and parking present."
                rule = "b8"            
    
    if residential == False:
        if lts < 3:
            lts = 3
            message = "Increasing LTS to 3 because highway is not 'residential'."
            rule = "b9"  
    
    if lts == 1:
        message = "LTS is 1 because there is parking present, the maxspeed is less than or equal to 40, highway='residential', and there are 2 lanes or less."
        rule = 'b1'
        
    return [lts,message,rule]

#%%

def mixed_traffic(r,):  

    message = 'Does not meet criteria for Separated Path or Bike Lane. Treating as Mixed Traffic.'
    
    lts = None
    message = None
    rule = None
    
    if r['highway'] == 'residential':
        residential = True
    else:
        residential = False    
    
    if r['motor_vehicle'] == 'no':
        lts = 1
        message = "Setting LTS to 1 because motor_vehicle='no'."
        rule = "m17"
    
    if r['highway'] == 'steps':
        lts = 1
        message = "Setting LTS to 1 because highway='steps'."
        rule = "m1"
    
    if r['highway'] == 'pedestrian':
        lts = 1
        message = "Setting LTS to 1 because highway='pedestrian'."
        rule = "m16"
    
    if (r['highway'] == 'footway') & (r['footway'] == 'crossing'):
        lts = 2
        message = "Setting LTS to 2 because highway='footway' and footway='crossing'."
        rule = "m14"
    
    if (r['highway'] == 'service') & (r['service'] == 'alley'):
        lts = 2
        message = "Setting LTS to 2 because highway='service' and service='alley'."
        rule = "m2"
        
    if r['highway'] == 'track':
        lts = 2
        message = "Setting LTS to 2 because highway='track'."
        rule = "m15"
    
    if r['maxspeed'] <= 30:
        if r['highway'] == 'service':
            if r['service'] == 'parking_aisle':
                lts = 2
                message = "Setting LTS to 2 because maxspeed is 30mph or less and service is 'parking_aisle'."
                rule = "m3"
            if r['service'] == 'driveway':
                lts = 2
                message = "Setting LTS to 2 because maxspeed is 30mph or less and service is 'driveway'."
                rule = "m4"            
            if r['maxspeed'] <= 20:
                lts = 2
                message = "Setting LTS to 2 because maxspeed is less than 20mph and highway='service'."
                rule = "m16"  
    
        if r['maxspeed'] <= 20:
            if (r['lanes'] <= 3) & (residential):
                lts = 2
                message = "Setting LTS to 2 because maxspeed is up to 20mph, 3 or fewer lanes and highway='residential'."
                rule = "m5"  
            elif r['lanes'] <= 3:
                lts = 3
                message = "Setting LTS to 3 because maxspeed is up to 20mph and 3 or fewer lanes on non-residential highway."
                rule = "m6"
            elif r['lanes'] <= 5:
                lts = 3
                message = "Setting LTS to 3 because maxspeed is up to 20mph and 4 or 5 lanes."
                rule = "m7"
            else:
                lts = 4
                message = "Setting LTS to 4 because maxspeed is up to 20mph and the number of lanes is greater than 5."
                rule = "m8"
    
        else:
            if (r['lanes'] < 3) & (residential):
                lts = 2
                message = "Setting LTS to 2 because maxspeed is up to 30mph and lanes are 2 or less and highway='residential'"
                rule = "m9"
            elif r['lanes'] <= 3:
                lts = 3
                message = "Setting LTS to 3 because maxspeed is up to 30mph and lanes are 3 or less on non-residential highway."
                rule = "m10"
            else:
                lts = 4
                message = "Setting LTS to 4 because the number of lanes is greater than 3."
                rule = "m11"
    
    else:
        lts = 4
        message = "Setting LTS to 4 because maxspeed is greater than 30mph."
        rule = "m12"
        
    return [lts,message,rule]
    
    
def dedupe_var_replace(df,tag_to_impute):

    #In instances where multiple edges with same osmid have different values, replace with mode observed value
    
    dedupe_var_dict = {}
    
    for i in list(set(list(df['osmid']))):
        dedupe_var_dict[i] = df[df['osmid'] == i][tag_to_impute].mode().values[0]
    
    df[tag_to_impute] = df['osmid'].map(dedupe_var_dict)
    
    return df
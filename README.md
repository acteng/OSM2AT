# OSM2AT

### How to install and use

To install, clone the repository, navigate into it, and run:
pip install . 

The two major functions within the package are:
-	get_cycle_network: Returns a NetworkX object representing the cycling network, along with a DataFrame of the network’s edges and features.
-	get_foot_network: Returns a NetworkX object representing the pedestrian network, along with a DataFrame of the network’s edges and features.

### Other notes on this project
-	For imputing missing tags, the "mode rule" is the most efficient method we’ve tested. It provides negligible loss of information compared to more sophisticated approaches and is our default.
  -	Mode rule: assigns the most common value (for edge which have been tagged) for missing tags given the edges highway type which is always populated.
-	Similarly, the "LTS Ottawa" method is our default for computing LTS, as it has been demonstrated to perform well against reliable ground truth data.
  -	Repo - https://github.com/BikeOttawa/stressmodel 
-	A number of other methods are included in the scripts within the functions folder, but they are unlikely to be used in practice.

### Ongoing maintenance
If in use regularly the repo would require some light touch maintenance. Much of the functionality is dependent on OSMNX, so when OSMNX receives major updates it may impact the functionality of OSM2AT.
Users of this may also want to update and adapt it as work develops at ATE. For example an idea I have had it to introduce user profiles which would generate access costs dependent on the pre-determined user profiles, so reflect different people’s preferences for different edge level features. This would potentially link in with the Dynamic Access Costs project.

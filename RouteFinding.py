import math
import time
import heapq
import psutil
import os
from collections import deque
import matplotlib.pyplot as plt
from queue import PriorityQueue
from matplotlib.widgets import Slider
import pandas as pd


RED = 'Red'
GREEN = 'green'
BLUE = "blue"
YELLOW = 'yellow'
WHITE = 'white'
ORANGE = 'orange'
GREY= 'grey'
PURPLE ='purple'

CITY_RADIUS = 20
size = 120
FONT_SIZE = 16
#route collection storage
route_collection = []
#track memory usage
# Function to track memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss #return memory usage in bytes
def load_cities(filename):
    "load cities loads the cities from csv file using pandas pd"
    try:
        file = pd.read_csv(filename)
        cities = {} #dictionary to hold cities and locations on the map
        for row in file.itertuples():
            try:
                city = str(row[1].strip())
                lat = float(row[2])
                lon = float(row[3])
                cities[city] = (lat, lon) #add coordinates
            except (IndexError, ValueError) as e:
                print(f"processing error {row}, :{e}")
        return cities
    except FileNotFoundError:
        print(f"file {filename}  not found")
        return {}
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{filename}' is empty.")
        return {}

    except Exception as e:
        print(f"Unabel to load cities from {filename} : {e}")
        return {}


def load_adjacent_cities(text_file):
    adjacency = {}  # dictionary to store adjacent cities
    # read a file text file
    with open(text_file, mode='r') as file:
        for text in file:  # loop through the file
            cityA, cityB = text.strip().split(" ")

            # create undirected graph to create bidirection connection between cities
            if cityA not in adjacency:
                adjacency[cityA] = []
            if cityB not in adjacency:
                adjacency[cityB] = []
            adjacency[cityA].append(cityB)
            adjacency[cityB].append(cityA)
    return adjacency

def calculate_distance(x1, y1, x2, y2):
    #calculating the distance between two cities using  haversine formula
    #distance between two latitudes and  longitudes in radians
    dlat = (x2 - x1) * math.pi /180.0
    dlon = (y2 - y1) * math.pi /180.0

    #convert latitude to radians
    lat1 = x1 * math.pi/180
    lat2 = x2 * math.pi/180
    #apply  haversine formula
    a = (pow(math.sin(dlat / 2),2) + math.cos(lat1) * math.cos(lat2)
         *pow(math.sin(dlon/2),2))
    c=  2 * math.asin(math.sqrt(a)) #angular distance
    radius = 6371  #radius on the earth kilometers
    return radius * c
def distance(start_city, goal_city, cities):
    #calculate the distance between tow cities
    lat1, lon1 = cities[start_city] # get the latitude and longitude of the current city
    lat2, lon2 = cities[goal_city] #get the latitude and longitude of the goal city
    return calculate_distance(lat1, lon1, lat2, lon2) #return distance between them
def weighted(neighbors, cities):
    """The methods will convert all adjacent cities to weighted"""
    weight = {} # dictionary to store the weighted cities

    missing = set()  #tracks missing cities
    # loop through cities with their neighbors
    for city, adjacent in neighbors.items():
        weight[city] = [] #list of neighbors

        #check if the city exist
        if city not in cities:
            missing.add(city)
            continue #move on
        #get distance for each neighbor
        for neighbor in adjacent:
            if neighbor not in cities:
                missing.add(neighbor)
                continue

                # Calculate distance
            try:
                cost = distance(city, neighbor, cities)
                weight[city].append((neighbor, cost))
            except Exception as e:
                print(f"Error calculating distance between {city} and {neighbor}: {e}")

                # Show missing cities
            if missing:
                print(f"Warning: {len(missing)} cities are missing coordinates:")
                print(", ".join(sorted(list(missing)[:5])), end="")
                if len(missing) > 5:
                    print(f", and {len(missing) - 5} more")
                else:
                    print()
    return weight
#reconstruct the path from start to finish
def reconstruct(previous, start, end):
    path = [end]
    current = end
    while current != start:
        if current not in previous:
            return None #No path ever existed
        current = previous[current]
        path.append(current) #add path
    return list(reversed(path))

def best_first_search( start_city, end_goal_city, adj, cities):
    queue = []  # a queue is initialized to empty to explore all the cities will visit
    # create a miniheap for the cities
    heapq.heappush(queue, (distance(start_city, end_goal_city, cities), start_city, [start_city], 0))

    # initially mark all the cities as not visited
    visited = set()
    visited.add(start_city)
    path = [start_city]  # track path of visited cities
    edge = []  # track the edges/cost of visited cities
    visited.add(start_city) # mark the start city as visited
    previous_start = {start_city: None}
    while queue:
        # pop the current city
        heuristic, current_city, current_path, cost = heapq.heappop(queue)
        #add current to path
        #if current_city not in path:
            #path.append(current_city)
        if end_goal_city == current_city:
            # reconstruct the path
            full_path = reconstruct(previous_start, start_city, end_goal_city)
            return full_path, current_path, edge
        # explore neighbors
        for neighbor, edge_cost in adj[current_city]:
            if neighbor not in visited:
                visited.add(neighbor)
                new_cost = edge_cost + cost  # update the cost from current to neighbor
                previous_start[neighbor] = current_city  # track previous  visited cities

                #new heuristic calculated for neighbor
                heuristic_new=(distance(neighbor, end_goal_city, cities))
                heapq.heappush(queue,(heuristic_new ,neighbor, current_path + [neighbor], new_cost))
                edge.append((current_city, neighbor, edge_cost))  # record the new path

    return path, None, edge # no path found

    #Implementing a bfs to finding the adjacent cities
def depth_first_search( start_city, end_goal_city, adjacent):

    traversal_path = []  # an empty list to track the path from start to the end goal
    visited = set() #an Empty visited set
    stack= [(start_city, [start_city])] #stack initialized with the start city and path
    edges= []
    while stack:
        current, path = stack.pop() #get the current city and path
        if current not in  visited:
            traversal_path.append(current) #append current city to path
            visited.add(current)
        #if the we have morethan 1 cities in path list
        if len(path) > 1:
            previous_city = path[-2] #send last city to path
            edges.append((previous_city, current)) #the edge from the previous to the current is recorded
        #if reach our goal return path taken
        if current == end_goal_city:
            return traversal_path, path, edges
        #get the neighbors of the current city
        if current not in adjacent:
            continue

        neighbors = list(adjacent[current])
        #traverse the neighbors in reversed to explore the cities in correct order
        for neighbor in sorted(neighbors, reverse=True):
            if neighbor not in visited:
                #append the neighbor to stack and its path
                stack.append((neighbor, path + [neighbor]))
    #if nothing found return the path traversed, edges, and none
    return traversal_path, None, edges


#help function to perform depth limited search
def limited_search(city,goal, neighbors,depth, traversal_path, path,edges):
    #add the current city to traversal_path
    if city not in traversal_path:
        traversal_path.append(city)
    if city == goal:
        return True
    if depth <= 0:
        return False
    if city not in neighbors:
        return False
    for neighbor in sorted(neighbors[city], reverse=True):
        #check if neigbhor is not in the path
        if neighbor not in path:
            #append the current city and the neighbor
            edges.append((city, neighbor))
            path.append(neighbor) #add to the path
            #check if goal is found
            if limited_search(neighbor, goal, neighbors, depth-1,traversal_path, path, edges):
                return True
            path.pop() #backtrack back
    return False

def iterative_deep_search( start_city, end_goal_city, adjacent, max_search = 1000):
    traversal = []  # track all the traversal path
    edges = [] #track the edges
    for search in range(max_search):
        path = [start_city]
        depth = []
        if limited_search(start_city, end_goal_city,adjacent,max_search,depth, path, edges):
            #add all new discoverd cities
            for city in depth:
                if city not in traversal:
                    traversal.append(city)

            return traversal, path, edges
        #add other cities discoverd
        for city in depth:
            if city not in traversal:
                traversal.append(city)
    return traversal, None, edges #no end goal found



def breadth_first_search(start, goal,  adj):
    queue= deque ([(start, [start])]) #initialized the current city and its path
    # mark the first city visited
    visited = set([start])
    path = [start] #tracks the order on how cities are traversed

    edge = []
    while queue:
        #pop the current city
        current_city, current_path = queue.popleft()
        #if end_goal_city equal current city, we found the city
        if current_city == goal:
            return path, current_path, edge #return current path and edge lisr

        #explore neighbors
        for neighbor, edge_cost in adj[current_city]:
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                #add the neighbor to queue with updated path
                queue.append((neighbor , current_path + [neighbor]))
                #record the edge that connects the current city to neighbor
                edge.append((current_city, neighbor, edge_cost))


     #no path found
    return path, None, edge



def a_star_search(start_city,dest, adjacent, cities):
    #check if city to start with or end city is not available
    if start_city not in adjacent  or dest not in adjacent:
        return [start_city] if start_city in adjacent else[], None, []

    #priority queue to store cities that needs to be explored
    open_set = [(0, start_city, 0, [start_city])]
    closed_set = set() #set of closed cities
    traversal_path = [start_city]
    f_score = {start_city: distance(start_city, dest, cities)}
    start_from= {}
    g_score= {start_city:0}
    while open_set:
        _,current, cost, path = heapq.heappop(open_set)
        traversal_path.append(current)
        #checking if the city has already been explored
        if current  in closed_set:
            continue
        #add city to path

        traversal_path.append(current)


        #mark as provessed
        closed_set.add(current)
        #check if we found destination
        if current == dest:
            edges = []
            for i in range(len(path)-1):
                edges.append((path[i], path[i+1]))


            return traversal_path, path, edges
        #check for neighbors
        for neighbor, edge_cost in adjacent[current]:
            if neighbor in closed_set: #already processed
                continue
            next_g = g_score[current] + edge_cost

            #check for alternative path
            if neighbor not in g_score or next_g < g_score[neighbor]:
                start_from[neighbor] = current #update the path
                g_score[neighbor] = next_g
                new_f_score =next_g + distance(neighbor, dest, cities)
                f_score[neighbor] = new_f_score
                f_score[neighbor] = new_f_score
                #update the available open route with new path
                heapq.heappush(open_set, (new_f_score, neighbor, next_g, path +[neighbor]))
        # eck the current city with the neighbor
    return traversal_path, None, []




        # set cordinates to fit the screen
def scale_coordinates( coordinates, width=12, height=8):
    # initializing min and max latitude and longitude to infinit
    longitude = []
    latitude = []
    for location, (lat, lon) in coordinates.items():
        latitude.append(lat)
        longitude.append(lon)

    min_lat = min(latitude)
    max_lat = max(latitude)
    min_lon = min(longitude)
    max_lon = max(longitude)

    lat_scale = max_lat - min_lat
    lon_scale = max_lon - min_lon
    #avoid division by zero
    lat_scale = lat_scale if lat_scale > 0 else 1
    lon_scale = lon_scale if lon_scale > 0 else 1

    # scale the coordinates to fit the height and width
    scaled_coordinates = {}
    for location, (lat, lon) in coordinates.items():
        if lat_scale > 0 and lon_scale > 0:
            scale_lat = ((lat - min_lat) / lat_scale) * height
            scale_lon = ((lon - min_lon) / lon_scale) * width
        else:
            scale_lat = height / 2
            scale_lon = width / 2
        scaled_coordinates[location] = (scale_lat, scale_lon)

    return scaled_coordinates


# Function to visualize the cities and the path
def visualize(cities, start=None, dest=None, path=None, traverse=None ,ax = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    ax.set_title("Path finding Visualization")
    #extract values
    lats = [lat for lat, lon in cities.values()]
    lons = [lon for lat, lon in cities.values()]
    # Set axis limits
    min_lat, max_lat = min(lats)-1, max(lats)+1
    min_lon, max_lon  = min(lons)-1, max(lons)+1

    #add limts to x and y axis
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)

    #add labels on x and y
    ax.set_xlabel('Longitude')
    ax.set_ylabel("Latitude")


   #plot all cities
    for city, (lat, lon) in cities.items():
        color = 'b' #for blue
        size = 10
        if start and city == start:
            color = 'g' #green
            size = 10
        elif dest and city == dest:
            color = 'red'
            size = 8
        ax.plot(lon, lat, marker='o', color=color, markersize=size)
        ax.text(lon + 0.1, lat + 0.1, city, fontsize=9)
        # show traversed cities
        if traverse:
            traverse_coords = []
            for city1 in traverse:
                if city1 in cities:
                    traverse_coords.append(cities[city1])

            if traverse_coords:
                traverse_lat, traverse_lon = zip(*traverse_coords)
                ax.plot(traverse_lon, traverse_lat, 'o-', color=PURPLE, alpha=0.5,
                        markersize=3, linewidth=1, label='Traversal')

    if path:
        path_coords = [cities[city] for city in path]
        path_lat, path_lon = zip(*path_coords)
        ax.plot(path_lon, path_lat, color='r', linewidth=2, label='Path')
    #add legends

    # Zooming feature: Add a slider to control zoom level
    ax_slider = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor='yellow')
    zoom_slider = Slider(ax_slider, 'Zoom', 1, 3, valinit=1)

    def update(val):
        zoom_level = zoom_slider.val
        zoom_range = (max_lat - min_lat)/zoom_level
        zoom_range_lon =(max_lon - min_lon)/zoom_level

        #center map
        center_lat = (min_lat + max_lat)/2
        center_lon = (min_lon + max_lon)/2

        #set limit based on zoom level
        ax.set_xlim(center_lon - zoom_range_lon/2, center_lon + zoom_range_lon/2)
        ax.set_ylim(center_lat - zoom_range/2, center_lat + zoom_range_lon/2 )
        fig.canvas.draw_idle()

    zoom_slider.on_changed(update)

    plt.show()



def compare_all_algorithm(start_city, dest, weighted, cities):
    # Apply all algorithms with standardized returns

    def standardize_bfs(start, goal, adj):
        traversal, path, edges = breadth_first_search(start, goal, adj)
        return traversal or [start], path, edges or []

    def standardize_dfs(start, goal, adj):
        # For DFS, convert weighted adjacency to unweighted
        unweighted = {} #dictionary to hold unweightd
        for city, neighbors in adj.items():
            unweighted[city] = []
            for neighbor, _ in neighbors:
                unweighted[city].append(neighbor)
                #call the depth first search algorithm
        traversal, path, edges = depth_first_search(start, goal, unweighted)
        return  traversal or [start], path, edges or []

    def standardize_astar(start, goal, adj, cities):
        traversal, path, edges = a_star_search(start, goal, adj, cities)
        return traversal or [start], path, edges or []

    def standardize_best(start, goal, adj, cities):
        traversal, path, edges = best_first_search(start, goal, adj, cities)
        return traversal or [start], path, edges or []

    def standardize_ids(start, goal, adj):
        # For IDS, convert weighted adjacency to unweighted
        unweighted = {}
        for city, neighbors in  adj.items():
            unweighted[city] = []
            for neighbor, _ in neighbors:
                unweighted[city].append(neighbor)
        traverse, path, edges = iterative_deep_search(start,goal, unweighted)
        return traversal or [start],path, edges or []
    # Dictionary mapping algorithm names to their standardized functions
    algorithms = {
        "BFS": lambda: standardize_bfs(start_city, dest, weighted),
        "DFS": lambda: standardize_dfs(start_city, dest, weighted),
        "A*": lambda: standardize_astar(start_city, dest, weighted, cities),
        "Best": lambda: standardize_best(start_city, dest, weighted, cities),
        "IDDs": lambda: standardize_ids(start_city, dest, weighted)
    }

    # Execute all algorithms and collect results
    results = {}
    for name, algorithm_func in algorithms.items():
        try:
            initial_memory = get_memory_usage()  # Track memory before the algorithm start
            start_time = time.perf_counter()
            traversal, path, edges = algorithm_func()
            end_time = time.perf_counter()
            final_memory = get_memory_usage() #get the final memory usage

            # Calculate path distance if path exists
            path_distance = float('inf')
            if path:
                path_distance = sum(distance(path[i], path[i+1], cities)
                                  for i in range(len(path)-1))

            results[name] = {
                "Path": path,
                "Traverse": traversal,
                "Edges": edges,
                "Memory": final_memory - initial_memory,
                "Time": end_time - start_time,
                "Distance": path_distance,
                "Cities_Visited": len(traversal) if traversal else 0
            }
        except Exception as e:
            print(f"Error running {name} algorithm: {e}")
            results[name] = {
                "Path": None,
                "Traverse": [],
                "Edges": [],
                 "Memory": 0,
                "Time": 0,
                "Distance": float('inf'),
                "Cities_Visited": 0
            }

    return results
def start_program():
    """The start program demonstrate the city pathfinding algorithm"""
    print("====City Path Finding Program===")

    cities = load_cities('coordinates.csv')
    if not cities:
        print("No data loaded")
        return
    print(f"loaded {len(cities)} file loaded ")
    file = "Adjacencies.txt"
    adjacencies = load_adjacent_cities(file)
    if not adjacencies:
        print("No data loaded")
        return
    print(f"loaded {len(adjacencies)} file loaded ")
    #calculate weight
    weighted_adj =weighted(adjacencies, cities)
    #scaled for visualization
    scaled = scale_coordinates(cities)


    # Get valid start city
    start_city = None
    while start_city not in cities:
        start_input = input("Enter start city name: ")
        if start_input in cities:
            start_city = start_input
        else:
            print("City not found. Please try again.")


    # Get valid destination city
    dest_city = None
    while dest_city not in cities or dest_city == start_city:
        dest_input = input("Enter destination city name: ")
        if dest_input in cities and dest_input != start_city:
            dest_city = dest_input
        elif dest_input == start_city:
            print("Destination cannot be the same as start city.")
        else:
            print("City not found. Please try again.")

    print(f"\nFinding paths from {start_city} to {dest_city}...")

    # Run algorithm comparison
    results = compare_all_algorithm(start_city, dest_city,  weighted_adj, cities)
    if not results:
        print("No path found")
        return

    # Display results
    print("\nAlgorithm Comparison Results")
    print(f"{'Algorithm':<10} {'Path Found':<10} {'Path Length':<15} {'Cities Visited':<15} {'Memory (bytes})': <15}{'Time (ms)':<15}")
    print("-" * 65)

    for algo, data in results.items():
        path_found = "Yes" if data["Path"] else "No"
        path_length = len(data["Path"]) if data["Path"] else 0
        cities_visited = len(data["Traverse"]) if data["Traverse"] else 0
        time_taken = data["Time"]
        memory_used = data["Memory"]

        print(f"{algo:<10} {path_found:<10} {path_length:<15} {cities_visited:<15} {memory_used:<15.2f}{time_taken:<15.6f}")
    # Visualization menu
    while True:
        print("\nVisualization Options:")
        print("1. View all algorithms")
        print("2. View specific algorithm")
        print("3. Exit program")

        choice = input("Enter your choice (1-3): ")

        if choice == "1":
            # Visualize all algorithms one by one
            for algo, data in results.items():
                if data["Path"]:
                    print(f"\nVisualization for {algo}:")
                    print(f"Path: {' -> '.join(data['Path'])}")
                    visualize(scaled, start_city, dest_city, data["Path"], data["Traverse"])
                else:
                    print(f"\n{algo} did not find a path.")

        elif choice == "2":
            # Choose specific algorithm
            print("\nChoose algorithm:")
            for i, algo in enumerate(results.keys()):
                print(f"{i + 1}. {algo}")

            algo_choice = input("Enter algorithm number: ")
            try:
                algo_idx = int(algo_choice) - 1
                algo_name = list(results.keys())[algo_idx]
                data = results[algo_name]

                if data["Path"]:
                    print(f"\nVisualization for {algo_name}:")
                    print(f"Path: {' -> '.join(data['Path'])}")
                    visualize(scaled, start_city, dest_city, data["Path"], data["Traverse"])
                else:
                    print(f"\n{algo_name} did not find a path.")
            except (ValueError, IndexError):
                print("Invalid selection.")

        elif choice == "3":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == '__main__':
    start_program()
















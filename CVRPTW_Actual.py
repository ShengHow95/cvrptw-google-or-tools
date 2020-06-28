"""Capacited Vehicles Routing Problem (CVRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import pandas as pd
import numpy as np

import math

def create_data_model():
    """Stores the data for the problem."""
    
    data = {}

    # Read Data From File
    df = pd.read_excel('./Address_withCoord_Dist_Time_OffLoad.xlsx')

    firstTimeIndex = np.where(df.columns.values=="TimeFrom_0")
    firstDistanceIndex = np.where(df.columns.values=="DistanceFrom_0")
    weightIndex = np.where(df.columns.values=="Weight")

    timeWindows = df[['Time Window (Start Time)', 'Time Window (End Time)']].apply(tuple, axis=1).values.tolist()
    distanceMatrix = df.iloc[:,firstDistanceIndex[0][0]:(firstDistanceIndex[0][0]+len(df))].values.tolist()
    timeMatrix = df.iloc[:,firstTimeIndex[0][0]:(firstTimeIndex[0][0]+len(df))].values.tolist()

    weightList = df.iloc[:,weightIndex[0][0]].values.tolist()
    weightList = [ math.ceil(elem) for elem in weightList ]
    
    # Parse Data into Matrix Form
    data['distance_matrix'] = distanceMatrix
    data['time_matrix'] = timeMatrix
    data['time_windows'] = timeWindows
    data['demands'] = weightList
    data['vehicle_capacities'] = [3000, 3000, 3000, 5000]
    data['num_vehicles'] = 4
    data['depot'] = 0
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    
    total_distance = 0
    total_load = 0
    total_time = 0
    time_dimension = routing.GetDimensionOrDie('Time')

    for vehicle_id in range(data['num_vehicles']):
        
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0

        while not routing.IsEnd(index):

            # Time Variable to Store Cummulated Time
            time_var = time_dimension.CumulVar(index)

            # Node Index
            node_index = manager.IndexToNode(index)

            # Demands
            route_load += data['demands'][node_index]

            # plan_output += ' {0} Load({1}),Time({2},{3}) -> '.format(node_index, route_load, solution.Min(time_var), solution.Max(time_var))
            plan_output += ' {0} Load({1}),Time({2}) -> '.format(node_index, data['demands'][node_index], solution.Min(time_var))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            
            # route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            if(index < len(data['distance_matrix']) and previous_index < len(data['distance_matrix'])):
                route_distance += data['distance_matrix'][previous_index][index]
            elif(index >= len(data['distance_matrix']) and previous_index >= len(data['distance_matrix'])):
                route_distance += data['distance_matrix'][0][0]
            elif(index >= len(data['distance_matrix'])):
                route_distance += data['distance_matrix'][previous_index][0]
            elif(previous_index >= len(data['distance_matrix'])):
                route_distance += data['distance_matrix'][0][index]
        
        time_var = time_dimension.CumulVar(index)
        # plan_output += ' {0} Load({1}),Time({2},{3})\n'.format(manager.IndexToNode(index), route_load, solution.Min(time_var), solution.Max(time_var))
        plan_output += ' {0} Load({1}),Time({2})\n'.format(manager.IndexToNode(index), data['demands'][0], solution.Min(time_var))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}kg\n'.format(route_load)
        plan_output += 'Time of the route: {}min\n'.format(solution.Min(time_var))
        
        print(plan_output)

        # Add up total for all metrics
        total_distance += route_distance
        total_load += route_load
        total_time += solution.Min(time_var)

    print('Total distance of all routes: {}m'.format(total_distance))
    print('Total load of all routes: {}kg'.format(total_load))
    print('Total time of all routes: {}min'.format(total_time))


def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a distance callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    distance_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)


    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')
    

    # Add Time Window Constraint
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node] 

    time_callback_index = routing.RegisterTransitCallback(time_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)

    # Add Time Windows constraint.
    time = 'Time'
    routing.AddDimension(
        time_callback_index,
        0,  # allow waiting time
        600,  # maximum time per vehicle
        True,  # Don't force start cumul to zero.
        time)
    time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
    
    # Add time window constraints for each vehicle start node.
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0],
                                                data['time_windows'][0][1])

    # Instantiate route start and end times to produce feasible times.
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT)
    search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH)
    search_parameters.time_limit.seconds = 300
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.log_search = True

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)


if __name__ == '__main__':
    main()
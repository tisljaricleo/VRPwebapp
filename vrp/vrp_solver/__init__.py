"""Vehicles Routing Problem (VRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from .misc import open_pickle, save_pickle_data, distance_diff
from datetime import datetime
from ..models import RoutingPlan, VrpProblem



def get_distance_matrix(vrp_points):
    distance_matrix = list([])

    distance_matrix = list([])
    for i in range(0, len(vrp_points)):

        curr = vrp_points[i]
        temp = list([])

        for j in range(0, len(vrp_points)):

            if i == j:
                temp.append(0)
                continue

            next_ = vrp_points[j]
            distance = distance_diff(float(curr.lat), float(curr.lon), float(next_.lat), float(next_.lon))
            temp.append(distance)

        distance_matrix.append(temp)
    
    return distance_matrix



def create_data_model(vrp_points, depot_id, n_veh):
    """Stores the data for the problem."""
    data = {}
    # data['distance_matrix'] = open_pickle(r'C:\Users\student\Desktop\code_V1\small_distance_matrix.pkl')
    data['distance_matrix'] = get_distance_matrix(vrp_points)
    data['num_vehicles'] = n_veh
    data['depot'] = depot_id
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""

    total_string = []
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Route dist [m]: {}m\n'.format(route_distance)
        total_string.append(plan_output)
    return total_string
    # print(plan_output)
    # max_route_distance = max(route_distance, max_route_distance)
    #print('Maximum of the route distances: {}m'.format(max_route_distance))


def solution_dict(data, manager, routing, solution):
    """Prints solution on console."""

    routing_report = list([])

    for vehicle_id in range(data['num_vehicles']):
        veh_route = dict({})
        veh_route['veh_id'] = vehicle_id
        veh_route['route'] = list([])

        index = routing.Start(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            veh_route['route'].append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)

        veh_route['route'].append(manager.IndexToNode(index))
        veh_route['distance'] = route_distance

        routing_report.append(veh_route)

    return routing_report


def solve_vrp(vrp_points, problem_id, depot_id, ffs, lsm, n_veh):
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model(vrp_points, depot_id, n_veh)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3000000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    firstSolutions = {
        'PATH_CHEAPEST_ARC': (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC),
        'AUTOMATIC': (routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC),
        'PATH_MOST_CONSTRAINED_ARC': (routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC),
        'EVALUATOR_STRATEGY': (routing_enums_pb2.FirstSolutionStrategy.EVALUATOR_STRATEGY),
        'SAVINGS': (routing_enums_pb2.FirstSolutionStrategy.SAVINGS),
        'SWEEP': (routing_enums_pb2.FirstSolutionStrategy.SWEEP),
        'CHRISTOFIDES': (routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES),
        'ALL_UNPERFORMED': (routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED),
        'BEST_INSERTION': (routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION),
        'PARALLEL_CHEAPEST_INSERTION': (routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION),
        'LOCAL_CHEAPEST_INSERTION': (routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION),
        'GLOBAL_CHEAPEST_ARC': (routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC),
        'LOCAL_CHEAPEST_ARC': (routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC),
        'FIRST_UNBOUND_MIN_VALUE': (routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE)    
    }

    search_parameters.first_solution_strategy = firstSolutions[ffs]
    
    localSearchMetas = {
        'AUTOMATIC': (routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC),
        'GREEDY_DESCENT': (routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT),
        'GUIDED_LOCAL_SEARCH': (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH),
        'SIMULATED_ANNEALING': (routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING),
        'TABU_SEARCH': (routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH)
    }
    
    search_parameters.local_search_metaheuristic = localSearchMetas[lsm]


    


    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        total_string = print_solution(data, manager, routing, solution)
        rez = solution_dict(data, manager, routing, solution)
        # save_pickle_data('routing_rez.pkl', rez)

        for r in rez:

            rp = ';'.join([str(i) for i in r['route']])

            routing_plan=RoutingPlan(problem=VrpProblem.objects.get(id=problem_id),
                               routing_plan=rp,
                               vehicle_id=r['veh_id'],
                               total_distance=r['distance'])
            routing_plan.save()

        return total_string, rez

    return None, None

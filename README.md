This repository contains the solution to the MSc Business Analytics 2023-24 assignment on logistics optimization, aimed at minimizing the total energy required for transporting products to a set of geographically dispersed customers. The solution implements various algorithms to optimize the transportation routes, including the Clarke and Wright savings algorithm and tabu search.

Problem Overview:
The objective is to minimize the total tn x km (ton-kilometer) travelled by a fleet of homogeneous vehicles, considering both the tare weight of the vehicles and the product weight. The logistics problem involves transporting products from a central warehouse to 250 customers, each with a predetermined demand.

Problem Characteristics:
  Fleet: Homogeneous leased fleet, each vehicle has:
    Tare weight (empty vehicle weight): 6 tons
    Maximum carrying capacity: 8 tons
  Depot: Central warehouse where all vehicles start their routes
  Customers: 250 geographically dispersed customers, each with a known demand
  Routes: Each customer is visited exactly once by a vehicle, and vehicles may have open routes (routes end at the last served customer).

Objective: Minimize the total gross tn x km (empty weight + weight of transported goods) travelled by all vehicles.

Constraints:
The solution must be computed within 5 minutes on a modern PC.
Every customer must be served exactly once by a single vehicle.
The total load on any vehicle should not exceed its capacity.

Solution Approach:
Clarke and Wright Savings Algorithm: A greedy algorithm that builds initial feasible routes by merging customer pairs that offer the most significant savings.
Tabu Search: A metaheuristic optimization technique used to iteratively improve the solution by exploring neighboring solutions while avoiding cycling back to previously visited solutions.
2-opt Local Search: A local search optimization used within the routes to swap nodes and reduce the cost.

Files:
Instance.txt: The input file containing the customer data (coordinates and demand) and other parameters of the problem.
lso_comp_assignment.py: Python code for building the model, solving the optimization problem, and generating the solution file.
sol_checker.py: A solution checker script to verify if the generated solution meets the problem's constraints.



import random
import math
import copy
import matplotlib.pyplot as plt

filepath = r"path\Instance.txt"

class Model:
    def __init__(self):
        self.allNodes = []
        self.customers = []
        self.matrix = []
        self.capacity = -1
        self.empty_vehicle_weight = -1

    def BuildModel(self, filepath):
        with open(filepath, 'r') as file:
            self.capacity = int(next(file).split(',')[1])
            self.empty_vehicle_weight = int(next(file).split(',')[1])
            next(file)
            next(file)
            next(file)

            for line in file:
                parts = line.strip().split(',')
                if len(parts) < 4:
                    continue
                idd, x, y, demand = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                node = Node(idd, x, y, demand)
                self.allNodes.append(node)
                if idd != 0:
                    self.customers.append(node)


        rows = len(self.allNodes)
        self.matrix = [[0.0 for _ in range(rows)] for _ in range(rows)]
        for i in range(rows):
            for j in range(rows):
                a = self.allNodes[i]
                b = self.allNodes[j]
                dist = math.sqrt(math.pow(a.x - b.x, 2) + math.pow(a.y - b.y, 2))
                self.matrix[i][j] = dist

class Node:
    def __init__(self, idd, xx, yy, dem):
        self.x = xx
        self.y = yy
        self.ID = idd
        self.demand = dem
        self.isRouted = False

class Route:
    def __init__(self, dp, cap):
        self.sequenceOfNodes = []
        self.sequenceOfNodes.append(dp)
        self.cost = 0
        self.capacity = cap
        self.load = 0

class Solution:
    def __init__(self):
        self.cost = 0.0
        self.routes = []

class Saving:
    def __init__(self, n1, n2, sav):
        self.n1 = n1
        self.n2 = n2
        self.score = sav

class TabuList:
    def __init__(self, max_size):
        self.tabu_list = []
        self.max_size = max_size

    def add_solution(self, solution):
        if len(self.tabu_list) >= self.max_size:
            self.tabu_list.pop(0)
        self.tabu_list.append(solution)

class SolDrawer:
    @staticmethod
    def get_cmap(n, name='hsv'):
        return plt.cm.get_cmap(name, n)

    @staticmethod
    def draw(name, sol, nodes):
        plt.clf()
        SolDrawer.drawPoints(nodes)
        SolDrawer.drawRoutes(sol)
        plt.savefig(str(name))

    @staticmethod
    def drawPoints(nodes:list):
        x = []
        y = []
        for i in range(len(nodes)):
            n = nodes[i]
            x.append(n.x)
            y.append(n.y)
        plt.scatter(x, y, c="blue")

    @staticmethod
    def drawRoutes(sol):
        cmap = SolDrawer.get_cmap(len(sol.routes))
        if sol is not None:
            for r in range(0, len(sol.routes)):
                rt = sol.routes[r]
                for i in range(0, len(rt.sequenceOfNodes) - 1):
                    c0 = rt.sequenceOfNodes[i]
                    c1 = rt.sequenceOfNodes[i + 1]
                    plt.plot([c0.x, c1.x], [c0.y, c1.y], c=cmap(r))


class Solver:
    def __init__(self, m, tabu_list_size=10):
        self.allNodes = m.allNodes
        self.customers = m.customers
        self.depot = m.allNodes[0]
        self.distanceMatrix = m.matrix
        self.capacity = m.capacity
        self.empty_vehicle_weight = m.empty_vehicle_weight
        self.sol = None
        self.bestSolution = None
        self.tabu_list = TabuList(tabu_list_size)

    def tabu_search(self, iterations=100):
        current_solution = self.sol
        best_solution = current_solution

        for _ in range(iterations):
            neighbors = self.generate_neighbors(current_solution)
            best_neighbor = self.find_best_neighbor(neighbors)

            if best_neighbor.cost < current_solution.cost and best_neighbor not in self.tabu_list.tabu_list:
                current_solution = best_neighbor
                self.tabu_list.add_solution(current_solution)

            if best_neighbor.cost < best_solution.cost:
                best_solution = best_neighbor

        self.sol = best_solution
        return best_solution

    def local_search(self, iterations=50):
        for _ in range(iterations):
            for route in self.sol.routes:
                if len(route.sequenceOfNodes) > 3:
                    improvement = self.apply_2_opt_in_route(route)
                    if improvement:
                        self.sol.cost = self.calculate_total_cost(self.sol, self.empty_vehicle_weight, self.distanceMatrix)
                        break

    def calculate_total_cost(self, sol, empty_vehicle_weight, distanceMatrix):
        total_cost = 0
        for route in sol.routes:
            route_cost = 0
            total_route_load = empty_vehicle_weight + sum(node.demand for node in route.sequenceOfNodes)
            for i in range(len(route.sequenceOfNodes) - 1):
                from_node = route.sequenceOfNodes[i]
                to_node = route.sequenceOfNodes[i + 1]
                route_cost += distanceMatrix[from_node.ID][to_node.ID] * total_route_load
                total_route_load -= to_node.demand
            total_cost += route_cost
        return total_cost

    def calculate_route_cost(self, route, new_sequence_of_nodes):
        total_cost = 0
        for i in range(len(new_sequence_of_nodes) - 1):
            from_node = new_sequence_of_nodes[i]
            to_node = new_sequence_of_nodes[i + 1]
            total_cost += self.distanceMatrix[from_node.ID][to_node.ID]
        return total_cost

    def generate_neighbors(self, solution):
        neighbors = []

        for i in range(len(solution.routes)):
            route = solution.routes[i]
            for j in range(1, len(route.sequenceOfNodes) - 1):
                for k in range(j + 1, len(route.sequenceOfNodes)):
                    neighbor_solution = self.swap_nodes(solution, i, j, k)
                    neighbors.append(neighbor_solution)

        return neighbors

    def swap_nodes(self, solution, route_index, node_index1, node_index2):
        neighbor_solution = copy.deepcopy(solution)
        route = neighbor_solution.routes[route_index]
        route.sequenceOfNodes[node_index1], route.sequenceOfNodes[node_index2] = \
            route.sequenceOfNodes[node_index2], route.sequenceOfNodes[node_index1]

        self.UpdateRouteCostAndLoad(route)

        return neighbor_solution

    def find_best_neighbor(self, neighbors):
        best_neighbor = None
        for neighbor in neighbors:
            neighbor_cost = self.calculate_total_cost(neighbor, self.empty_vehicle_weight, self.distanceMatrix)

            if best_neighbor is None or neighbor_cost < self.calculate_total_cost(best_neighbor,
                                                                                  self.empty_vehicle_weight,
                                                                                  self.distanceMatrix):
                best_neighbor = neighbor

        return best_neighbor

    def Clarke_n_Wright(self):
        self.sol = self.create_initial_routes()
        savings = self.calculate_savings()
        savings.sort(key=lambda s: s.score, reverse=True)

        for sav in savings:
            n1, n2 = sav.n1, sav.n2
            rt1, rt2 = n1.route, n2.route

            if rt1 == rt2 or self.not_first_or_last(rt1, n1) or self.not_first_or_last(rt2,
                                                                                       n2) or rt1.load + rt2.load > self.capacity:
                continue

            self.merge_routes(n1, n2)


        for route in self.sol.routes:
            if route.load > route.capacity:
                print('Capacity violation in merged route:', route.sequenceOfNodes)
                return

        self.sol.cost = self.calculate_total_cost(self.sol, self.empty_vehicle_weight, self.distanceMatrix)

        for i, route in enumerate(self.sol.routes):
            print(f"Route {i}: Nodes {[node.ID for node in route.sequenceOfNodes]}, Load: {route.load}")

    def calculate_savings(self):
        savings = []
        for i in range(0, len(self.customers)):
            n1 = self.customers[i]
            for j in range(i + 1, len(self.customers)):
                n2 = self.customers[j]
                score = self.distanceMatrix[n1.ID][self.depot.ID] + self.distanceMatrix[self.depot.ID][n2.ID]
                score -= self.distanceMatrix[n1.ID][n2.ID]
                sav = Saving(n1, n2, score)
                savings.append(sav)
        return savings

    def create_initial_routes(self):
        s = Solution()
        for i in range(0, len(self.customers)):
            n = self.customers[i]
            if n.demand > 8:
                raise Exception(f"Node with ID {n.ID} has demand greater than capacity: {n.demand}")
            rt = Route(self.depot, self.capacity)
            n.route = rt
            n.position_in_route = 1
            rt.sequenceOfNodes.append(n)
            rt.load = n.demand
            rt.cost = self.distanceMatrix[self.depot.ID][n.ID]
            s.routes.append(rt)
            s.cost += rt.cost
        return s

    def not_first_or_last(self, rt, n):
        if n.position_in_route != 1 and n.position_in_route != len(rt.sequenceOfNodes):
            return True
        return False

    def update_route_customers(self, rt):
        for i in range(1, len(rt.sequenceOfNodes) - 1):
            n = rt.sequenceOfNodes[i]
            n.route = rt
            n.position_in_route = i

    def calculate_route_cost_and_load(self, route):
        tc = 0
        tl = 0
        for i in range(len(route.sequenceOfNodes) - 1):
            A = route.sequenceOfNodes[i]
            B = route.sequenceOfNodes[i + 1]
            tc += self.distanceMatrix[A.ID][B.ID]
            tl += B.demand

        route.load = tl
        route.cost = tc

    def merge_routes(self, n1, n2):

        rt1 = n1.route
        rt2 = n2.route

        if rt1 == rt2 or rt1.load + rt2.load > self.capacity:
            return

        rt1.sequenceOfNodes.extend(rt2.sequenceOfNodes[1:])
        rt1.load = rt1.load + rt2.load


        self.calculate_route_cost_and_load(rt1)

        if rt1.load > self.capacity:

            rt1.sequenceOfNodes = rt1.sequenceOfNodes[:len(rt1.sequenceOfNodes) - len(rt2.sequenceOfNodes[1:])]
            rt1.load -= rt2.load
            return


        if rt2 in self.sol.routes:
            self.sol.routes.remove(rt2)


        for i, n in enumerate(rt1.sequenceOfNodes[1:], start=1):
            n.route = rt1
            n.position_in_route = i

    def apply_2_opt(self):
        improvement = True
        while improvement:
            improvement = False
            for route in self.sol.routes:
                for i in range(1, len(route.sequenceOfNodes) - 2):
                    for j in range(i + 2, len(route.sequenceOfNodes)):
                        if self.two_opt_swap(route, i, j):
                            improvement = True

    def apply_2_opt_in_route(self, route):
        for i in range(1, len(route.sequenceOfNodes) - 2):
            for j in range(i + 2, len(route.sequenceOfNodes)):
                improvement = self.two_opt_swap(route, i, j)
                if improvement:
                    return True
        return False

    def two_opt_swap(self, route, i, j):
        new_route = route.sequenceOfNodes[:i] + route.sequenceOfNodes[i:j + 1][::-1] + route.sequenceOfNodes[j + 1:]
        new_cost = self.calculate_route_cost(route, new_route)
        if new_cost < route.cost:
            route.sequenceOfNodes = new_route
            route.cost = new_cost
            return True
        return False

    def SetRoutedFlagToFalseForAllCustomers(self):
        for i in range(0, len(self.customers)):
            self.customers[i].isRouted = False
        for c in self.customers:
            c.isRouted = False

    def solve(self):
        self.SetRoutedFlagToFalseForAllCustomers()
        self.Clarke_n_Wright()
        self.apply_2_opt()
        self.sol.cost = self.calculate_total_cost(self.sol, self.empty_vehicle_weight, self.distanceMatrix)
        self.ReportSolution(self.sol)
        return self.sol

    def ReportSolution(self, sol):
        output_file_path = "path\\solution_output.txt"
        with open(output_file_path, "w") as file:

            file.write("Cost:\n{:.5f}\n".format(sol.cost))


            file.write("Routes:\n{}\n".format(len(sol.routes)))


            for rt in sol.routes:
                route_str = "0," + ",".join(
                    str(node.ID) for node in rt.sequenceOfNodes[1:])
                file.write(route_str + "\n")
            SolDrawer.draw('Clarke and Wright', self.sol, self.allNodes)


# Example usage
m = Model()
filepath = "path\\Instance.txt"
m.BuildModel(filepath)
s = Solver(m)
sol = s.solve()


#solution checker

import math

class Node:
    def __init__(self, idd, xx, yy, dem=0, st=0):
        self.x = xx
        self.y = yy
        self.ID = idd
        self.isRouted = False
        self.demand = dem


def load_model(Instance):
    all_nodes = []
    all_lines = list(open(Instance, "r"))

    separator = ','

    line_counter = 0

    ln = all_lines[line_counter]
    no_spaces = ln.split(sep=separator)
    capacity = int(no_spaces[1])

    line_counter += 1
    ln = all_lines[line_counter]
    no_spaces = ln.split(sep=separator)
    empty_vehicle_weight = int(no_spaces[1])

    line_counter += 1
    ln = all_lines[line_counter]
    no_spaces = ln.split(sep=separator)
    tot_customers = int(no_spaces[1])

    line_counter += 3
    ln = all_lines[line_counter]

    no_spaces = ln.split(sep=separator)
    x = float(no_spaces[1])
    y = float(no_spaces[2])
    depot = Node(0, x, y)
    all_nodes.append(depot)

    for i in range(tot_customers):
        line_counter += 1
        ln = all_lines[line_counter]
        no_spaces = ln.split(sep=separator)
        idd = int(no_spaces[0])
        x = float(no_spaces[1])
        y = float(no_spaces[2])
        demand = float(no_spaces[3])
        customer = Node(idd, x, y, demand)
        all_nodes.append(customer)

    return all_nodes, capacity, empty_vehicle_weight

def distance(from_node, to_node):
    dx = from_node.x - to_node.x
    dy = from_node.y - to_node.y
    dist = math.sqrt(dx ** 2 + dy ** 2)
    return dist

def calculate_route_details(nodes_sequence, empty_vehicle_weight):
    tot_dem = sum(n.demand for n in nodes_sequence)
    tot_load = empty_vehicle_weight + tot_dem
    tn_km = 0
    for i in range(len(nodes_sequence) - 1):
        from_node = nodes_sequence[i]
        to_node = nodes_sequence[i+1]
        tn_km += distance(from_node, to_node) * tot_load
        tot_load -= to_node.demand
    return tn_km, tot_dem


def test_solution(solution_output, all_nodes, capacity, empty_vehicle_weight):
    all_lines = list(open(solution_output, "r"))
    line = all_lines[1]
    objective_reported = float(line.strip())
    objective_calculated = 0

    times_visited = {}
    for i in range(1, len(all_nodes)):
        times_visited[i] = 0

    line = all_lines[3]
    vehs_used = int(line.strip())

    separator = ','
    line_counter = 4
    for i in range(vehs_used):
        ln = all_lines[line_counter]
        ln = ln.strip()
        no_commas = ln.split(sep=separator)
        ids = [int(no_commas[i]) for i in range(len(no_commas))]
        nodes_sequence = [all_nodes[idd] for idd in ids]
        rt_tn_km, rt_load = calculate_route_details(nodes_sequence, empty_vehicle_weight)
        for nn in range(1,len(nodes_sequence)):
            n_in = nodes_sequence[nn].ID
            times_visited[n_in] = times_visited[n_in] + 1

        if rt_load > capacity:
            print('Capacity violation. Route', i, 'total load is', rt_load)
            return
        objective_calculated += rt_tn_km
        line_counter += 1

    if abs(objective_calculated - objective_reported) > 0.001:
        print('Cost Inconsistency. Cost Reported', objective_reported, '--- Cost Calculated', objective_calculated)
        return


    for t in times_visited:
        if times_visited[t] != 1:
            print('Error: customer', t, 'not present once in the solution')
            return


    print('Solution is ΟΚ. Total Cost:', objective_calculated)



instance_file_path = r"path\Instance.txt"
all_nodes, capacity, empty_vehicle_weight = load_model("path\\Instance.txt")


solution_output_file_path = r"path\\solution_output.txt"
test_solution(solution_output_file_path, all_nodes, capacity, empty_vehicle_weight)


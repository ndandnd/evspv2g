import pandas as pd
import string
import matplotlib.pyplot as plt

charge_cost_premium =  1 + 1e-2 # multiply to charge cost when charging. (so that repeated cycles of charging and discharging do not occur)

def make_locs(n):
    """
    Return the first n uppercase letters as location names:
    1 → ['A']
    2 → ['A','B']
    etc.
    """
    if not 1 <= n <= len(string.ascii_uppercase):
        raise ValueError(f"points must be 1..{len(string.ascii_uppercase)}")
    return list(string.ascii_uppercase[:n])


def plot_net_with_delta(net, delta, time_blocks, solar_mult, mode_name, base_eps, points):
    times = sorted(time_blocks)
    # Scale values to kW (100 kWh per time block becomes 100 kW)
    net_vals = [net.get(t, 0) * 100 for t in times]
    delta_vals = [-delta.get(t, 0) * 100 for t in times]  # Net generation (solar minus demand)
    
    plt.figure()
    plt.bar(times, net_vals, align='center', label='Net (Dis)charge')
    plt.plot(times, delta_vals, marker='o', label='Net Generation', color='red')
    plt.axhline(0, linewidth=0.8)
    plt.xlabel('Hour')
    plt.ylabel('Power (kW)')
    plt.title('Net Generation Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    # Save with descriptive filename
    filename = f"net_generation_s{solar_mult}_m{mode_name}_e{base_eps}_p{points}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


## add c_g
def calculate_truck_route_cost(route, truck_cost, charging_cost_data):
    """
    Cost function for truck routes (covers trips), now using time‐varying prices.
    """
    total = truck_cost

    cs = route.get("charging_stops", {})
    # paid charging: add cost_data[h,t] for every (h,t) in chi_plus
    for (h,t) in cs.get("chi_plus", []):
        total += charging_cost_data.at[t, h] * charge_cost_premium
    # discharging: negative cost 
    for (h,t) in cs.get("chi_minus", []):
        total -= charging_cost_data.at[t, h]
    # (free charging events chi_plus_free incur zero cost)

    return total

def calculate_battery_route_cost(route, batt_cost, charging_cost_data):
    """
    Cost function for battery‐only routes (no trips), with time‐varying prices.
    """
    total = batt_cost

    cs = route.get("charging_stops", {})
    for (h,t) in cs.get("chi_plus", []):
        total += charging_cost_data.at[t, h] * charge_cost_premium
    for (h,t) in cs.get("chi_minus", []):
        total -= charging_cost_data.at[t, h]

    return total

def extract_duals(rmp):
    alpha, beta, gamma = {}, {}, {}

    for c in rmp.getConstrs():
        cname = c.ConstrName
        dual  = c.Pi
        if cname.startswith("trip_coverage_"):
            # Extract the trip index from the constraint name "trip_coverage_<i>"
            i_str = cname.split("_")[-1]
            i_trip = int(i_str)
            alpha[i_trip] = dual
        elif cname.startswith("freecharge_"):
            # Extract time t from "freecharge_<t>"
            t_str = cname.split("_")[-1]
            t_val = int(t_str)
            beta[t_val] = dual
        elif cname.startswith("discharge_"):
            # Extract time t from "discharge_<t>"
            t_str = cname.split("_")[-1]
            t_val = int(t_str)
            gamma[t_val] = dual
    return alpha, beta, gamma

def extract_batt_route_from_solution(model, bar_t, h="h1"):
    """
    Extracts a route from the battery pricing model solution and returns a dictionary 
    in the same format as `R_prime_simple`, compatible with R_batt.

    Assumes battery route starts at depot "O", visits a single station h, and returns to "O".

    Parameters:
        model: Gurobi model after optimization.
        bar_t: Number of time blocks.
        h: The station used in the battery route (default "h1").

    Returns:
        route_dict: a dictionary representing the battery route.
    """
    # chi_plus       = model.getVarByName
    # chi_plus_free  = model.getVarByName
    # chi_minus      = model.getVarByName
    # chi_zero       = model.getVarByName

    time_blocks = list(range(1, bar_t + 1))

    def varX(name, t):
        return model.getVarByName(f"{name}[{t}]").X

    route_dict = {
        "route": ["O", h, "O"],
        "charging_stops": {
            "stations": [h],
            "cst": [],
            "cet": [],
            "chi_plus_free": [],
            "chi_minus_free": [],
            "chi_minus": [],
            "chi_plus": [],
            "chi_zero": []
        },
        "charging_activities": 1,
        "type": "batt"
    }

    cst = None
    cet = None

    for t in time_blocks:
        if varX("chi_plus_free", t) > 0.5:
            route_dict["charging_stops"]["chi_plus_free"].append((h, t))
            if cst is None:
                cst = t
            cet = t
        if varX("chi_plus", t) > 0.5:
            route_dict["charging_stops"]["chi_plus"].append((h, t))
            if cst is None:
                cst = t
            cet = t
        if varX("chi_minus", t) > 0.5:
            route_dict["charging_stops"]["chi_minus"].append((h, t))
            if cst is None:
                cst = t
            cet = t
        if varX("chi_zero", t) > 0.5:
            route_dict["charging_stops"]["chi_zero"].append((h, t))
            if cst is None:
                cst = t
            cet = t
        if varX("chi_minus_free", t) > 0.5:
            route_dict["charging_stops"]["chi_minus_free"].append((h, t))
            if cst is None:
                cst = t
            cet = t


    route_dict["charging_stops"]["cst"].append(cst if cst is not None else 1)
    route_dict["charging_stops"]["cet"].append(cet if cet is not None else bar_t)

    return route_dict

def extract_route_from_solution(vars_dict, T, S, bar_t, depot="O", value_getter=lambda v: v.X):
    """
    Extracts a route from the pricing model solution and returns a route dictionary.

    value_getter: a function var -> value, e.g. v.X or v.Xn for pool solutions.
    """
    # 1) Reconstruct node sequence
    route_nodes = [depot]
    first_node = None
    # check pull-out from depot
    for i in T:
        if value_getter(vars_dict["wA_trip"][i]) > 0.5:
            first_node = i
            break
    if first_node is None:
        for h in S:
            if value_getter(vars_dict["wA_station"][h]) > 0.5:
                first_node = h
                break
    if first_node is None:
        raise RuntimeError("No node found leaving the depot!")
    route_nodes.append(first_node)
    current = first_node
    # walk until return
    while True:
        next_node = None
        if current in T:
            # trip->trip
            for j in T:
                if j != current and (current, j) in vars_dict["x"]:
                    if value_getter(vars_dict["x"][(current,j)]) > 0.5:
                        next_node = j; break
            # trip->station
            if next_node is None:
                for h in S:
                    if value_getter(vars_dict["y"][(current,h)]) > 0.5:
                        next_node = h; break
            # return to depot
            if next_node is None and value_getter(vars_dict["wOmega_trip"][current]) > 0.5:
                next_node = depot
        else:  # station
            for i in T:
                if value_getter(vars_dict["z"][(current,i)]) > 0.5:
                    next_node = i; break
            if next_node is None and value_getter(vars_dict["wOmega_station"][current]) > 0.5:
                next_node = depot
        if next_node is None:
            raise RuntimeError(f"No outgoing arc from {current}")
        route_nodes.append(next_node)
        if next_node == depot:
            break
        current = next_node
    # 2) Build dictionary
    route_dict = {"route": route_nodes,
                "charging_stops": {k:[] for k in ["stations","cst","cet",
                                                    "chi_plus_free","chi_minus_free",
                                                    "chi_minus","chi_plus","chi_zero"]},
                "charging_activities": 0}
    # 3) Record stops
    for node in route_nodes[1:-1]:
        if node in S:
            route_dict["charging_stops"]["stations"].append(node)
            # start/end
            route_dict["charging_stops"]["cst"].append(value_getter(vars_dict["cst"][node]))
            route_dict["charging_stops"]["cet"].append(value_getter(vars_dict["cet"][node]))
            # events
            for t in range(1, bar_t+1):
                if value_getter(vars_dict["chi_plus_free"][(node,t)]) > 0.5:
                    route_dict["charging_stops"]["chi_plus_free"].append((node,t))
                if value_getter(vars_dict["chi_minus_free"][(node,t)]) > 0.5:
                    route_dict["charging_stops"]["chi_minus_free"].append((node,t))
                if value_getter(vars_dict["chi_minus"][(node,t)]) > 0.5:
                    route_dict["charging_stops"]["chi_minus"].append((node,t))
                if value_getter(vars_dict["chi_plus"][(node,t)]) > 0.5:
                    route_dict["charging_stops"]["chi_plus"].append((node,t))
                if value_getter(vars_dict["chi_zero"][(node,t)]) > 0.5:
                    route_dict["charging_stops"]["chi_zero"].append((node,t))
            route_dict["charging_activities"] += 1
    
    # 4) Type (truck or batt) and remaining SOC
    route_dict["type"] = "truck" if any(n in T for n in route_nodes) else "batt"
    route_dict["remaining_soc"] = value_getter(vars_dict["g_return"])
    return route_dict

def generate_trip_data(locs, start_time, end_time):
    records = []
    trip_id = 0
    for t in range(start_time, end_time):
        for i in range(len(locs)):
            for j in range(i+1, len(locs)):
                # forward
                records.append({
                    'Start loc': locs[i],
                    'End loc':   locs[j],
                    'Start time': t,
                    'End time':   t + 2
                })
                # reverse
                records.append({
                    'Start loc': locs[j],
                    'End loc':   locs[i],
                    'Start time': t,
                    'End time':   t + 2
                })
                trip_id += 2
    df = pd.DataFrame(records)
    df.index.name = 'Trip'
    return df

from gurobipy import Model, Column, LinExpr, GRB, quicksum
from utils import calculate_truck_route_cost, calculate_battery_route_cost

def build_master(R_truck,
    R_batt,
    mode,
    T,
    time_blocks,
    delta_dict,
    bus_cost,              
    batt_cost,             
    charging_cost_data,    
    binary=False
):
    """
    mode=0  VSP no charge
    mode=1  EVSP only (no solar, no V2G):  skip both free‐charge & discharge
    mode=2  EVSP+solar       (no V2G):  include free‐charge only
    mode=3  EVSP+solar+V2G                include both free‐charge & discharge
    """

    # Build the new RMP with truck and battery-only routes
    rmp = Model("RMP_truck_batt")

    #  1) Decision Variables 
    if binary == False:
        a = rmp.addVars(len(R_truck), vtype=GRB.CONTINUOUS, lb=0, ub=1, name="a")  # relaxed version
        if mode > 0: # mode is not VSP
            b = rmp.addVars(len(R_batt), vtype=GRB.CONTINUOUS, lb=0, name="b")
            route_costs_batt = [calculate_battery_route_cost(rt, bus_cost, charging_cost_data)
                                for rt in R_batt]
        else:
            b = {}
            route_costs_batt = []
        
    else:
        a = rmp.addVars(len(R_truck), vtype=GRB.INTEGER, lb=0, ub=1, name="a")  # relaxed version
        if mode > 0: # mode is not VSP
            b = rmp.addVars(len(R_batt), vtype=GRB.INTEGER, lb=0, name="b")
            route_costs_batt = [calculate_battery_route_cost(rt, bus_cost, charging_cost_data)
                                for rt in R_batt]
        else:
            b = {}
            route_costs_batt = []
        
    
    #  2) Route Costs 
    route_costs_truck = [calculate_truck_route_cost(route, bus_cost, charging_cost_data) for route in R_truck]
    route_costs_batt = [calculate_battery_route_cost(route, bus_cost, charging_cost_data) for route in R_batt]

    #  3) Objective Function 
    expr = quicksum(a[r]*route_costs_truck[r] for r in range(len(R_truck)))
    
    if mode > 0: # if mode is not VSP
        expr += quicksum(b[r]*route_costs_batt[r] for r in range(len(R_batt)))
    rmp.setObjective(expr, GRB.MINIMIZE)


    #  4) Trip Coverage Constraints 
    for i in T:
        covering_truck_routes = [
            r_idx for r_idx, route in enumerate(R_truck)
            if i in route["route"]
        ]
        rmp.addConstr(
            quicksum(a[r] for r in covering_truck_routes) >= 1,
            name=f"trip_coverage_{i}"
        )
    
    # 5) free‐charge (solar)
    if mode >= 3: # allow v2v and solar
        for t in time_blocks:
            rhs_free = max(0, -delta_dict[t])
            truck_contrib = quicksum(
                a[r] * (
                    sum(1 for (_,tt) in R_truck[r]["charging_stops"].get("chi_plus_free", [])  if tt==t)
                - sum(1 for (_,tt) in R_truck[r]["charging_stops"].get("chi_minus_free", []) if tt==t)
                )
                for r in range(len(R_truck))
            )
            batt_contrib = quicksum(
                b[r] * (
                    sum(1 for (_,tt) in R_batt[r]["charging_stops"].get("chi_plus_free", [])  if tt==t)
                - sum(1 for (_,tt) in R_batt[r]["charging_stops"].get("chi_minus_free", []) if tt==t)
                )
                for r in range(len(R_batt))
            )
            rmp.addConstr(truck_contrib + batt_contrib <= rhs_free, # delete
                        name=f"freecharge_{t}")
    elif mode == 2: # allow solar but not v2v
        for t in time_blocks:
            rhs_free = max(0, -delta_dict[t])
            truck_contrib = quicksum(
                a[r] * (
                    sum(1 for (_,tt) in R_truck[r]["charging_stops"].get("chi_plus_free", [])  if tt==t)
                #- sum(1 for (_,tt) in R_truck[r]["charging_stops"].get("chi_minus_free", []) if tt==t)
                )
                for r in range(len(R_truck))
            )
            batt_contrib = quicksum(
                b[r] * (
                    sum(1 for (_,tt) in R_batt[r]["charging_stops"].get("chi_plus_free", [])  if tt==t)
                #- sum(1 for (_,tt) in R_batt[r]["charging_stops"].get("chi_minus_free", []) if tt==t)
                )
                for r in range(len(R_batt))
            )
            rmp.addConstr(truck_contrib + batt_contrib <= rhs_free, # delete
                        name=f"freecharge_{t}")
    # 6) discharge (only for V2G, i.e. mode == 3)
    if mode == 3:
        for t in time_blocks:
            rhs_discharge = max(0, delta_dict[t])
            truck_discharge_expr = quicksum(
                a[r] * sum(1 for (_,tt) in R_truck[r]["charging_stops"].get("chi_minus", []) if tt==t)
                for r in range(len(R_truck))
            )
            batt_discharge_expr = quicksum(
                b[r] * sum(1 for (_,tt) in R_batt[r]["charging_stops"].get("chi_minus", []) if tt==t)
                for r in range(len(R_batt))
            )
            rmp.addConstr(-truck_discharge_expr - batt_discharge_expr >= -rhs_discharge,
                        name=f"discharge_{t}")

    return rmp, a, b

def init_master(
    R_truck,
    R_batt,
    mode,
    T,
    time_blocks,
    delta_dict,
    bus_cost,              
    batt_cost,             
    charging_cost_data,    
    binary=False
):

    """
    Build the *persistent* RMP.  Return:
    - rmp         : the Gurobi model
    - a, b        : dicts mapping route‐indices → Var
    - trip_cov    : dict mapping trip i → coverage constraint
    - freechg     : dict mapping time t → free‐charge constraint (if any)
    - discharge   : dict mapping time t → discharge constraint (if any)
    """
    rmp = Model("RMP_truck_batt")

    # 1) Trip‐coverage constraints
    trip_cov = {}
    for i in T:
        trip_cov[i] = rmp.addConstr(
            LinExpr() >= 1,
            name=f"trip_coverage_{i}"
        )

    # 2) Solar / V2V free‐charge constraints
    freechg = {}
    if mode >= 2:
        for t in time_blocks:
            rhs = max(0, -delta_dict[t])
            freechg[t] = rmp.addConstr(
                LinExpr() <= rhs,
                name=f"freecharge_{t}"
            )

    # 3) V2G discharge constraints
    discharge = {}
    if mode == 3:
        for t in time_blocks:
            rhs = max(0, delta_dict[t])
            # note: we will add columns with negative coefficients so that
            #   -truck_discharge - batt_discharge >= -rhs
            # becomes LinExpr(var-coefs) >= -rhs
            discharge[t] = rmp.addConstr(
                LinExpr() >= -rhs,
                name=f"discharge_{t}"
            )

    # 4) Build the initial columns for R_truck
    a = {}
    for idx, route in enumerate(R_truck):
        cost = calculate_truck_route_cost(route, bus_cost, charging_cost_data)
        col  = Column()
        # coverage
        for i in route["route"]:
            if i in T:
                col.addTerms(1.0, trip_cov[i])
        # free‐charge contributions
        if mode >= 2:
            for t in time_blocks:
                plus_free  = sum(1 for (_,tt) in route["charging_stops"].get("chi_plus_free",  []) if tt==t)
                minus_free = sum(1 for (_,tt) in route["charging_stops"].get("chi_minus_free", []) if tt==t)
                if (plus_free-minus_free) != 0:
                    col.addTerms(plus_free-minus_free, freechg[t])
        # discharge contributions
        if mode == 3:
            for t in time_blocks:
                minus = sum(1 for (_,tt) in route["charging_stops"].get("chi_minus", []) if tt==t)
                if minus:
                    # recall constraint is LinExpr >= -rhs,
                    # so we add (−minus)·var to that LinExpr
                    col.addTerms(-minus, discharge[t])

        vtype = GRB.INTEGER if binary else GRB.CONTINUOUS
        a[idx] = rmp.addVar(
            obj=cost,
            lb=0, ub=1,
            vtype=vtype,
            column=col,
            name=f"a[{idx}]"
        )

    # 5) Build the initial columns for R_batt (if mode>0)
    b = {}
    if mode > 0:
        for idx, route in enumerate(R_batt):
            cost = calculate_battery_route_cost(route, batt_cost, charging_cost_data)
            col  = Column()
            # no trip‐coverage for battery
            # but free‐charge & discharge just like above:
            if mode >= 2:
                for t in time_blocks:
                    pf = sum(1 for (_,tt) in route["charging_stops"].get("chi_plus_free",  []) if tt==t)
                    mf = sum(1 for (_,tt) in route["charging_stops"].get("chi_minus_free", []) if tt==t)
                    if (pf-mf)!=0:
                        col.addTerms(pf-mf, freechg[t])
            if mode == 3:
                for t in time_blocks:
                    m = sum(1 for (_,tt) in route["charging_stops"].get("chi_minus", []) if tt==t)
                    if m:
                        col.addTerms(-m, discharge[t])

            b[idx] = rmp.addVar(
                obj=cost,
                lb=0, ub=1,
                vtype=vtype,
                column=col,
                name=f"b[{idx}]"
            )

    # 6) update and solve
    rmp.update()
    rmp.modelSense = GRB.MINIMIZE
    return rmp, a, b, trip_cov, freechg, discharge

def solve_master(R_truck,
    R_batt,
    mode,
    T,
    time_blocks,
    delta_dict,
    bus_cost,              
    batt_cost,             
    charging_cost_data,    
    binary=False):
    
    rmp, a, b = build_master(R_truck,
    R_batt,
    mode,
    T,
    time_blocks,
    delta_dict,
    bus_cost,              
    batt_cost,             
    charging_cost_data,    
    binary=False
    )
    rmp.optimize()
    return rmp, a, b

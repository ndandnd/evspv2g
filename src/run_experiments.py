
#%%
import pandas as pd
import numpy as np
from gurobipy import *
from gurobipy import Model, Column, LinExpr, GRB
import time
import matplotlib.pyplot as plt
import os

from config  import n_fast_cols, n_exact_cols, time_blocks, tolerance, bar_t, charge_mult, factor, charge_cost_premium

from utils   import make_locs, generate_trip_data, extract_route_from_solution, extract_batt_route_from_solution, extract_duals, calculate_truck_route_cost, calculate_battery_route_cost

from master  import init_master, solve_master, build_master

#%%


def create_pricing_variables(model, T, S, bar_t, G):

    vars_dict = {}

    #  pull‐out / pull‐in
    vars_dict["wA_trip"]   = model.addVars(T, vtype=GRB.BINARY, name="wA_trip")
    vars_dict["wOmega_trip"] = model.addVars(T, vtype=GRB.BINARY, name="wOmega_trip")

    vars_dict["wA_station"]   = model.addVars(S, vtype=GRB.BINARY, name="wA")
    vars_dict["wOmega_station"] = model.addVars(S, vtype=GRB.BINARY, name="wOmega")

    vars_dict["x"] = model.addVars(
        T, T,
        vtype=GRB.BINARY,
        name="x",
        ub={(i,i):0 for i in T}
    )
    vars_dict["y"] = model.addVars(T, S, vtype=GRB.BINARY, name="y")
    vars_dict["z"] = model.addVars(S, T, vtype=GRB.BINARY, name="z")

    # timing cst/cet
    vars_dict["cst"] = model.addVars(
        S,
        lb=0, ub=bar_t,
        vtype=GRB.INTEGER,
        name="cst"
    )
    vars_dict["cet"] = model.addVars(
        S,
        lb=0, ub=bar_t,
        vtype=GRB.INTEGER,
        name="cet"
    )

    # charge-mode binaries over (h,t)
    vars_dict["charge"]        = model.addVars(S, time_blocks, vtype=GRB.BINARY, name="charge")
    vars_dict["chi_plus"]      = model.addVars(S, time_blocks, vtype=GRB.BINARY, name="chi_plus")
    vars_dict["chi_plus_free"] = model.addVars(S, time_blocks, vtype=GRB.BINARY, name="chi_plus_free")
    vars_dict["chi_minus"]     = model.addVars(S, time_blocks, vtype=GRB.BINARY, name="chi_minus")
    vars_dict["chi_minus_free"]= model.addVars(S, time_blocks, vtype=GRB.BINARY, name="chi_minus_free")
    vars_dict["chi_zero"]      = model.addVars(S, time_blocks, vtype=GRB.BINARY, name="chi_zero")

    #  number of charges
    vars_dict["bar_l"] = model.addVar(vtype=GRB.INTEGER, name="bar_l")

    # net charge v[h]
    vars_dict["v"] = { h: model.addVar(lb=-G, ub=G, vtype=GRB.INTEGER, name=f"v_{h}") for h in S }

    #  SoC g[i]
    vars_dict["g"] = {}
    # for i in (T + S + ["O"]):
    #     vars_dict["g"][i] = model.addVar(lb=0, ub=G, vtype=GRB.CONTINUOUS, name=f"g_{i}")
    vars_dict["g"] = model.addVars(
        T + S + ["O"],
        lb=0, ub=G,
        vtype=GRB.CONTINUOUS,
        name="g"
    )
    # # final g
    vars_dict["g_return"] = model.addVar(lb=0, ub=G, vtype=GRB.CONTINUOUS, name="g_return")

    return vars_dict


def build_pricing(alpha, beta, gamma, mode):
    HT = [(h,t) for h in S for t in time_blocks]           

    pricing_model = Model("EV_Routing")
    pricing_model.setParam(GRB.Param.MIPFocus, 2)  
   

    vars_dict = create_pricing_variables(pricing_model,
                                        T,
                                        S,
                                        bar_t=bar_t,
                                        G=G)


    wA_trip       = vars_dict["wA_trip"]
    wOmega_trip   = vars_dict["wOmega_trip"]
    wA_station    = vars_dict["wA_station"]
    wOmega_station= vars_dict["wOmega_station"]
    x             = vars_dict["x"]
    y             = vars_dict["y"]
    z             = vars_dict["z"]
    cst           = vars_dict["cst"]
    cet           = vars_dict["cet"]
    chi_plus      = vars_dict["chi_plus"]
    chi_plus_free = vars_dict["chi_plus_free"]
    chi_minus     = vars_dict["chi_minus"]
    chi_minus_free= vars_dict["chi_minus_free"]
    chi_zero      = vars_dict["chi_zero"]
    charge        = vars_dict["charge"]
    bar_l         = vars_dict["bar_l"]
    v             = vars_dict["v"]
    g             = vars_dict["g"]
    g_return     = vars_dict["g_return"]

    cs = pricing_model.addVars(S, time_blocks, vtype=GRB.BINARY, name="cs")  # cs[h,t]
    gcharge = pricing_model.addVars(S, time_blocks, vtype=GRB.CONTINUOUS, lb=0, ub=G, name="gcharge")


    # pull-out:
    pullout = quicksum(wA_station[h] for h in S_l[1]) \
                + quicksum(wA_trip[i] for i in T)
    # pull-in:
    pullin = quicksum(wOmega_station[h] for h in S) \
                + quicksum(wOmega_trip[i] for i in T)
    
    pricing_model.addConstr(pullout == pullin, name="start_end_balance")
    # equal 1
    pricing_model.addConstr(pullout == 1, name="start_end_once_value")
    # (equivalently, pullin == 1)

    # pull-out==0 for all h in layers 2…L
    pricing_model.addConstrs(
        (wA_station[h] == 0
        for l in range(2, L+1) for h in S_l[l]),
        name="pullout_charge"
    )

    # at most one charge per layer
    pricing_model.addConstrs(
        ( quicksum(wA_station[h] for h in S_l[l])
        + quicksum(y[(i,h)] for i in T for h in S_l[l]) <= 1
        for l in range(1, L+1) ),
        name="one_charge_per_layer"
    )

    # sequential layer
    pricing_model.addConstrs(
        ( quicksum(wOmega_station[h] for h in S_l[l+1])
        + quicksum(z[(h,i)] for h in S_l[l+1] for i in T)
        <=
        quicksum(wOmega_station[h] for h in S_l[l])
        + quicksum(z[(h,i)] for h in S_l[l]   for i in T)
        for l in range(1, L) ),
        name="sequential_layer"
    )

    pricing_model.addConstrs(
                ( wA_station[h]
                    + quicksum(y[(i,h)] for i in T)
                    ==
                    wOmega_station[h]
                    + quicksum(z[(h,i)] for i in T)
                    for h in S ),
                name="flow_charge_balance"
                )


    ### Num Charges
    lhs_in  = quicksum(wA_station[h] + quicksum(y[(i,h)] for i in T) for h in S)
    lhs_out = quicksum(wOmega_station[h] + quicksum(z[(h,i)] for i in T) for h in S)

    pricing_model.addConstr(bar_l == lhs_in,  name="num_charge_in")
    pricing_model.addConstr(bar_l == lhs_out, name="num_charge_out")



    pricing_model.addConstrs((cst[h] >= 0    for h in S), name="cst_lb")
    pricing_model.addConstrs((cst[h] <= bar_t for h in S), name="cst_ub")
    pricing_model.addConstrs((cet[h] >= 0    for h in S), name="cet_lb")
    pricing_model.addConstrs((cet[h] <= bar_t for h in S), name="cet_ub")
    pricing_model.addConstrs(
    ( cst[h]
        >= wA_station[h]*tau[("O",h)]
        + quicksum(y[(i,h)]*(et[i]+tau[(i,h)]) for i in T)
        for h in S ),
    name="cst_def"
    )

    pricing_model.addConstrs(
    ( cet[h]
        <= wOmega_station[h]*(bar_t - tau[(h,"O")])
        + quicksum(z[(h,i)]*(st[i] - tau[(h,i)]) for i in T)
        for h in S ),
    name="cet_def"
    )

    pricing_model.addConstrs((cst[h] <= cet[h] for h in S),
                            name="charge_start_end")

    for h,t in HT:
        pricing_model.addGenConstrIndicator(
            charge[h,t],                    # the binary var
            1,                               # value it takes
            cst[h] - t,                          # left‐hand var
            GRB.LESS_EQUAL,                  # sense
            0,                               # right‐hand constant
            name=f"ind_charge_cst_{h}_{t}"
        )
        pricing_model.addGenConstrIndicator(
            charge[h,t],                    # the binary var
            1,                               # value it takes
            cet[h] - t,                          # left‐hand var
            GRB.GREATER_EQUAL,              # sense
            0,                               # right‐hand constant
            name=f"ind_charge_cet_{h}_{t}"
        )

        #  Enforce cs[h, t] = 1 iff t == cst[h] 
        pricing_model.addGenConstrIndicator(
            cs[h, t],  # the binary variable
            1,         # value it takes
            cst[h] - t,  # left-hand side
            GRB.EQUAL,  # sense
            0,         # right-hand constant
            name=f"ind_cs_{h}_{t}"
        )
        pricing_model.addGenConstrIndicator(
            cs[h, t],  # the binary variable
            1,         # value it takes
            gcharge[h, t] - g[h],  # left-hand side
            GRB.EQUAL,  # sense
            0,         # right-hand constant
            name=f"ind_gcharge_{h}_{t}"
        )
        pricing_model.addSOS(
            GRB.SOS_TYPE1,
            [ chi_plus[(h,t)],
            chi_plus_free[(h,t)],
            chi_zero[(h,t)],
            chi_minus[(h,t)],
            chi_minus_free[(h,t)] ]
        )

    if mode == 0:
        pricing_model.addConstrs(
            ( y[(i,h)] + z[(h,i)] + wA_station[h] + wOmega_station[h] == 0
            for h in S for i in T ),
            name="no_visit_charge"
            )
        pricing_model.addConstrs(
            ( chi_zero[(h,t)] == charge[(h,t)]
            for h in S for t in time_blocks ),
            name="mode0_chi_zero"
            )
        pricing_model.addConstrs(
            ( chi_plus[(h,t)] + chi_plus_free[(h,t)]
            + chi_minus[(h,t)] + chi_minus_free[(h,t)] == 0
            for h in S for t in time_blocks ),
            name="mode0_no_other"
            )
    elif mode == 1:
        # EVSP only: only paid charging or idle
        pricing_model.addConstrs(
            ( chi_plus[(h,t)] + chi_zero[(h,t)] == charge[(h,t)]
            for h in S for t in time_blocks),
            name="mode1_charge_modes"
            )
        pricing_model.addConstrs(
            ( chi_plus_free[(h,t)]
            + chi_minus[(h,t)] + chi_minus_free[(h,t)] == 0
            for h in S for t in time_blocks ),
            name="mode1_no_other"
            )
    elif mode == 2:
        # EVSP+solar: paid + free charge + idle
        pricing_model.addConstrs(
            ( chi_plus[(h,t)] + chi_zero[(h,t)] + chi_plus_free[(h,t)] == charge[(h,t)]
            for h in S for t in time_blocks),
            name="mode2_charge_modes"
            )
        pricing_model.addConstrs(
            ( chi_minus[(h,t)] + chi_minus_free[(h,t)]       == 0
            for h in S for t in time_blocks ),
            name="mode2_no_discharge"
            )
    elif mode == 3:
        # EVSP+solar+V2G: paid + free charge + idle + discharge
        pricing_model.addConstrs(
            ( chi_plus[(h,t)] + chi_zero[(h,t)] + chi_plus_free[(h,t)]
            + chi_minus[(h,t)] + chi_minus_free[(h,t)] == charge[(h,t)]
            for h in S for t in time_blocks),
            name="mode3_charge_modes"
            )


    ## Flow trip
    pricing_model.addConstrs(
        ( wA_trip[i]
            + quicksum(x[(j,i)] for j in T if j!=i)
            + quicksum(z[(h,i)] for h in S_l[l])
            ==
            wOmega_trip[i]
            + quicksum(x[(i,j)] for j in T if j!=i)
            + (quicksum(y[(i,h)] for h in S_l[l+1]) if l < L else 0)
            for l in range(1, L+1) for i in T ),
        name="flow_trip"
        )


    ### AMOUNT CHARGED

    pricing_model.addConstrs(
    ( v[h]
        == charge_mult * quicksum(
            chi_plus_free[(h,t)] + chi_plus[(h,t)]
        - chi_minus[(h,t)] - chi_minus_free[(h,t)]
        for t in time_blocks)
        for h in S ),
    name="amt_charged"
    )



    # g["O"] = G
    pricing_model.addConstr(g["O"] == G, name="initial_soc")

    pricing_model.addConstrs(
    ( g[h] + v[h] <= G   for h in S ), name="soc_sum_ub"
    )
    pricing_model.addConstrs(
    ( g[h] + v[h] >= 0   for h in S ), name="soc_sum_lb"
    )
    # # 5) Enough i->O


    ## main h in S ##
    # Enough h -> O
    for h in S:
        pricing_model.addGenConstrIndicator(
            wOmega_station[h],
            1,
            g[h] + v[h] - d[(h,"O")],
            GRB.GREATER_EQUAL,
            0,
            name=f"ind_suff_stat2O_{h}"
        )
        # final SOC
        pricing_model.addGenConstrIndicator(
            wOmega_station[h],           # the binary var
            1,                         # value it takes
            g_return - (g[h] + v[h] - d[(h,"O")]),  # left‐hand var
            GRB.EQUAL,         # sense
            0,                         # right‐hand constant
            name=f"ind_final_soc_stat_{h}"
        )

    # SOC update origin to h
        pricing_model.addGenConstrIndicator(
            wA_station[h],             # the binary var
            1,                         # value it takes
            g[h] - (g["O"] - d[("O", h)]),  # left‐hand var
            GRB.EQUAL,                 # sense
            0,                         # right‐hand constant
            name=f"ind_soc_depot2station_{h}"
        )

        pricing_model.addConstr(
            quicksum(cs[h, t] for t in time_blocks) == wA_station[h] \
                    + quicksum(y[(i,h)] for i in T),
            name=f"only_one_cs_{h}"
        )

        for t in time_blocks[:-1]:
            pricing_model.addConstr(
                gcharge[h, t + 1] == gcharge[h, t] +
                    chi_plus[h, t] + chi_plus_free[h, t] - chi_minus[h, t] - chi_minus_free[h,t],
                name=f"soc_update_gcharge_{h}_{t}"
            )
        pricing_model.addSOS(
            GRB.SOS_TYPE1,
            [ cs[h,t] for t in time_blocks ]
        )

    ## main i in T ##
    for i in T:
        ## soc update origin to i##
        pricing_model.addGenConstrIndicator(
            wA_trip[i],             # the binary var
            1,                       # value it takes
            g[i] - (g["O"] - d[("O", i)]),  # left‐hand var
            GRB.EQUAL,               # sense
            0,                       # right‐hand constant
            name=f"ind_soc_depot2trip_{i}"
        )

        # 5) Enough i->O
        pricing_model.addGenConstrIndicator(
            wOmega_trip[i],           # the binary var
            1,                         # value it takes
            g[i] - (d[(i,"O")] + epsilon[i]),  # left‐hand var
            GRB.GREATER_EQUAL,         # sense
            0,                         # right‐hand constant
            name=f"ind_suff_trip2O_{i}"
        )
        # final SOC
        pricing_model.addGenConstrIndicator(
            wOmega_trip[i],           # the binary var
            1,                         # value it takes
            g_return - (g[i] - d[(i,"O")]),  # left‐hand var
            GRB.EQUAL,         # sense
            0,                         # right‐hand constant
            name=f"ind_final_soc_trip_{i}"
        )
        # SoC update i->j
        for j in T:
            if i != j:
                ## SoC update trip to trip ##
                pricing_model.addGenConstrIndicator(
                    x[(i,j)],             # binary arc‐selection var
                    1,                     # value it takes
                    g[j] - (g[i] - epsilon[i] - d[(i,j)]),  # left‐hand var
                    GRB.EQUAL,             # sense
                    0,                     # right‐hand constant
                    name=f"ind_soc_trip2trip_{i}_{j}"
                )

                ## time sequencing ##
                pricing_model.addGenConstrIndicator(
                    x[(i,j)],             # binary arc‐selection var
                    1,
                    et[i] + tau[(i,j)] - (st[j]),                  
                    GRB.LESS_EQUAL,
                    0,
                    name=f"ind_time_{i}_{j}"
                )
        
        for h in S:
            # SoC update i->h
            pricing_model.addGenConstrIndicator(
                y[(i,h)],             # binary arc‐selection var
                1,                     # value it takes
                g[h] - (g[i] - epsilon[i] - d[(i,h)]),  # left‐hand var
                GRB.EQUAL,             # sense
                0,                     # right‐hand constant
                name=f"ind_soc_trip2station_{i}_{h}"
            )
            # SoC update h->i
            pricing_model.addGenConstrIndicator(
                z[(h,i)],
                1,                     # value it takes
                g[i] - (g[h] + v[h] - d[(h,i)]),  # left‐hand var
                GRB.EQUAL,             # sense
                0,                     # right‐hand constant
                name=f"ind_soc_station2trip_{h}_{i}"
            )

    obj_expr = 0

    obj_expr += bus_cost
    for h,t in HT:
        if t in charging_cost_data.index and h in charging_cost_data.columns:
            charging_cost = charging_cost_data.loc[t, h]  # c_{h,t}
            # Add charging cost contribution
            obj_expr += charging_cost * chi_plus.get((h, t), 0) * charge_cost_premium
            obj_expr -= charging_cost * chi_minus.get((h, t), 0)

    # The rest of the objective follows:
    # Subtract alpha[i] times (wA_i + sum_j x[j,i] + sum_h z[h,i])
    for i in T:
        alpha_val = alpha.get(i, 0)  # Default to 0 if alpha[i] is not present
        trip_expr = (
            wA_trip[i]
            + quicksum(x[(j, i)] for j in T if j != i)
            + quicksum(z[(h, i)] for h in S)
        )
        obj_expr -= alpha_val * trip_expr

    # Subtract beta[t] * chi_plus_free[h,t] across all stations h
    for t, beta_val in beta.items():
        free_charge_expr = quicksum(
            chi_plus_free[(h, t)] - chi_minus_free[(h,t)] for h in S if (h, t) in chi_plus_free
        )
        obj_expr -= beta_val * free_charge_expr

    # Subtract gamma[t] * (chi_minus[h,t] - chi_plus[h,t]) across all stations h
    for t, gamma_val in gamma.items():
        discharge_net_expr = quicksum(
            - chi_minus.get((h, t), 0)  for h in S
        )
        obj_expr -= gamma_val * discharge_net_expr

    pricing_model.setObjective(obj_expr, GRB.MINIMIZE)
    return pricing_model, vars_dict

def solve_pricing_fast(alpha, beta, gamma, mode, num_fast_cols = 10):
    m, vars_dict = build_pricing(alpha, beta, gamma, mode)
    m.Params.PoolSearchMode = 1 # random search direction for solutions
    m.Params.PoolSolutions  = num_fast_cols
    m.Params.SolutionLimit  = num_fast_cols

    m.optimize()
    return m, vars_dict

def solve_pricing_exact(alpha, beta, gamma, mode, num_exact_cols = 10):
    m, vars_dict = build_pricing(alpha, beta, gamma, mode)
    m.Params.PoolSearchMode = 2 # systematic search for the n best solutions
    m.Params.PoolSolutions  = num_exact_cols
    return m, vars_dict

#### new pricing model, with batt routes
def build_battery_pricing(beta, gamma, mode):
    """
    mode=1: EVSP only (no free‐charge, no discharge)
    mode=2: EVSP+solar (free‐charge OK, but no discharge)
    mode=3: EVSP+solar+V2G (everything OK)
    """
    from gurobipy import Model, GRB

    m = Model("Battery_Pricing")
    m.setParam(GRB.Param.MIPFocus, 2)

    # unpack parameters for clarity



    # variables
    vars_batt = {}
    vars_batt["g_batt"]        = m.addVars(time_blocks, lb=0, ub=G, vtype=GRB.CONTINUOUS, name="g")
    vars_batt["chi_plus"]      = m.addVars(time_blocks, vtype=GRB.BINARY, name="chi_plus")
    vars_batt["chi_plus_free"] = m.addVars(time_blocks, vtype=GRB.BINARY, name="chi_plus_free")
    vars_batt["chi_minus"]     = m.addVars(time_blocks, vtype=GRB.BINARY, name="chi_minus")
    vars_batt["chi_minus_free"]= m.addVars(time_blocks, vtype=GRB.BINARY, name="chi_minus_free")
    vars_batt["chi_zero"]      = m.addVars(time_blocks, vtype=GRB.BINARY, name="chi_zero")

    # bjective
    expr = batt_cost
    for t in time_blocks:
        price   = charging_cost_data.at[t, "h1"]
        expr   += price * (vars_batt["chi_plus"][t] * charge_cost_premium
                        - vars_batt["chi_minus"][t])
        expr   -= beta.get(t, 0) * (vars_batt["chi_plus_free"][t]
                                - vars_batt["chi_minus_free"][t])
        expr   -= gamma.get(t, 0) * (- vars_batt["chi_minus"][t])

    m.setObjective(expr, GRB.MINIMIZE)

    # constraints
    # 1 initial SoC
    m.addConstr(vars_batt["g_batt"][1] == G, name="initial_soc")

    # 3 SoC update
    for t in time_blocks[:-1]:
        m.addConstr(
            vars_batt["g_batt"][t+1]
            == vars_batt["g_batt"][t]
            + vars_batt["chi_plus"][t]
            + vars_batt["chi_plus_free"][t]
            - vars_batt["chi_minus"][t]
            - vars_batt["chi_minus_free"][t],
            name=f"soc_update_{t}"
        )

    # only one can happen.
    for t in time_blocks:
        cp  = vars_batt["chi_plus"][t]
        cpf = vars_batt["chi_plus_free"][t]
        cm  = vars_batt["chi_minus"][t]
        cmf = vars_batt["chi_minus_free"][t]
        cz  = vars_batt["chi_zero"][t]

        if mode == 1:
            # only paid charging or idle
            m.addConstr( cp + cz == 1, name=f"mode1_onehot_{t}" )
            # force everything else off
            m.addConstr(cpf == 0, name=f"mode1_no_free_{t}")
            m.addConstr(cm  == 0, name=f"mode1_no_discharge_{t}")
            m.addConstr(cmf == 0, name=f"mode1_no_dischargefree_{t}")

        elif mode == 2:
            # paid + free charge + idle + free discharge
            m.addConstr(cp + cpf + cz + cmf == 1, name=f"mode2_onehot_{t}")
            # but no “true” discharge
            m.addConstr(cm == 0, name=f"mode2_no_discharge_{t}")

        else:  # mode == 3
            # allow everything
            m.addConstr(cp + cpf + cm + cmf + cz == 1, name=f"mode3_onehot_{t}")

    return m, vars_batt
def solve_battery_pricing_fast(beta, gamma, mode, num_fast_cols=10):
    m, vars_batt = build_battery_pricing(beta, gamma, mode)
    # stop as soon as we find up to num_fast_cols feasible columns
    m.Params.PoolSearchMode = 1
    m.Params.PoolSolutions  = num_fast_cols
    m.Params.SolutionLimit   = num_fast_cols
    m.optimize()
    return m, vars_batt

def solve_battery_pricing_exact(beta, gamma, mode, num_exact_cols=10):
    m, vars_batt = build_battery_pricing(beta, gamma, mode)
    m.Params.PoolSearchMode = 2
    m.Params.PoolSolutions  = num_exact_cols
    return m, vars_batt



### modes ###
VSP = 0           # no charge
EVSP        = 1   # no free‐charge, no discharge
EVSP_SOLAR   = 2   # free‐charge OK, but no v2g discharge
EVSP_V2G = 3   # full model


csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'delta.csv')

df = pd.read_csv(csv_path, header=1)





master_times = []
pricing_times = []
iteration_times = []





# Keep only rows with time ending in ':00' (on the hour)
df_hourly = df[df["Time"].str.endswith(":00")].copy()

# Keep only 01:00 to 24:00
df_hourly = df_hourly[df_hourly["Time"].between("01:00", "24:00")]

df_hourly.reset_index(drop=True, inplace=True)
delta_dict = {}
delta_dict = {i+1: df_hourly.loc[i, "Delta_int"] for i in range(len(df_hourly))}



results = []
modes = [3]
epsilons      = [2.5]
points_list   = [2]    # number of locations
solar_multipliers = [7]  # 7 was the default, scaled for the data.
for solar_mult in solar_multipliers:
    df_hourly['Delta_mod']   = (df_hourly['Power (MW)'] / 5 - df_hourly['PV(MW)']) * solar_mult
    df_hourly['Delta_int']   = df_hourly['Delta_mod'].round().astype(int)
    
    delta_dict = {i+1: int(df_hourly.loc[i,'Delta_int'])
                  for i in range(len(df_hourly))}
    for mode in modes:
        selected_mode = mode
        mode_names = {
            0: "VSP",
            1: "EVSP",
            2: "EVSP_SOLAR",
            3: "EVSP_V2G"
        }
        mode_name = mode_names[selected_mode]
        for base_eps in epsilons:

            epsilon_const = base_eps * (10 if selected_mode==0 else 1)
            

            for points in points_list:
                
                print(f"Running solar_mult={solar_mult}, mode={mode_name}, ε={base_eps}, points={points}")

                locs = make_locs(points)
                morning = generate_trip_data(locs, 4, 9)   # 5–10 inclusive
                evening = generate_trip_data(locs, 18, 23)  # 19–22 inclusive
                trip_data = pd.concat([morning, evening], ignore_index=True)
                trip_data.index.name = 'Trip'
                
                #trip_data = generate_trip_data_solar_skip(locs, 8, 18, delta_dict)  # trips from 1 to 24
                epsilon = {i: epsilon_const for i in trip_data.index}

                time_begin = time.time()


                # PARAMETERS


                L = 4              # up to L charging activities
                if selected_mode == 0:
                    G = 125             # battery capacity
                    
                else:
                    G = 7             # battery capacity
                    # epsilon_const =  # energy consumed per trip


                ## case 3
                fuel_cost = 5
                bus_cost = fuel_cost * G + 10        # fixed cost per bus
                batt_discount = 9    # subtract cost from bus_cost
                batt_cost = bus_cost - batt_discount
                

                sl = trip_data["Start loc"].to_dict()  # sl[i] = 'A'/'B'
                el = trip_data["End loc"].to_dict()    # el[i] = 'B'/'A'
                st = trip_data["Start time"].to_dict() # st[i] = 4 or 12
                et = trip_data["End time"].to_dict()   # et[i] = 8 or 16

                T = list(range(len(trip_data)))  # [0..19]




                base_stations = ["h"] # , "k"]

                S_l = { l:[f"{s}{l}" for s in base_stations]
                        for l in range(1, L+1) }

                S = [h for layer in S_l.values() for h in layer]

                Nodes = ["O"] + T + S


                
                # LOCATION COORDINATES
                
                coordinates = {
                    "O": (0, 0),     # Depot
                    "A": (.25, .25),
                    "B": (-.25, .25),
                    "C": (.25, -.25),
                    "D": (-.25, -.25),
                    "h1": (0, 0), # We will have one charging station, at same location as depot. 
                    "h2": (0, 0),
                    "h3": (0, 0),
                    "h4": (0, 0),
                    "h5": (0, 0),
                    "h6": (0, 0),
                    # "k1": (0, -1),
                    # "k2": (0, -1),
                }

                # Simple Manhattan distance function
                def manhattan_distance(c1, c2):
                    return abs(c1[0]-c2[0]) + abs(c1[1]-c2[1])

                # Build distance dictionary d[(i,j)]
                d = {}
                all_keys = list(coordinates.keys())
                for i in all_keys:
                    for j in all_keys:
                        d[(i,j)] = manhattan_distance(coordinates[i], coordinates[j])
                # trip "i" as a location from sl[i]->el[i].
                # “O”-> "i" means the distance from O->sl[i]
                for i in T:
                    d[("O", i)] = manhattan_distance(coordinates["O"], coordinates[sl[i]])
                    d[(i, "O")] = manhattan_distance(coordinates[el[i]], coordinates["O"])
                    # station vs. trip
                    for h in S: #,"k1","k2"]:
                        d[(i, h)] = manhattan_distance(coordinates[el[i]], coordinates[h])
                        d[(h, i)] = manhattan_distance(coordinates[h], coordinates[sl[i]])
                    # trip vs. trip
                    for j in T:
                        if i != j:
                            # If the end loc of i matches the start loc of j, distance=0, else normal
                            if el[i] == sl[j]:
                                d[(i, j)] = 0
                            else:
                                d[(i, j)] = manhattan_distance(coordinates[el[i]], coordinates[sl[j]])



                speed = 1    # e.g. truck goes 10 "distance‐units" per time‐block
                tau = {}      # travel‐time
                for (i,j), dist in d.items():
                    tau[(i,j)] = dist / speed

                cost_data_rows = []
                for t in time_blocks:
                    # repeat “1” once per station in S
                    cost_data_rows.append([fuel_cost]*len(S))

                charging_cost_data = pd.DataFrame(
                    data=cost_data_rows,
                    index=time_blocks,
                    columns=S
                )


                # 
                # R' simple = 4 routes: O->0->O, O->1->O, O->2->O, O->3->O
                # Initialize separate route lists
                R_truck = []
                R_batt = []

                # R_prime_simple
                # Create truck routes covering trips (existing)
                for i in T:
                    # total “distance” cost in SoC units for O→i→O
                    g_return = d[("O", i)] + d[(i, "O")]
                    route_dict = {
                        "route": ["O", i, "O"],
                        "charging_stops": {
                            "stations": [],
                            "cst": [],
                            "cet": [],
                            "chi_plus_free": [],
                            "chi_minus": [],
                            "chi_minus_free": [],
                            "chi_plus": [],
                            "chi_zero": [],
                        },
                        "charging_activities": 0,
                        "type": "truck",
                        "remaining_soc": G - g_return
                    }
                    R_truck.append(route_dict)

                if selected_mode > 0:
                    # Create battery-only routes (no trips) for charging needs (defined by delta_dict)
                    for t in time_blocks:
                        if delta_dict[t] < 0:
                            # Free charging required at time t
                            for _ in range(-delta_dict[t]):  
                                route_dict = {
                                    "route": ["O", "h1", "O"],
                                    "charging_stops": {
                                        "stations": ["h1"],
                                        "cst": [t],
                                        "cet": [t],
                                        "chi_plus_free": [("h1", t)],
                                        "chi_minus": [],
                                        "chi_minus_free": [],
                                        "chi_plus": [],
                                        "chi_zero": []
                                    },
                                    "charging_activities": 1,
                                    "type": "batt"
                                }
                                R_batt.append(route_dict)

                # Verify initial route counts
                print("Initial Truck Routes (R_truck):", len(R_truck))
                print("Initial Battery-only Routes (R_batt):", len(R_batt))
                # record how many of each we started with
                init_truck = len(R_truck)
                init_batt  = len(R_batt)


                # COLGEN LOOP
                iteration = 0
                new_pricing_obj = -1.0   # force first pass
                new_batt_obj    = 0
                max_iter = 333

                ## initial run
                rmp, a, b, trip_cov, freechg, discharge = init_master(
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
                )

                rmp.optimize()


                pricing_mode_used = [] # fast or exact

                truck_added = []
                batt_added  = []

                while (new_pricing_obj < -tolerance) or (new_batt_obj < -tolerance):
                    rmp.reset()
                    iteration += 1
                    print(f"\n--- Iteration {iteration} ---")

                    if iteration > max_iter:
                        print("Reached iteration limit, stopping.")
                        break

                    # 1) solve master
                    master_start = time.time()
                    rmp.optimize()
                    print(" Master obj:", rmp.ObjVal)
                    master_times.append(time.time() - master_start)

                    # 2) extract duals
                    alpha, beta, gamma = extract_duals(rmp)

                    #
                    # 3) TRUCK‐PRICING (fast then exact)
                    #
                    pricing_start = time.time()
                    pricing_model, vars_pr = solve_pricing_fast(
                        alpha, beta, gamma,
                        mode=selected_mode,
                        num_fast_cols=n_fast_cols
                    )
                    pricing_times.append(time.time() - pricing_start)

                    # collect improving truck routes
                    new_trucks = []
                    new_pricing_obj = float('inf')
                    for sol in range(pricing_model.SolCount):
                        pricing_model.Params.SolutionNumber = sol
                        rc = pricing_model.PoolObjVal
                        new_pricing_obj = min(new_pricing_obj, rc)
                        if rc < -tolerance:
                            truck = extract_route_from_solution(
                                vars_pr, T, S, bar_t,
                                value_getter=lambda v: v.Xn
                            )
                            if truck not in R_truck and truck not in new_trucks:
                                new_trucks.append(truck)
                    truck_added.append(len(new_trucks))

                    # if no fast columns, try exact
                    if not new_trucks and new_pricing_obj < -tolerance:
                        print("No fast truck cols → running exact pricing")
                        pricing_model, vars_pr = solve_pricing_exact(
                            alpha, beta, gamma,
                            mode=selected_mode,
                            num_exact_cols=n_exact_cols
                        )
                        pricing_model.optimize()

                        for sol in range(pricing_model.SolCount):
                            pricing_model.Params.SolutionNumber = sol
                            rc = pricing_model.PoolObjVal
                            new_pricing_obj = min(new_pricing_obj, rc)
                            if rc < -tolerance:
                                truck = extract_route_from_solution(
                                    vars_pr, T, S, bar_t,
                                    value_getter=lambda v: v.Xn
                                )
                                if truck not in R_truck and truck not in new_trucks:
                                    new_trucks.append(truck)

                    # add truck columns
                    for route in new_trucks:
                       
                        cost = calculate_truck_route_cost(route, bus_cost, charging_cost_data)
                        col = Column()
                        for i in route["route"]:
                            if i in T:
                                col.addTerms(1.0, trip_cov[i])
                        if selected_mode >= 2:
                            for t in time_blocks:
                                pf = sum(1 for (_,tt) in route["charging_stops"].get("chi_plus_free", [])  if tt==t)
                                mf = sum(1 for (_,tt) in route["charging_stops"].get("chi_minus_free", []) if tt==t)
                                delta = pf - mf
                                if delta:
                                    col.addTerms(delta, freechg[t])
                        if selected_mode == 3:
                            for t in time_blocks:
                                m = sum(1 for (_,tt) in route["charging_stops"].get("chi_minus", []) if tt==t)
                                if m:
                                    col.addTerms(-m, discharge[t])
                        idx = len(a)
                        a[idx] = rmp.addVar(obj=cost, lb=0, ub=1,
                                            vtype=GRB.CONTINUOUS,
                                            column=col, name=f"a[{idx}]")
                        R_truck.append(route)

                        if route["charging_stops"].get("chi_minus_free"):
                            variant = {
                                **route,
                                "charging_stops": {
                                    **route["charging_stops"],
                                    "chi_minus_free": []
                                }
                            }
                            cost_var = calculate_truck_route_cost(variant, bus_cost, charging_cost_data)
                            col_var = Column()
                            for i in variant["route"]:
                                if i in T:
                                    col_var.addTerms(1.0, trip_cov[i])
                            if selected_mode >= 2:
                                for t in time_blocks:
                                    pf = sum(1 for (_,tt) in variant["charging_stops"].get("chi_plus_free", []) if tt==t)
                                    if pf:
                                        col_var.addTerms(pf, freechg[t])
                            if selected_mode == 3:
                                for t in time_blocks:
                                    m = sum(1 for (_,tt) in variant["charging_stops"].get("chi_minus", []) if tt==t)
                                    if m:
                                        col_var.addTerms(-m, discharge[t])
                            idx2 = len(a)
                            a[idx2] = rmp.addVar(obj=cost_var, lb=0, ub=1,
                                                vtype=GRB.CONTINUOUS,
                                                column=col_var, name=f"a[{idx2}]")
                            R_truck.append(variant)


                    if selected_mode > 0:
                            
                        #
                        # 4) BATTERY‐PRICING
                        #
                        batt_model, vars_batt = solve_battery_pricing_fast(
                            beta, gamma, selected_mode,
                            num_fast_cols=n_fast_cols
                        )

                        new_batt_obj = float('inf')
                        new_batts = []
                        for sol in range(batt_model.SolCount):
                            batt_model.Params.SolutionNumber = sol
                            rc_b = batt_model.PoolObjVal
                            new_batt_obj = min(new_batt_obj, rc_b)
                            if rc_b < -tolerance:
                                batt_route = extract_batt_route_from_solution(batt_model, bar_t, h="h1")
                                if batt_route not in R_batt:
                                    new_batts.append(batt_route)

                        batt_added.append(len(new_batts))

                        
                        if not new_batts and new_batt_obj < -tolerance:
                            print("No fast batt cols → running exact batt pricing")
                            batt_model, vars_batt = solve_battery_pricing_exact(
                                beta, gamma, selected_mode,
                                num_exact_cols=n_exact_cols
                            )
                            batt_model.optimize()
                            for sol in range(batt_model.SolCount):
                                batt_model.Params.SolutionNumber = sol
                                rc_b = batt_model.PoolObjVal
                                new_batt_obj = min(new_batt_obj, rc_b)
                                if rc_b < -tolerance:
                                    batt_route = extract_batt_route_from_solution(batt_model, bar_t, h="h1")
                                    if batt_route not in R_batt:
                                        new_batts.append(batt_route)

                        # add battery columns
                        for route in new_batts:
                            
                            cost_b = calculate_battery_route_cost(route, batt_cost, charging_cost_data)
                            col_b = Column()
                            if selected_mode >= 2:
                                for t in time_blocks:
                                    pf = sum(1 for (_,tt) in route["charging_stops"]["chi_plus_free"]  if tt==t)
                                    mf = sum(1 for (_,tt) in route["charging_stops"]["chi_minus_free"] if tt==t)
                                    delta = pf - mf
                                    if delta:
                                        col_b.addTerms(delta, freechg[t])
                            if selected_mode == 3:
                                for t in time_blocks:
                                    m = sum(1 for (_,tt) in route["charging_stops"]["chi_minus"] if tt==t)
                                    if m:
                                        col_b.addTerms(-m, discharge[t])
                            idx_b = len(b)
                            b[idx_b] = rmp.addVar(obj=cost_b, lb=0, ub=1,
                                                vtype=GRB.CONTINUOUS,
                                                column=col_b, name=f"b[{idx_b}]")
                            R_batt.append(route)

                # Final LP


                rmp_lp, a_lp, b_lp = solve_master(R_truck,
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
                final_LP_obj = rmp_lp.ObjVal

                # Final check MIP


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
                for idx, var in a.items():
                    var.start = a_lp[idx].X
                for idx, var in b.items():
                    var.start = b_lp[idx].X
                rmp.Params.LPWarmStart = 2

                if selected_mode == 3:
                    rmp.Params.MIPGap      = 0.01
                else:
                    rmp.Params.MIPGap      = 0.01

                rmp.setParam("MIPFocus", 3)      # aggressive heuristics
                rmp.setParam("Heuristics", 0.5)  # increase heuristic effort
                rmp.setParam("Cuts", 2)          # more cutting‐plane generation
                rmp.setParam("Presolve", 0)      # aggressive presolve
                rmp.optimize()
                final_MIP_obj = rmp.ObjVal



                for r in range(len(R_truck)):
                    if a[r].X == 1:
                        print(f"Route {r} selected: a[{r}] = {a[r].X}")
                        print(R_truck[r])
                        print("-" * 40)

                if selected_mode > 0: # only if not vsp
                    for r in range(len(R_batt)):
                        if b[r].X == 1:
                            print(f"Route {r} selected: b[{r}] = {b[r].X}")
                            print(R_batt[r])
                            print("-" * 40)


                if selected_mode == 0: # only for VSP we calculate the fuel used
                        
                    total_fuel_used = 0.0

                    for r, var in a.items():
                        if var.X == 1:
                            route = R_truck[r]
                           
                            rem_soc   = route.get("remaining_soc")
                            if rem_soc is None:
                                raise KeyError(f"Route {r} has no remaining_soc")
                            fuel_used = G - rem_soc
                            total_fuel_used += fuel_used
                            print(f"Route {r}: fuel used = {fuel_used:.1f} kWh (returned SOC {rem_soc:.1f})")

                    print(f"\nTotal fuel used by all buses: {total_fuel_used:.1f} kWh")

                else: # for EVSP, we calculate charging

                    # 1) per‐route net energy (paid charge minus discharge)
                    net_truck = 0.0
                    for r, var in a.items():
                        if var.X == 1:
                            cs = R_truck[r]["charging_stops"]
                            plus  = len(cs.get("chi_plus", []))
                            minus = len(cs.get("chi_minus", []))
                            net_truck += plus - minus

                    net_batt = 0.0
                    for r, var in b.items():
                        if var.X == 1:
                            cs = R_batt[r]["charging_stops"]
                            plus  = len(cs.get("chi_plus", []))
                            minus = len(cs.get("chi_minus", []))
                            net_batt += plus - minus


                    print(f"net paid‐charge on truck routes:  {net_truck:.0f} kWh")
                    print(f"net paid‐charge on batt routes:   {net_batt:.0f} kWh")
                    print(f"total net paid‐charge (fleet):    {net_truck+net_batt:.0f} kWh")

                # 
                time_end   = time.time()
                time_taken = time_end - time_begin
                print(f"Total time: {time_taken:.1f} s")

                new_truck_num = len(R_truck) - init_truck
                new_batt_num  = len(R_batt)  - init_batt

                print(f"New truck‑routes generated:   {new_truck_num}")
                print(f"New battery‑routes generated: {new_batt_num}")

                # 


                # 
                print(f"Average Master Problem Time per Iteration: {np.average(master_times):.2f} seconds")

                print(f"Average Pricing Problem Time per Iteration: {np.average(pricing_times):.2f} seconds")

                print(f"Average Iteration Time: {np.average(iteration_times):.2f} seconds")

                selected_routes = []
                for r in range(len(R_truck)):
                    if a[r].X > tolerance:
                        selected_routes.append(R_truck[r])
                for r in range(len(R_batt)):
                    if b[r].X > tolerance:
                        selected_routes.append(R_batt[r])


                print(" Master obj:", rmp.ObjVal)

                selected_routes = []
                route_labels   = []

                for r, var in a.items():
                    if var.X > tolerance:
                        selected_routes.append(R_truck[r])
                        route_labels.append(f"Truck[{r}]")


                # Battery routes (b[r] can be >1)
                for r, var in b.items():
                    count = int(round(var.X))
                    for k in range(count):
                        selected_routes.append(R_batt[r])
                        route_labels.append(f"Batt[{r}] - {k+1}")



                #   Gantt 
                times = sorted(time_blocks)
                n = len(selected_routes)
                height = 0.8

                fig, ax = plt.subplots(figsize=(10, n*0.4))
                for idx, route in enumerate(selected_routes):
                    cs = route['charging_stops']
                    for _, t in cs.get('chi_plus', []):
                        ax.barh(idx, 1, left=t, height=height, color='red',    alpha=0.7)
                    for _, t in cs.get('chi_plus_free', []):
                        ax.barh(idx, 1, left=t, height=height, color='green', alpha=0.7)
                    for _, t in cs.get('chi_minus_free', []):
                        ax.barh(idx, 1, left=t, height=height, color='cyan',   alpha=0.7)
                    for _, t in cs.get('chi_minus', []):
                        ax.barh(idx, 1, left=t, height=height, color='blue',   alpha=0.7)

                ax.set_yticks(range(len(route_labels)))
                ax.set_yticklabels(route_labels)
                ax.set_xticks(times)
                ax.set_xlabel("Time Block")
                ax.set_title(f"Charging/Discharging Timeline by Route ({mode_name})")
                legend_handles = [
                    plt.Rectangle((0,0),1,1,color=c,alpha=0.7,label=l)
                    for c,l in [('red','Paid charge'),
                                ('green','Free charge'),
                                ('cyan','V2V discharge'),
                                ('blue','V2G discharge')]
                ]
                ax.legend(handles=legend_handles, bbox_to_anchor=(1.02,1), loc='upper left')
                plt.grid(axis='x', linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.show()

                import matplotlib.pyplot as plt

                iters = list(range(len(master_times)))

                exact_idx = [i for i,m in enumerate(pricing_mode_used) if m=='exact']

                plt.figure(figsize=(10,6))
                plt.scatter(iters, master_times, marker='o', label='Master Time', color='C0')

                plt.scatter(iters, pricing_times, marker='o', label='Pricing (fast)', color='C1')

                plt.scatter(exact_idx, [pricing_times[i] for i in exact_idx],
                            marker='o', label='Pricing (exact)', color='C2')

                plt.xlabel("Iteration")
                plt.ylabel("Time (s)")
                plt.title(f"{bar_t} Timeblocks and {len(trip_data)} Trips")
                plt.legend()
                plt.grid(True)
                plt.show()




                print("Final LP, MIP obj:", final_LP_obj, final_MIP_obj)

                tolerance = 1e-6
                used_trucks = [r for r, var in a.items() if var.X > 0.5]   
                num_used = len(used_trucks)
                print(f"Number of truck routes used: {num_used}")
                print("  routes:", used_trucks)

                if selected_mode == 0:
                    fuel = total_fuel_used
                else:
                    fuel = (net_truck + net_batt) * 100 / 33 # convert kWh to fuel equivalent

                results.append({
                    'solar_mult' : solar_mult,
                    'mode'       : mode_name,
                    'epsilon'    : epsilon_const,
                    'points'     : points,
                    'locs'       : locs,
                    'lp_obj'     : final_LP_obj,
                    'mip_obj'    : final_MIP_obj,
                    'fuel_used'  : fuel,
                    'run_time_s' : time_taken,
                    'trucks'     : num_used
                })


df = pd.DataFrame(results)

points_levels = sorted(df['points'].unique())

metrics = ['trucks', 'fuel_used']
ylabels = {'trucks': 'Number of Vehicles', 
           'fuel_used': 'Net Energy (kWh)'}

for metric in metrics:
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=len(points_levels), 
        figsize=(5*len(points_levels), 4), 
        sharey=True
    )
    if len(points_levels)==1:
        axes = [axes]
    for ax, pts in zip(axes, points_levels):
        sub = df[df['points']==pts]
        for eps in sorted(sub['epsilon'].unique()):
            grp = sub[sub['epsilon']==eps]
            ax.plot(
                grp['solar_mult'], 
                grp[metric], 
                marker='o', 
                linestyle='-',
                label=f'ε={eps}'
            )
        ax.set_title(f'{pts} Locations')
        ax.set_xlabel('Solar Multiplier')
        ax.set_ylabel(ylabels[metric])
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(title='ε')
    fig.suptitle(ylabels[metric], fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
# %%

from min_delay_charging.benders_charge_scheduling import solve_full_problem_gurobi, solve_with_benders, \
    build_subproblem
from min_delay_charging.heuristic_charge_scheduling import repeat_heuristic
from gurobipy import GRB
import logging
from datetime import datetime, timedelta
import time
import copy
import numpy as np


def build_simple_case(route_recov, service_hrs=6):
    # Build up example
    # Route information
    routes = ['A', 'B', 'C']
    # Distance in miles
    route_dist = {'A': 15, 'B': 25, 'C': 15}
    # Double distance and halve number of trips to test easier instance
    # (edited from original)
    route_dist = {k: route_dist[k]*2 for k in route_dist}
    # Time to complete in minutes
    route_time = {'A': 40, 'B': 90, 'C': 45}
    route_time = {r: timedelta(minutes=route_time[r]) for r in route_time}
    # Recovery time in minutes
    route_recov = {r: timedelta(minutes=route_recov[r]) for r in route_recov}
    # Headway in minutes
    route_hw = {'A': 30, 'B': 30, 'C': 60}
    route_hw = {r: timedelta(minutes=route_hw[r]) for r in route_hw}
    # Terminals of route
    route_terms = {'B': ('W', 'E'), 'A': ('S', 'N'), 'C': ('S', 'E')}
    # Create candidate charging sites and associated data
    all_locs = ['W', 'NW', 'N', 'NE', 'E', 'S']
    chg_sites = ['W', 'NW', 'S']
    chg_coords = {'W': (-12.5, 0), 'N': (0, 7.5), 'E': (12.5, 0), 'S': (0, -7.5),
                  'NW': (-10, 1), 'NE': (10, 1)}
    # Calculate charging distance (Euclidean)
    chg_dists = {(a, b): np.sqrt((chg_coords[a][0] - chg_coords[b][0])**2
                              + (chg_coords[a][1] - chg_coords[b][1])**2)
                 for a in all_locs for b in all_locs}
    coord_dists = {(chg_coords[a], chg_coords[b]):
                       np.sqrt((chg_coords[a][0] - chg_coords[b][0]) ** 2
                            + (chg_coords[a][1] - chg_coords[b][1]) ** 2)
                   for (a, b) in chg_dists}
    dh_data = {
        k: {'distance': coord_dists[k],
            'duration': coord_dists[k] * 60/25}
        for k in coord_dists}

    # Start time for first block on each route
    start_dt = datetime(2021, 8, 25, 7)
    # End time for last block on each route
    # (edited from original)
    end_dt = datetime(2021, 8, 25, 7 + service_hrs)

    # Create blocks, trips, and associated data
    blocks = list()
    trip_start_times = dict()
    trip_end_times = dict()
    trip_dists = dict()
    end_chg_dists = dict()
    start_chg_dists = dict()
    for r in routes:
        block_idx = 0
        new_block_start = start_dt
        # Create new routes until
        while new_block_start < start_dt + route_time[r] + route_recov[r]:
            block_idx += 1
            v_f = r + 'F' + str(block_idx)
            v_r = r + 'R' + str(block_idx)
            blocks.append(v_f)
            blocks.append(v_r)
            # Create trips for this block
            t_idx = 0
            # Add in trips to/from depot at end/start of day. These can just
            # be set to zero for the simple case, but we need a 0 index for
            # the model to work properly.
            trip_dists[v_f, 0] = 0
            trip_dists[v_r, 0] = 0
            block_start_min = (new_block_start - start_dt).total_seconds() / 60
            trip_start_times[v_f, 0] = block_start_min
            trip_start_times[v_r, 0] = block_start_min
            trip_end_times[v_f, 0] = block_start_min
            trip_end_times[v_r, 0] = block_start_min
            for s in chg_sites:
                end_chg_dists[v_f, 0, s] = 100
                end_chg_dists[v_r, 0, s] = 100
                start_chg_dists[v_f, 1, s] = 100
                start_chg_dists[v_r, 1, s] = 100

            # end_idx = t_idx + 1
            # trip_dists[v, end_idx] = gmap_last_trip['distance']
            # # Maintain penalty for late arrival to depot
            # trip_start_times[v, end_idx] = trip_end_times[v, end_idx - 1]
            # trip_end_times[v, end_idx] = trip_end_times[v, end_idx - 1]

            trip_start = new_block_start
            while trip_start < end_dt:
                t_idx += 1
                # Set start and end times
                for v in [v_f, v_r]:
                    trip_start_times[v, t_idx] = \
                        (trip_start - start_dt).total_seconds() / 60
                    trip_end_times[v, t_idx] = \
                        (trip_start + route_time[r] - start_dt).total_seconds() / 60
                    # Set distance
                    trip_dists[v, t_idx] = route_dist[r]

                # Set location of current trip end, next trip start
                end_loc_f = route_terms[r][t_idx % 2]
                end_loc_r = route_terms[r][(t_idx - 1) % 2]
                for s in chg_sites:
                    end_chg_dists[v_f, t_idx, s] = chg_dists[end_loc_f, s]
                    end_chg_dists[v_r, t_idx, s] = chg_dists[end_loc_r, s]
                    start_chg_dists[v_f, t_idx+1, s] = chg_dists[s, end_loc_f]
                    start_chg_dists[v_r, t_idx+1, s] = chg_dists[s, end_loc_r]
                # Set next trip start time
                trip_start = trip_start + route_time[r] + route_recov[r]
            # Create a new block by adding the headway to the first start
            new_block_start += route_hw[r]
    logging.info('Blocks: {}'.format(blocks))
    logging.info('Number of blocks: {}'.format(len(blocks)))
    veh_trip_pairs = list(trip_start_times.keys())
    # Assume no deadhead between trips
    inter_trip_times = {vt: 0 for vt in veh_trip_pairs}
    inter_trip_dists = {vt: 0 for vt in veh_trip_pairs}
    # Assume all trips take 3 kWh/mile
    energy_rates = {vt: 3 for vt in veh_trip_pairs}
    # Assume 25 mph to drive to/from chargers
    end_chg_times = {vts: end_chg_dists[vts] * 60 / 25 for vts in end_chg_dists}
    start_chg_times = {
        vts: start_chg_dists[vts] * 60 / 25 for vts in start_chg_dists}

    # Set site parameters
    good_sites = ['N', 'E', 'S', 'W']
    site_costs = {s: 5e5 if s in good_sites else 5e4 for s in chg_sites}
    # site_costs = {s: 400 if s in good_sites else 200 for s in chg_sites}
    charger_costs = {s: 698447 for s in chg_sites}
    max_ch = {s: 4 if s in good_sites else 4 for s in chg_sites}
    # max_ch = {s: 20 for s in chg_sites}
    chg_power = {s: 300/60 for s in chg_sites}

    # Set charge limits
    ch_lims = {v: 400 for v in blocks}

    opt_kwargs = dict(
        vehicles=blocks, veh_trip_pairs=veh_trip_pairs, chg_sites=chg_sites,
        chg_lims=ch_lims, trip_start_times=trip_start_times, max_chargers=max_ch,
        trip_end_times=trip_end_times, trip_dists=trip_dists,
        inter_trip_dists=inter_trip_dists, inter_trip_times=inter_trip_times,
        trip_start_chg_dists=start_chg_dists, trip_end_chg_dists=end_chg_dists,
        chg_rates=chg_power, site_costs=site_costs, charger_costs=charger_costs,
        trip_start_chg_times=start_chg_times, trip_end_chg_times=end_chg_times,
        energy_rates=energy_rates, zero_time=start_dt,
        depot_coords=chg_coords['S'])

    return opt_kwargs


def get_recharge_opt_params(rho_kw, service_hrs=6):
    """
    Set up simple case so it is suitable for one of our recharge
    planning algorithms.

    :param rho_kw: dict of charger power outputs in kW
    :param service_hrs: number of hours each BEB should be in service
        for consecutively
    :return: dict of instance parameters
    """
    rho = rho_kw / 60
    u_max = 400
    ocl_dict = build_simple_case(
        route_recov={'A': 20, 'B': 30, 'C': 60},
        service_hrs=service_hrs
    )

    # Exclude route B
    buses = [v for v in ocl_dict['vehicles'] if v[0] != 'B']
    # Filter down trips
    trips = [(v, t) for (v, t) in ocl_dict['veh_trip_pairs'] if v in buses]
    print('Number of buses: {}'.format(len(buses)))
    print('Number of trips: {}'.format(len(trips)))

    delta = {
        (v, t): ocl_dict['energy_rates'][v, t] * ocl_dict[
            'trip_dists'][v, t]
        for (v, t) in trips
    }
    sigma = {
        (v, t): ocl_dict['trip_start_times'][v, t] for (v, t) in trips
    }
    max_chg_time = dict()
    tau = {(v, t): ocl_dict['trip_end_times'][v, t]
                   - ocl_dict['trip_start_times'][v, t]
           for (v, t) in trips}
    for v in buses:
        t_v = sorted(tt for (vv, tt) in trips if vv == v)
        if v[1] == 'F':
            # Forward block starts at Charger 1
            for ix in range(int(len(t_v))):
                if ix % 2 == 0:
                    max_chg_time['Charger 1', v, ix] = 0
                    max_chg_time['Charger 2', v, ix] = u_max / rho
                else:
                    # Charging is available at charger 2, but not 1
                    max_chg_time['Charger 2', v, ix] = 0
                    max_chg_time['Charger 1', v, ix] = u_max / rho

        else:
            # Reverse block starts at Charger 2
            for ix in range(int(len(t_v))):
                if ix % 2 == 1:
                    max_chg_time['Charger 1', v, ix] = 0
                    max_chg_time['Charger 2', v, ix] = u_max / rho
                else:
                    max_chg_time['Charger 2', v, ix] = 0
                    max_chg_time['Charger 1', v, ix] = u_max / rho

    return {
        'trips': trips,
        'delta': delta,
        'sigma': sigma,
        'tau': tau,
        'rho': {'Charger 1': rho, 'Charger 2': rho},
        'max_chg_time': max_chg_time,
        'u_max': u_max
    }


def run_simple_case(rho_kw, run_direct_solve, run_benders):
    input_dict = get_recharge_opt_params(rho_kw=rho_kw, service_hrs=12)

    trips = input_dict['trips']
    sigma = input_dict['sigma']
    tau = input_dict['tau']
    delta = input_dict['delta']
    rho = input_dict['rho']
    max_chg_time = input_dict['max_chg_time']
    u_max = input_dict['u_max']
    chargers = ['Charger 1']

    # Try solving complete model
    if run_direct_solve:
        full_params = {'TimeLimit': 3600}

        opt_x, full_arcs = solve_full_problem_gurobi(
            trips=trips,
            chargers=chargers,
            sigma=sigma,
            tau=tau,
            delta=delta,
            chg_pwr=rho,
            max_chg_time=max_chg_time,
            max_chg=u_max,
            u0=u_max,
            delay_cutoff=1e12,
            gurobi_params=full_params
        )

        chg_opps = [
            (c, i, j) for c in chargers for (i, j) in trips
            if max_chg_time[c, i, j] > 0
        ]
        # Which trips don't have charging?
        skip_trips = [
            k for k in chg_opps if k not in opt_x
        ]

        # Build the subproblem
        sp_m = build_subproblem(
            trips_skip=skip_trips,
            chargers=chargers,
            arcs_chg=full_arcs,
            trips=trips,
            delta=delta,
            chg_pwr=rho,
            sigma=sigma,
            tau=tau,
            max_chg_time=max_chg_time,
            max_chg=u_max,
            u0=u_max
        )
        sp_m.optimize()
        print('** Subproblem validation **')
        if sp_m.status == GRB.OPTIMAL:
            print('Subproblem objective value: {:.2f}'.format(sp_m.ObjVal))
        elif sp_m.status == GRB.INFEASIBLE:
            print('Subproblem was found to be infeasible!')
        else:
            raise ValueError(
                'Unrecognized solver status: {}'.format(sp_m.status))

    soln_dict = repeat_heuristic(
        case_data=input_dict,
        chargers=chargers,
        rho=rho,
        u_max=u_max,
        n_runs=500,
        random_mult=0.5
    )

    if run_benders:
        solve_with_benders(
            case_data=input_dict,
            chargers=chargers,
            rho=rho,
            u_max=u_max,
            heur_solns=soln_dict,
            cut_gap=0.5,
            delay_cutoff=min(soln_dict.values()),
            time_limit=3600
        )


if __name__ == '__main__':
    rho_vals = [300, 350, 400, 450, 500]
    for rho in rho_vals:
        print('--- Charger power: {} kW ---'.format(rho))
        run_simple_case(rho_kw=rho, run_direct_solve=True, run_benders=False)



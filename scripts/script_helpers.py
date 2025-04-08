from min_delay_charging.data import GTFSData, get_updated_osm_data
import logging
import pandas as pd
import numpy as np


def build_trips_df(
    gtfs, date, routes, depot_coords, routes_60, route_method='exclusive',
    add_depot_dh=True, add_trip_dh=False, add_kwh_per_mi=False,
    add_durations=False, rng=None
):
    # Initialize GTFS
    day_trips = gtfs.get_trips_from_date(date)

    # Determine which blocks have trips exclusively on these routes
    gb = day_trips.groupby('block_id')
    route_blocks = list()
    for block_id, subdf in gb:
        if route_method == 'exclusive':
            include_block = all(subdf['route_short_name'].isin(routes))

        elif route_method == 'inclusive':
            include_block = any(subdf['route_short_name'].isin(routes))

        else:
            raise ValueError(
                'route_method must be either "exclusive" or "inclusive"'
            )

        if include_block:
            route_blocks.append(block_id)

    beb_trips = day_trips[
        day_trips['block_id'].isin(route_blocks)
    ]
    # Add all trip data columns (e.g. locations and distances)
    beb_trips = gtfs.add_trip_data(beb_trips, date)
    beb_trips['duration_sched'] = (
            beb_trips['end_time'] - beb_trips['start_time']
        ).dt.total_seconds() / 60

    beb_trips = beb_trips.rename(columns={'route_short_name': 'route'})
    block_types = beb_trips.groupby('block_id')['route'].unique().apply(
        lambda x: any(rt in routes_60 for rt in x)
    ).astype(int).rename('60_dummy')
    beb_trips = beb_trips.merge(
        block_types, left_on='block_id', right_index=True
    )

    if add_trip_dh:
        # Add deadhead to next trip
        beb_trips = GTFSData.add_deadhead(beb_trips)
        block_gb = beb_trips.groupby('block_id')
        dh_dfs = list()
        for block_id, block_df in block_gb:
            block_df = block_df.sort_values(by='trip_idx', ascending=True)

            # Associate DH with the next trip, since we assume charging
            # would happen before DH in charge scheduling approach.
            block_df['dh_dist'] = block_df['dh_dist'].shift(1).fillna(0)
            dh_dfs.append(block_df)
        beb_trips = pd.concat(dh_dfs)

    # Add pull-in and pull-out trip distances (note that scheduling
    # and charger location code handle these differently, and depot DH
    # should only be added with this approach for charge scheduling).
    if add_depot_dh:
        beb_trips = GTFSData.add_depot_deadhead(beb_trips, *depot_coords)

    if add_durations:
        # Add duration and kwh per mi
        beb_trips = add_realtime_durations(
            trips_to_lookup=beb_trips,
            realtime_summary=load_realtime_summary(),
            sim_all=False,
            rng=rng
        )
    if add_kwh_per_mi:
        beb_trips = predict_kwh_per_mi(beb_trips, rng=rng)

    # Optimization expects trips to be indexed from zero
    beb_trips['trip_idx'] -= 1

    return beb_trips


def build_scheduling_inputs(
    beb_trips, chargers_df, u_max, dh_cutoff_dist=1, kwh_per_mi=3
):
    # Create a copy so we don't modify the input
    beb_trips = beb_trips.copy()
    beb_trips['kwh'] = beb_trips['kwh_per_mi'] * (
            beb_trips['total_dist'] + beb_trips['dh_dist']
    )
    block_energy = beb_trips.groupby('block_id')['kwh'].sum()
    opt_blocks = block_energy[block_energy > u_max].index.tolist()

    # only include blocks that need opportunity charging
    beb_trips = beb_trips[beb_trips['block_id'].isin(opt_blocks)].copy()

    # Initialize parameter dicts
    delta = dict()
    sigma = dict()
    tau = dict()
    max_chg_time = dict()

    # Track terminal coords to get dist to chargers
    term_coords = list()

    # Set reference time (midnight on first day observed)
    t_ref = pd.to_datetime(beb_trips['start_time'].dt.date).min()

    # Constants for handling time
    for ix, rw in beb_trips.iterrows():
        dict_ix = (rw['block_id'], rw['trip_idx'])
        start_time = rw['start_time']

        sigma[dict_ix] = (start_time - t_ref).total_seconds() / 60

        # Set energy consumption for this trip
        delta[dict_ix] = kwh_per_mi * (rw['total_dist'] + rw['dh_dist'])
        duration = rw['duration_sched']

        # Set the trip duration parameter based on the supplied method
        tau[dict_ix] = duration

        if (rw['end_lon'], rw['end_lat']) not in term_coords:
            term_coords.append((rw['end_lat'], rw['end_lon']))

    # Set charging upper bound
    # First, get all DH distances
    charger_coords = list(
        zip(
            chargers_df['lat'].tolist(),
            chargers_df['lon'].tolist()
        )
    )
    charger_dh = get_updated_osm_data(term_coords, charger_coords)
    # chargers_df = chargers_df.copy().set_index('name')
    # Use it to set charging limit
    for ix, rw in beb_trips.iterrows():
        for c in chargers_df.index:
            if charger_dh[
                (rw['end_lat'], rw['end_lon']),
                (chargers_df.loc[c, 'lat'], chargers_df.loc[c, 'lon'])
            ]['distance'] < dh_cutoff_dist:
                max_chg_time[c, rw['block_id'], rw['trip_idx']] = \
                    60 * u_max / chargers_df.loc[c, 'kw']

            else:
                max_chg_time[c, rw['block_id'], rw['trip_idx']] = 0

    case_data = {
        'sigma': sigma,
        'tau': tau,
        'delta': delta,
        'max_chg_time': max_chg_time
    }

    return case_data




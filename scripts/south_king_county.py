from min_delay_charging.data import GTFSData
from datetime import datetime
import pandas as pd
from min_delay_charging.heuristic_charge_scheduling import repeat_heuristic
from min_delay_charging.benders_charge_scheduling import solve_with_benders
from scripts.script_helpers import build_trips_df, build_scheduling_inputs
from numpy.random import default_rng
from min_delay_charging.vis import plot_trips_and_terminals, plot_charger_timelines
from pathlib import Path


def set_up_king_cty_case(
        gtfs_dir, date_in, depot_coords, locs_df, routes, u_max, charger_kw,
        kwh_per_mi=3., show_map=False
):

    # Initialize GTFS
    gtfs = GTFSData.from_dir(dir_name=gtfs_dir)

    beb_trips = build_trips_df(
        gtfs=gtfs,
        date=date_in,
        routes=routes,
        routes_60=[],
        depot_coords=depot_coords,
        add_depot_dh=True,
        add_trip_dh=True
    )

    if show_map:
        inst_map = plot_trips_and_terminals(
            trips_df=beb_trips, locs_df=locs_df,
            shapes_df=gtfs.shapes_df)
        config = {
            'toImageButtonOptions': {
                'format': 'png',
                'scale': 3
                # Multiply title/legend/axis/canvas sizes by this factor
            }
        }
        inst_map.show(config=config)

    logging.info(
        'There are {} total trips to be served by {} BEB blocks.'.format(
            len(beb_trips),
            beb_trips['block_id'].nunique()
        )
    )

    # beb_trips = predict_kwh_per_mi(beb_trips)
    beb_trips['kwh_per_mi'] = kwh_per_mi

    blocks_excl = [str(b) for b in [7157686, 7160963, 7157747, 7160999]]
    beb_trips = beb_trips[~beb_trips.isin(blocks_excl)].copy()

    beb_trips['kwh'] = beb_trips['kwh_per_mi'] * (
            beb_trips['total_dist'] + beb_trips['dh_dist']
    )
    total_kwh = beb_trips.groupby('block_id')['kwh'].sum()
    opp_chg_blocks = total_kwh[total_kwh > u_max].index.tolist()
    logging.info(
        '{} blocks must use opportunity charging; {} can use depot charging'
        ' only.'.format(len(opp_chg_blocks),
                       len(total_kwh[total_kwh <= u_max]))
    )
    logging.info(
        'Opportunity charging buses complete {} trips.'.format(
            len(beb_trips[beb_trips['block_id'].isin(opp_chg_blocks)])
        )
    )

    chargers_df = locs_df.rename(columns={'y': 'lat', 'x': 'lon'})
    chargers_df['kw'] = charger_kw

    return build_scheduling_inputs(
        beb_trips=beb_trips,
        chargers_df=chargers_df,
        u_max=u_max,
        kwh_per_mi=kwh_per_mi,
    )


def run_multi_charger_case(
        u_max=300., rho_kw=250, n_runs=100, scenario='a',
        show_map=False, random_mult=1., kwh_per_mi=3.):
    data_dir = Path(__file__).parent.parent / 'data'
    # Read in charger locations
    if scenario == 'a':
        locs_df = pd.read_csv(data_dir / 'kc_sites_a.csv', index_col=0)
    elif scenario == 'b':
        locs_df = pd.read_csv(data_dir / 'kc_sites_b.csv', index_col=0)
    else:
        raise ValueError('Scenario must be either \'a\' or \'b\'')
    chargers = locs_df.index.tolist()

    rho = {c: rho_kw / 60 for c in chargers}
    routes = ['F Line', 'H Line', 131, 132, 150, 153, 161, 165]
    routes = [str(r) for r in routes]
    case_data = set_up_king_cty_case(
        gtfs_dir=data_dir / 'gtfs' / 'metro_mar24',
        date_in=datetime(2024, 4, 3),
        depot_coords=(47.495809, -122.286190),
        locs_df=locs_df,
        routes=routes,
        u_max=u_max,
        charger_kw=rho_kw,
        show_map=show_map,
        kwh_per_mi=kwh_per_mi,
    )
    return repeat_heuristic(
        case_data, chargers, rho, u_max, n_runs, random_mult, return_type='df',
        rng=default_rng(100)

    )


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    # Run Scenario A
    soln_df = run_multi_charger_case(
        rho_kw=220, show_map=True, random_mult=0.5, scenario='a',
        n_runs=500, u_max=525*0.75, kwh_per_mi=3.19
    )
    # Save charging plan
    soln_df.to_csv('commtr_a_soln.csv')
    # Plot charger timelines
    plot_charger_timelines(soln_df, datetime(2024, 4, 3), True, False,
                           'Charger Timeline at {}')

    # Run Scenario B
    soln_df_b = run_multi_charger_case(
        rho_kw=220, show_map=False, random_mult=0.5, scenario='b',
        n_runs=500, u_max=525*0.75, kwh_per_mi=3.19
    )
    # Save charging plan
    soln_df_b.to_csv('commtr_b_soln.csv')
    # Plot charger timelines
    plot_charger_timelines(soln_df_b, datetime(2024, 4, 3), True, False,
                           'Charger Timeline at {}')





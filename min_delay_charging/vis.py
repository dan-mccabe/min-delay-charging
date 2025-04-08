import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from pathlib import Path
import datetime
from min_delay_charging.data import get_shape
from dotenv import dotenv_values, find_dotenv



def plot_trips_and_terminals(
        trips_df: pd.DataFrame, locs_df: pd.DataFrame,
        shapes_df: pd.DataFrame, light_or_dark: str = 'light'):
    # Mapbox API key
    try:
        config = dotenv_values(find_dotenv())
        token = config['MAPBOX_KEY']

    except KeyError:
        raise KeyError(
            'No Openrouteservice key found in .env file. For more information'
            ', please see the project README file.'
        )

    if light_or_dark == 'light':
        text_and_marker_color = 'black'

    elif light_or_dark == 'dark':
        text_and_marker_color = 'white'

    else:
        raise ValueError(f'Unrecognized light_or_dark value: {light_or_dark}')

    px.set_mapbox_access_token(token)

    # Compile terminal counts
    start_cts = trips_df.groupby(
        ['start_lat', 'start_lon']).count()['route_id'].rename('start')
    start_cts.index.set_names(['lat', 'lon'], inplace=True)
    end_cts = trips_df.groupby(
        ['end_lat', 'end_lon']).count()['route_id'].rename('end')
    end_cts.index.set_names(['lat', 'lon'], inplace=True)
    all_cts = pd.merge(
        start_cts, end_cts, left_index=True, right_index=True, how='outer')
    all_cts = all_cts.fillna(0)
    all_cts['total'] = all_cts['start'] + all_cts['end']
    all_cts = all_cts.sort_values(by='total', ascending=False).reset_index()
    all_cts['name'] = ''
    all_cts['symbol'] = 'circle'
    all_cts['size'] = all_cts['total']
    all_cts['label_name'] = [
        '{} trips start here, {} trips end here'.format(
            int(all_cts['start'][i]), int(all_cts['end'][i]))
        for i in range(len(all_cts))]
    all_cts['color'] = 'blue'

    # Charging sites
    if locs_df is not None:
        # locs_df = locs_df.set_index('name')
        locs_df['symbol'] = 'fuel'
        fig = px.scatter_mapbox(
            locs_df, lat='lat', lon='lon', text='label_name', zoom=10,
            size_max=30, hover_data={c: False for c in locs_df.columns})

        fig.update_traces(marker={'size': 10, 'symbol': locs_df['symbol']})
        fig.update_traces(textposition='bottom center', textfont_size=15,
                          textfont_color=text_and_marker_color)

    else:
        fig = go.Figure()

    # Trip terminals
    # Marker size: scale linearly from minimum to maximum
    min_marker = 15
    max_marker = 30
    msize = np.round(min_marker + (max_marker - min_marker) * (
            all_cts['size'] - all_cts['size'].min())/all_cts['size'].max())

    new_trace = go.Scattermapbox(lat=all_cts['lat'], lon=all_cts['lon'],
                                 showlegend=True, hoverinfo='text',
                                 mode='markers', text=all_cts['label_name'],
                                 marker=go.scattermapbox.Marker(
                                     color='rgba(60, 120, 255, 1)',
                                     size=msize),
                                 name='Trip Start/End Locations    ')
    fig.add_trace(new_trace)

    # Trips
    shape_cts = trips_df.groupby('shape_id').count()['route_id'].sort_values()
    for shp in shape_cts.index:
        shape_pts = get_shape(shapes_df, shp)
        shape_pt_df = pd.DataFrame(shape_pts).transpose()
        alpha = 0.2 + 0.5 * shape_cts[shp] / max(shape_cts)
        rgba_str = 'rgba(255, 80, 80, {:.2f})'.format(alpha)

        new_trace = go.Scattermapbox(mode='lines',
                                     lat=shape_pt_df["shape_pt_lat"],
                                     lon=shape_pt_df["shape_pt_lon"],
                                     showlegend=False, hoverinfo='skip',
                                     line={'color': rgba_str, 'width': 2})
        fig.add_trace(new_trace)

    # Trace for legend
    new_trace = go.Scattermapbox(
        mode='lines', lat=shape_pt_df["shape_pt_lat"],
        lon=shape_pt_df["shape_pt_lat"], showlegend=True,
        line={'color': 'rgba(255, 80, 80, 0.9)'},
        name='Passenger Trip   ')
    fig.add_trace(new_trace)

    # Reverse order to put markers on top
    fdata = fig.data
    fig.data = tuple(list(fdata[1:]) + [fdata[0]])
    fig.update_layout(
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=0.98, font={'size': 14},
        ),
        margin=dict(l=0, r=0, t=5, b=0))

    # if light_or_dark == 'dark':
    fig.update_layout(mapbox_style=light_or_dark)

    return fig


def plot_charger_timelines(df, zero_time, show_service=False, highlight=False,
                           title_fmt='{}'):
    df['block_id'] = df['block_id'].astype(str)
    # Define relevant times
    t_start = zero_time + pd.to_timedelta(
        df['start_time'].min(), unit='minute') - datetime.timedelta(hours=1)
    t_end = zero_time + pd.to_timedelta(
        df['end_time'].max(), unit='minute') + datetime.timedelta(hours=1)
    t_srs = pd.date_range(
        t_start, t_end, freq='min').to_series()
    highlight_df = t_srs.between(
        datetime.datetime(2023, 12, 6, 7, 0),
        datetime.datetime(2023, 12, 6, 9, 0)).to_frame(
        name='peak')
    trip_col_labels = {
        'start_time': 'Start Time',
        'end_time': 'End Time',
        'delay': 'Delay'
    }

    df['Delayed'] = df['delay'] > 0
    df['Delayed'] = df['Delayed'].map(
        {True: 'Delayed Trip', False: 'On-Time Trip'})
    df['trip_name'] = 'Block ' + df['block_id'].astype(str) \
                      + ', Trip ' + df['trip_idx'].astype(str)

    fig_list = list()
    for c in df['charger'].unique():
        c_df = df[df['charger'] == c].copy()
        c_df['start_time'] = zero_time + pd.to_timedelta(
            c_df['start_time'] + c_df['delay'], unit='minute')
        c_df['end_time'] = zero_time + pd.to_timedelta(
            c_df['end_time'] + c_df['delay'], unit='minute')
        c_df['plugin_time'] = zero_time + pd.to_timedelta(c_df['plugin_time'],
                                                          unit='minute')
        c_df['finish_chg_time'] = c_df['plugin_time'] + pd.to_timedelta(
            c_df['chg_time'], unit='minute')

        # Make subplots to highlight time block with delays
        fig = make_subplots(specs=[[{'secondary_y': True}]])

        # Highlight delayed period
        if highlight:
            hl_trace = go.Scatter(
                x=highlight_df.index, y=highlight_df['peak'],
                fill='tonexty', fillcolor='rgba(240, 228, 66, 0.7)',
                line_shape='hv', line_color='rgba(0,0,0,0)',
                showlegend=False
            )
            fig.add_trace(hl_trace, row=1, col=1, secondary_y=False)

        else:
            # Add a dummy trace. Skipping this makes the formatting weird because
            # there is then only one y axis used.
            fig.add_trace(
                go.Scatter(
                    x=[t_start], y=[0],
                    fillcolor='rgba(0,0,0,0)', line_color='rgba(0,0,0,0)',
                    showlegend=False, hoverinfo='skip'),
                row=1, col=1, secondary_y=False)

        # Hide highlight axis
        fig.update_xaxes(showgrid=False)
        fig.update_layout(yaxis1_range=[0, 0.5], yaxis1_showgrid=False,
                          yaxis1_showticklabels=False)

        # Only include blocks that charge here
        time_by_block = c_df.groupby('block_id')['chg_time'].sum()
        c_blocks = time_by_block[time_by_block > 0.1].index.tolist()
        c_df = c_df[c_df['block_id'].isin(c_blocks)]

        # Sort blocks by first plugin time
        order_df = c_df[c_df['chg_time'] > 0.1]
        block_order = order_df.sort_values(by='plugin_time', ascending=True)[
            'block_id'].unique().tolist()
        block_order.reverse()
        # Add trip timeline
        if show_service:
            # There is a weird bug in plotly where it only plots one color when using subplots.
            # To get around this we'll add two traces manually.
            on_time_trips = c_df[c_df['Delayed'] == 'On-Time Trip']
            delayed_trips = c_df[c_df['Delayed'] == 'Delayed Trip']

            trip_hover = {
                'Delayed': False,
                'block_id': False,
                'trip_idx': False,
                'delay': ':.2f',
                'start_time': True,
                'end_time': True
            }

            if len(on_time_trips) > 0:
                tl_ontime = px.timeline(
                    on_time_trips,
                    x_start='start_time',
                    x_end='end_time',
                    y='block_id',
                    category_orders={'block_id': block_order},
                    range_x=[t_start, t_end],
                    hover_name='trip_name',
                    hover_data=trip_hover,
                    labels=trip_col_labels,
                    color='Delayed',
                    color_discrete_map={'Delayed Trip': 'rgba(213, 94, 0, 1)',
                                        'On-Time Trip': 'rgba(128, 128, 128, 1)'}
                )
                fig.add_trace(tl_ontime.data[0], secondary_y=True)

            if len(delayed_trips) > 0:
                trip_name = 'Block ' + delayed_trips['block_id'].astype(str) \
                            + ', Trip ' + delayed_trips['trip_idx'].astype(str)
                tl_delayed = px.timeline(
                    delayed_trips,
                    x_start='start_time',
                    x_end='end_time',
                    y='block_id',
                    hover_name=trip_name,
                    hover_data=trip_hover,
                    labels=trip_col_labels,
                    category_orders={'block_id': block_order},
                    range_x=[t_start, t_end],
                    color='Delayed',
                    color_discrete_map={'Delayed Trip': 'rgba(213, 94, 0, 1)',
                                        'On-Time Trip': 'rgba(128, 128, 128, 1)'}
                )
                fig.add_trace(tl_delayed.data[0], secondary_y=True)

        # Add charger utilization timeline
        # Filter out trips without charging
        c_df = c_df[c_df['chg_time'] > 0.1]
        c_df['Status'] = 'Charging'
        tl_chg = px.timeline(
            c_df,
            x_start='plugin_time',
            x_end='finish_chg_time',
            y='block_id',
            category_orders={'block_id': block_order},
            labels={
                'plugin_time': 'Plugin Time',
                'block_id': 'Block ID',
                'finish_chg_time': 'Charging End Time'
            },
            range_x=[t_start, t_end],
            color='Status',
            color_discrete_map={'Charging': 'rgba(0,114,178, 1)'},
            hover_name='trip_name',
            hover_data={
                'Status': False,
                'block_id': False,
                #                 'trip_idx': False,
                #                 'delay': ':.2f',
                'plugin_time': True,
                'finish_chg_time': True
            }
        )
        fig.add_trace(tl_chg.data[0], secondary_y=True)

        # Clean up formatting
        # Sort y axis. This gets lost even though we specified it when making the timeline.
        fig.update_layout(
            yaxis2_categoryorder='array', yaxis2_categoryarray=block_order,
            yaxis2_side='left',
            yaxis2_tickfont_size=10, yaxis2_title='Block ID')
        fig.update_layout(
            dict(
                barmode='overlay',
                title=title_fmt.format(c),
                #                 yaxis_title='Bus',
                xaxis={'range': [t_start, t_end]},
                margin=dict(l=20, r=20, t=40, b=20)
            )
        )
        fig.update_layout(legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=0.95
        ))
        #         fig.update_yaxes(type='category')

        config = {
            'toImageButtonOptions': {
                'format': 'png',
                'scale': 3
            }
        }
        fig.show(config=config)
        fig_list.append(fig)

    return fig_list
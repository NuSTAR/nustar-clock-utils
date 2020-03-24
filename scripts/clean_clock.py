"""Flask + Plotly + Dash application to interactively flag bad clock points."""

import os
import copy
import time

import numpy as np
from astropy.table import Table
from astropy.time import Time
import pickle

from astropy import log
from nuclockutils.nustarclock import NUSTAR_MJDREF, get_malindi_data_except_when_out
from nuclockutils.nustarclock import temperature_correction_table, \
    residual_roll_std, flag_bad_points
from nuclockutils.nustarclock import read_clock_offset_table, read_temptable, \
    read_freq_changes_table
from nuclockutils.utils import fix_byteorder, \
    get_obsid_list_from_heasarc, spline_through_data, \
    aggregate_all_tables, cross_two_gtis, get_rough_trend_fun, \
    merge_and_sort_arrays, eliminate_array_from_array, find_idxs, filter_dict_with_re

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from textwrap import dedent as d
import json
from flask_caching import Cache


def spline_detrending(clock_offset_table, temptable):
    tempcorr_idx = np.searchsorted(temptable['met'], clock_offset_table['met'])
    clock_residuals = np.array(clock_offset_table['offset'] - temptable['temp_corr'][tempcorr_idx])

    detrend_fun = spline_through_data(clock_offset_table['met'],
                                      clock_residuals)

    r_std = residual_roll_std(
        clock_residuals - detrend_fun(clock_offset_table['met']))

    clidx = np.searchsorted(clock_offset_table['met'], temptable['met'])
    clidx[clidx == clock_offset_table['met'].size] = \
        clock_offset_table['met'].size - 1
    temptable['std'] = r_std[clidx]

    temptable['temp_corr_trend'] = detrend_fun(temptable['met'])

    temptable['temp_corr_detrend'] = temptable['temp_corr'] + temptable['temp_corr_trend']

    return temptable



def eliminate_trends_in_residuals(temp_table, clock_offset_table,
                                  gtis, debug=False):

    good = clock_offset_table['met'] < np.max(temp_table['met'])
    clock_offset_table = clock_offset_table[good]

    tempcorr_idx = np.searchsorted(temp_table['met'],
                                   clock_offset_table['met'])

    clock_residuals = clock_offset_table['offset'] - \
                      temp_table['temp_corr'][tempcorr_idx]

    use_for_interpol, bad_malindi_time = \
        get_malindi_data_except_when_out(clock_offset_table)

    clock_residuals[bad_malindi_time] -= 0.0005

    good = (clock_residuals == clock_residuals) & ~clock_offset_table['flag'] & use_for_interpol

    clock_offset_table = clock_offset_table[good]
    clock_residuals = clock_residuals[good]

    for g in gtis:
        log.info(f"Treating data from METs {g[0]}--{g[1]}")
        start, stop = g

        cl_idx_start, cl_idx_end = \
            np.searchsorted(clock_offset_table['met'], g)

        if cl_idx_end - cl_idx_start == 0:
            continue

        temp_idx_start, temp_idx_end = \
            np.searchsorted(temp_table['met'], g)

        table_new = temp_table[temp_idx_start:temp_idx_end]
        cltable_new = clock_offset_table[cl_idx_start:cl_idx_end]
        met = cltable_new['met']
        residuals = clock_residuals[cl_idx_start:cl_idx_end]

        p_new = get_rough_trend_fun(met, residuals)

        if p_new is not None:
            p = p_new

        table_new['temp_corr'] += p(table_new['met'])

        if debug:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.plot(table_new['met'], table_new['temp_corr'], alpha=0.5)
            plt.scatter(cltable_new['met'], cltable_new['offset'])
            plt.plot(table_new['met'], table_new['temp_corr'])
            plt.savefig(f'{int(start)}--{int(stop)}_detr.png')
            plt.close(fig)

        print(f'df/f = {(p(stop) - p(start)) / (stop - start)}')

    # btis = get_btis(gtis, start_time=gtis[1, 0], stop_time=gtis[-2, 1])
    btis = np.array(
        [[g0, g1] for g0, g1 in zip(gtis[:-1, 1], gtis[1:, 0])])

    # Interpolate the solution along bad time intervals
    for g in btis:
        log.info(f"Treating bad data from METs {g[0]}--{g[1]}")
        start, stop = g
        temp_idx_start, temp_idx_end = \
            np.searchsorted(temp_table['met'], g)
        if temp_idx_end - temp_idx_start == 0:
            continue
        table_new = temp_table[temp_idx_start:temp_idx_end]

        last_good_tempcorr = temp_table['temp_corr'][temp_idx_start - 1]
        next_good_tempcorr = temp_table['temp_corr'][temp_idx_end + 1]
        last_good_time = temp_table['met'][temp_idx_start - 1]

        time_since_last_good_tempcorr = \
            table_new['met']- last_good_time

        m = (next_good_tempcorr - last_good_tempcorr) / \
            (time_since_last_good_tempcorr[-1] - time_since_last_good_tempcorr[0])
        q = last_good_tempcorr

        table_new['temp_corr'][:] = q + \
            time_since_last_good_tempcorr * m

        # temp_table[temp_idx_start:temp_idx_end]

    log.info("Final detrending...")
    t0 = time.time()
    table_new = spline_detrending(clock_offset_table, temp_table)
    log.info(f"Done. It took {time.time() - t0} s")

    return table_new



def find_good_time_intervals(temperature_table,
                             clock_offset_table,
                             clock_jump_times):
    start_time = temperature_table['met'][0]
    stop_time = temperature_table['met'][-1]
    clock_gtis = []
    current_start = start_time
    for jump in clock_jump_times:
        # To avoid that the gtis get fused, I subtract 1 ms
        # from GTI stop
        clock_gtis.append([current_start, jump])
        current_start = jump
    clock_gtis.append([current_start, stop_time])
    clock_gtis = np.array(clock_gtis)

    temp_condition = np.concatenate(
        ([False], np.diff(temperature_table['met']) > 600, [False]))

    temp_edges_l = np.concatenate((
        [temperature_table['met'][0]], temperature_table['met'][temp_condition[:-1]]))

    temp_edges_h = np.concatenate((
        [temperature_table['met'][temp_condition[1:]], [temperature_table['met'][-1]]]))

    temp_gtis = np.array(list(zip(
        temp_edges_l, temp_edges_h)))

    for t in temp_gtis:
        print(t[0], t[1], t[1] - t[0])

    gtis = cross_two_gtis(temp_gtis, clock_gtis)

    return gtis


def calculate_stats(all_data):
    from statsmodels.robust.scale import mad
    log.info("Calculating statistics")
    scatter = mad(all_data['residual_detrend'])
    print("Stats:")
    print(f"Median absolute deviation over whole NuSTAR history: {scatter * 1e6} us")


def load_and_flag_clock_table(clockfile="latest_clock.dat"):
    clock_offset_table = read_clock_offset_table(clockfile)
    clock_offset_table = flag_bad_points(
        clock_offset_table, db_file='BAD_POINTS_DB.dat')
    return clock_offset_table


def recalc(outfile='save_all.pickle'):
    if os.path.exists('tcxo_tmp_archive.hdf5'):
        temptable_raw = \
            read_temptable('tcxo_tmp_archive.hdf5')
        temptable_raw = fix_byteorder(temptable_raw)
    else:
        temptable_raw = \
            read_temptable('tcxo_tmp_archive.csv')
        temptable_raw.write('tcxo_tmp_archive.hdf5', overwrite=True)

    log.info("Querying history of NuSTAR observations...")
    all_nustar_obs = get_obsid_list_from_heasarc()
    all_nustar_obs.sort('MET')

    all_nustar_obs['text'] = [
        (f"Source: {aobs['NAME']}<br>"+
         f"ObsID: {aobs['OBSID']}<br>"+
         f"Start: MJD {aobs['TIME']}<br>"+
         f"End: MJD {aobs['END_TIME']}<br>"+
         f"Type: {aobs['OBS_TYPE']}<br>"+
         f"Mode: {aobs['OBSERVATION_MODE']}<br>")
        for aobs in all_nustar_obs]

    clock_offset_table = load_and_flag_clock_table(clockfile="latest_clock.dat")

    freq_change_table = read_freq_changes_table("latest_freq.dat")

    table_times = temptable_raw['met']
    met_start = table_times[0]
    met_stop = table_times[-1]
    table = temperature_correction_table(
        met_start, met_stop, temptable=temptable_raw,
        freqchange_file='latest_freq.dat',
        time_resolution=10, craig_fit=False, hdf_dump_file='dump.hdf5')

    clock_jump_times = np.array([78708320, 79657575, 81043985, 82055671])

    gtis = find_good_time_intervals(temptable_raw,
                                    clock_offset_table,
                                    clock_jump_times)

    table_new = eliminate_trends_in_residuals(
        copy.deepcopy(table), clock_offset_table, gtis)

    mets = np.array(table_new['met'])
    start = mets[0]
    stop = mets[-1]

    good_mets = clock_offset_table['met'] < table_new['met'].max()
    clock_offset_table = clock_offset_table[good_mets]

    clock_mets = clock_offset_table['met']
    clock_mjds = clock_offset_table['mjd']
    dates = Time(clock_mjds[:-1], format='mjd')

    tempcorr_idx = np.searchsorted(table_new['met'], clock_offset_table['met'])
    clock_residuals = np.array(clock_offset_table['offset'] - table_new['temp_corr'][tempcorr_idx])

    clock_residuals_detrend = np.array(clock_offset_table['offset'] - table_new['temp_corr_detrend'][tempcorr_idx])

    all_data = Table({'met': clock_mets[:-1],
                      'mjd': np.array(clock_mjds[:-1], dtype=int),
                      'doy': dates.strftime("%Y:%j"),
                      'utc': dates.strftime("%Y:%m:%d"),
                      'offset': clock_offset_table['offset'][:-1],
                      'residual': clock_residuals[:-1],
                      'residual_detrend': clock_residuals_detrend[:-1],
                      'station': clock_offset_table['station'][:-1]})
    temptable_data = Table({'met': temptable_raw['met'],
                            'temperature': temptable_raw['temperature'],
                            'temperature_smooth': temptable_raw['temperature_smooth'],})

    pickle.dump((all_data, temptable_data, freq_change_table, table_new, gtis, all_nustar_obs),
                open(outfile, 'wb'))

    calculate_stats(all_data)

    return all_data, temptable_data, freq_change_table, table_new, gtis, all_nustar_obs


def plot_dash(all_data, temptable_data, freq_change_data, table_new, gti, all_nustar_obs,
    axis_ranges=None):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import datashader.transfer_functions as tf
    from datashader.colors import inferno
    import datashader as ds

    log.info("Updating plot")
    all_data = flag_bad_points(all_data)

    temptable_data, table_new = \
        aggregate_all_tables([temptable_data, table_new], max_number=300)

    bad = all_data['flag'] == True

    hovertemplate=('MET: %{x:d}<br><br>'
                   '<b>Current observation:</b><br>'
                   '%{text}')

    # Add a final line to include overflow point
    all_nustar_obs.add_row(all_nustar_obs[-1])
    for col in all_nustar_obs.colnames:
        if isinstance(all_nustar_obs[col][0], str):
            all_nustar_obs[col][-1] = ""
        else:
            all_nustar_obs[col][-1] *= 0

    idx = np.searchsorted(all_nustar_obs['MET'][:-1], table_new['met'])

    all_nustar_obs_reindex = all_nustar_obs[idx]

    fig = make_subplots(3, 1, shared_xaxes=True, vertical_spacing=0.02)
    fig.append_trace(go.Scattergl({
        'x': table_new['met'],
        'y': table_new['temp_corr'] * 1e3,
        'hovertemplate': hovertemplate,
        'text': all_nustar_obs_reindex['text'],
        'mode': 'lines',
        'name': f'Temperature correction',
        'marker': {'color': 'grey'}
    }), 1, 1)
    fig.append_trace(go.Scattergl({
        'x': table_new['met'],
        'y': table_new['temp_corr_trend'] * 1e3,
        'hovertemplate': hovertemplate,
        'text': all_nustar_obs_reindex['text'],
        'mode': 'lines',
        'showlegend': False,
        'marker': {'color':'grey'}
    }), 2, 1)
    for sign in [-1, 1]:
        fig.append_trace(go.Scattergl({
            'x': table_new['met'],
            'y': sign * table_new['std'] * 1e3,
            'hovertemplate': hovertemplate,
            'text': all_nustar_obs_reindex['text'],
            'mode': 'lines',
            'showlegend': False,
            'marker': {'color':'black'}
         }), 3, 1)

    all_nustar_obs = all_nustar_obs[:-1]

    idx = np.searchsorted(all_nustar_obs['MET'][:-1], np.array(all_data['met']))

    all_nustar_obs_reindex = all_nustar_obs[idx]

    fig.append_trace(go.Scattergl({
        'x': all_data['met'][bad],
        'y': all_data['offset'][bad] * 1e3,
        'hovertemplate': hovertemplate,
        'text': all_nustar_obs_reindex['text'],
        'mode': 'markers',
        'name': f'Bad clock offset measurements',
        'marker': {'color': 'grey', 'symbol': "x-dot", 'size': 3}
    }), 1, 1)

    all_data_bad = all_data[bad]

    for ydata, row in zip(['residual', 'residual_detrend'], [2, 3]):
        fig.append_trace(go.Scattergl({
            'x': all_data_bad['met'],
            'y': all_data_bad[ydata] * 1e3,
            'hovertemplate': hovertemplate,
            'text': all_nustar_obs_reindex['text'][bad],
            'mode': 'markers',
            'showlegend': False,
            'marker': {'color': 'grey', 'symbol': "x-dot", 'size': 3}
         }), row, 1)

    for station, color in zip(['MLD', 'SNG', 'UHI'], ['blue', 'red', 'orange']):
        good = (all_data['station'] == station) & ~bad
        all_data_good = all_data[good]
        fig.append_trace(go.Scattergl({
            'x': all_data_good['met'],
            'y': all_data_good['offset'] * 1e3,
            'hovertemplate': hovertemplate,
            'text': all_nustar_obs_reindex['text'][good],
            'mode': 'markers',
            'marker': {'color': color, 'size': 3},
            'name': f'Clock offset - {station}'
        }), 1, 1)
        for ydata, row in zip(['residual', 'residual_detrend'], [2, 3]):
            fig.append_trace(go.Scattergl({
                'x': all_data_good['met'],
                'y': all_data_good[ydata] * 1e3,
                'hovertemplate': hovertemplate,
                'text': all_nustar_obs_reindex['text'][good],
                'showlegend': False,
                'mode': 'markers',
                'marker': {'color': color, 'size': 3}
            }), row, 1)

    bad_intervals = np.array(
        [[g0, g1] for g0, g1 in zip(gti[:-1, 1], gti[1:, 0])])

    shapes = []
    for bti in bad_intervals:
        if bti[-1] < all_data['met'][0]:
            continue
        if bti[0] > all_data['met'][-1]:
            continue
        shapes.append(dict(type="rect",
                # x-reference is assigned to the x-values
                xref="x",
                # y-reference is assigned to the plot paper [0,1]
                yref="paper",
                x0=max(bti[0], all_data['met'][0]),
                y0=0,
                x1=min(bti[1], all_data['met'][-1]),
                y1=1,
                fillcolor="LightSalmon",
                opacity=0.5,
                layer="below",
                line_width=0,
            ))

    if axis_ranges is not None:
        if 'xaxis.range[0]' in axis_ranges:
            xranges = axis_ranges[f'xaxis.range[0]'], axis_ranges[f'xaxis.range[1]']
            fig.update_xaxes(range=xranges, row=3, col=1)
        for i, row in enumerate([1, 2, 3]):
            label = 'yaxis.range' if row == 1 else f'yaxis{row}.range'
            if label + '[0]' in axis_ranges:
                ranges = axis_ranges[f'{label}[0]'], axis_ranges[f'{label}[1]']
                fig.update_yaxes(row=row, range=ranges, col=1)

    fig.update_xaxes(
        title_text=f"Mission Epoch Time (s from MJD {NUSTAR_MJDREF})",
        row=3, col=1)

    for axis_no, title in zip(
        [1, 2, 3],
        ['Clock offset (ms)', 'Residuals (ms)', 'Detrended Residuals (ms)']):
        # Update yaxis properties
        fig.update_yaxes(title_text=title, row=axis_no, col=1)
        # Add shape regions
        for s in shapes:
            s['xref'] = f'x{axis_no}'
        fig.update_layout(shapes=shapes)

    fig['layout'].update(height=900, legend=dict(orientation="h", x=0, y=1),
                         margin={'t': 20})
    fig['layout']['clickmode'] = 'event+select'

    return fig


def default_axes():
    return {'yaxis.range[0]': -500, 'yaxis.range[1]': 1000,
        'yaxis2.range[0]': -60, 'yaxis2.range[1]': 30,
        'yaxis3.range[0]': -2, 'yaxis3.range[1]': 1}

CURRENT_AXES = default_axes()


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width"}])

CACHE_CONFIG = {
    # try 'filesystem' if you don't want to setup redis
    'CACHE_TYPE': 'simple',
    # 'CACHE_REDIS_URL': os.environ.get('REDIS_URL', 'redis://localhost:6379')
}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)


@cache.memoize()
def global_store(file):
    if not os.path.exists(file):
        recalc(file)

    log.info(f"Reading data from {file}")
    return pickle.load(open(file, 'rb'))


styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = html.Div(className="row", children=[
    html.Div(
        className="three columns div-user-controls",
        children=[
            html.H1(
                children='NuSTAR Clock offset cleaner',
                # style={
                # 'textAlign': 'center',
                # 'color': colors['text']
                #     }
                    ),
            dcc.Markdown(d(
                """
                To navigate the plot, use the Zoom and Pan tools in the graph's
                menu bar.
                [Detailed instructions here](https://plot.ly/chart-studio-help/zoom-pan-hover-controls/)

                Choose the lasso or rectangle tool and then select points in any
                of the three plots to mark them as "bad".

                Select in an empty area to eliminate the current selection.

                To save the current selection in the bad clock offset database,
                press "Update blacklist"

                When you are done with the selection, press "Recalculate"

                To undo the current selection, press "Reset"

                Light salmon areas indicate known issues like major clock readjustments
                or times where no temperature measurements are present.
            """)),
            html.Div(
                className="row",
                children=[
                    html.Button("Reset", id='refresh-button', n_clicks=0),
                    html.Button("Recalculate", id='recalculate-button', n_clicks=0),
                    html.Button("Update blacklist", id='bad-data-button', n_clicks=0),
                    html.Button("Eliminate from blacklist", id='actually-good-data-button', n_clicks=0),
                ]),
            html.Div([
                html.H2(
                    children='Selected Data',
                        ),
                html.P(
                    html.Pre(id='selected-data'),)
            ])
        ],

    ),
    html.Div(
        id='figure-div',
        className="six columns div-for-charts bg-grey",
        children=[
            dcc.Graph(figure=plot_dash(*global_store('save_all.pickle'),
                axis_ranges=CURRENT_AXES), id='my-figure',
                hoverData={'points': [{'x': 2e8}]}),
        ],
    ),
    html.Div(
        className="three columns div-for-charts bg-grey",
        children=[
        # dcc.Graph(id='precise-correction-time-series'),
        dcc.Graph(id='temperature-time-series'),
        # dcc.Graph(id='frequency-time-series'),
        ]
    ),
    html.Div(id='dummy', style={'display': 'none'}),
    html.Div(id='clock-intermediate-value', style={'display': 'none'}),
    html.Div(id='freq-intermediate-value', style={'display': 'none'}),
    html.Div(id='axis-properties', style={'display': 'none'})
])


@app.callback(
    Output('axis-properties', 'children'),
    [Input('my-figure', 'relayoutData')])
def update_x_ranges(relayoutData):
    global CURRENT_AXES
    if relayoutData is None:
        raise PreventUpdate

    log.info("Received relayoutData: ", relayoutData)
    rangevals_re = r'^[xy].*\.range.*$'
    xzoom_dict = filter_dict_with_re(relayoutData, rangevals_re)

    rangevals_re = r'^[xy].*\.autorange.*$'
    autorange_dict = filter_dict_with_re(relayoutData, rangevals_re)

    if autorange_dict == {} and xzoom_dict == {}:
        log.info("No range information.")
        raise PreventUpdate

    if autorange_dict != {}:
        CURRENT_AXES = default_axes()
    else:
        new_ranges = xzoom_dict
        for key, value in new_ranges.items():
            CURRENT_AXES[key] = value
    return json.dumps(CURRENT_AXES)


@app.callback(
    Output('my-figure', 'figure'),
    [Input('axis-properties', 'children')])
def display_relayout_data(children):
    if children is None:
        raise PreventUpdate

    relayoutData = json.loads(children)

    log.info(f"New relayoutData received: {relayoutData}")

    if relayoutData is None:
        raise PreventUpdate

    all_data, temptable_data, freq_change_data, \
        table_new_complete, gti, all_nustar_obs = global_store('save_all.pickle')
    if 'xaxis.range[0]' in relayoutData:

        met_min, met_max = relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']

        all_data = all_data[find_idxs(all_data['met'], met_min, met_max)]
        temptable_data = \
            temptable_data[find_idxs(temptable_data['met'], met_min, met_max)]
        table_new_complete = \
            table_new_complete[find_idxs(table_new_complete['met'], met_min, met_max)]

    return plot_dash(
        all_data, temptable_data, freq_change_data,
        table_new_complete, gti, all_nustar_obs,
        axis_ranges=relayoutData)


@app.callback(
    Output("figure-div", "children"),
    [Input("refresh-button", "n_clicks"),
     Input("bad-data-button", "n_clicks"),
     Input("actually-good-data-button", "n_clicks"),
     Input("recalculate-button", "n_clicks"),
     Input('selected-data', 'children')])
def refresh_output(n_cl_refresh, n_cl_bad, n_cl_good, n_cl_recalc, selectedData):
    global CURRENT_AXES
    ctx = dash.callback_context

    print(ctx)
    if not ctx.triggered:
        raise PreventUpdate

    who_triggered = ctx.triggered[0]['prop_id'].split('.')[0]

    log.info(f"Pressed {who_triggered}")

    if who_triggered == 'selected-data':
        print("Preventing update")
        raise PreventUpdate

    selectedData = json.loads(selectedData)
    ALL_BAD_POINTS = np.genfromtxt('BAD_POINTS_DB.dat')
    NEW_BAD_POINTS = []

    if 'points' in selectedData:
        NEW_BAD_POINTS = np.array([p['x'] for p in selectedData['points']])

    if who_triggered == 'recalculate-button':
        log.info("Recalculating all")
        cache.delete_memoized(global_store, 'save_all.pickle')
        if os.path.exists('save_all.pickle'):
            os.unlink('save_all.pickle')
    elif who_triggered == 'actually-good-data-button' and len(NEW_BAD_POINTS) > 0:
        log.info("Removing point(s) from bad clock offset database")
        ALL_BAD_POINTS = eliminate_array_from_array(
            ALL_BAD_POINTS, NEW_BAD_POINTS)
        np.savetxt('BAD_POINTS_DB.dat', ALL_BAD_POINTS, fmt='%d')
    elif who_triggered == 'bad-data-button' and len(NEW_BAD_POINTS) > 0:
        log.info("Adding point(s) to bad clock offset database")
        ALL_BAD_POINTS = merge_and_sort_arrays(
            ALL_BAD_POINTS, NEW_BAD_POINTS)
        np.savetxt('BAD_POINTS_DB.dat', ALL_BAD_POINTS, fmt='%d')
    else:
        log.info("Refreshing plot")
        CURRENT_AXES = default_axes()

    return [
        dcc.Graph(figure=plot_dash(*global_store('save_all.pickle'),
            axis_ranges=CURRENT_AXES), id='my-figure',
            hoverData={'points': [{'x': 2e8}]}),]


@app.callback(
    Output('selected-data', 'children'),
    [Input('my-figure', 'selectedData')])
def display_selected_data(selectedData):
    if selectedData is None:
        return "No data selected"
    else:
        return json.dumps(selectedData, indent=2)


temptable = read_temptable('tcxo_tmp_archive.hdf5')

def create_temperature_timeseries(x, y, axis_type='linear'):
    return {
        'data': [dict(
            x=x,
            y=y,
            mode='lines'
        )],
        'layout': {
            'height': 300,
            'yaxis': {'title': 'TCXO Temperature',
                'type': 'linear' if axis_type == 'Linear' else 'log'},
            'xaxis': {'title': 'MET', 'showgrid': False,
            'margin':{'t': 20}}

        }
    }


@app.callback(
    Output('temperature-time-series', 'figure'),
    [Input('my-figure', 'hoverData'),
     Input("clock-intermediate-value", "children")])
def update_temperature_timeseries(hoverData, clockfile):
    if hoverData is None:
        raise PreventUpdate

    xstart = hoverData['points'][0]['x'] - 200000
    xstop = hoverData['points'][0]['x'] + 200000

    istart, istop = np.searchsorted(temptable['met'], [xstart, xstop])

    return create_temperature_timeseries(
        temptable['met'][istart:istop:18],
        temptable['temperature'][istart:istop:18])


if __name__ == '__main__':
    app.run_server(debug=True)

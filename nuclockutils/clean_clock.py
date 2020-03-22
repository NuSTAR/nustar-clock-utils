"""Flask + Plotly + Dash application to interactively flag bad clock points."""

import glob
import os
import copy
import time

import tqdm
import numpy as np
from astropy.table import Table
from astropy.time import Time
import corner
import lmfit
import pickle
import datashader as ds
from lmfit import fit_report
# from get_freq_from_clock_offset import get_f_from_clock_offset, get_temptable
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from astropy import log
from astroquery.heasarc import Heasarc
from nuclockutils.nustarclock import NUSTAR_MJDREF
from nuclockutils.nustarclock import temperature_correction_table
from nuclockutils.utils import sec_to_mjd, fix_byteorder
from nuclockutils.nustarclock import read_clock_offset_table, read_temptable, \
    read_freq_changes_table, sec_to_mjd, robust_linear_fit
from bokeh.models import HoverTool

from stingray.gti import create_gti_from_condition, cross_two_gtis, get_btis
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from textwrap import dedent as d
import json
from flask_caching import Cache



def get_obsid_list_from_heasarc(cache_file='heasarc.pickle'):
    if os.path.exists(cache_file):
        return pickle.load(open(cache_file, 'rb'))

    heasarc = Heasarc()
    all_nustar_obs = heasarc.query_object(
        '*', 'numaster', resultmax=10000,
        fields='OBSID,TIME,END_TIME,NAME,OBSERVATION_MODE,OBS_TYPE')

    all_nustar_obs = all_nustar_obs[all_nustar_obs["TIME"] > 0]
    for field in 'OBSID,NAME,OBSERVATION_MODE,OBS_TYPE'.split(','):
        all_nustar_obs[field] = [om.strip() for om in all_nustar_obs[field]]

    # all_nustar_obs = all_nustar_obs[all_nustar_obs["OBSERVATION_MODE"] == 'SCIENCE']
    all_nustar_obs['MET'] = np.array(all_nustar_obs['TIME'] - NUSTAR_MJDREF) * 86400
    return all_nustar_obs


def get_malindi_data_except_when_not(clock_offset_table):
    no_malindi_intvs = [[93681591, 98051312]]
    clock_mets = clock_offset_table['met']

    bad_malindi_time = np.zeros(len(clock_mets), dtype=bool)
    for nmi in no_malindi_intvs:
        bad_malindi_time = bad_malindi_time | (clock_mets >= nmi[0]) & (
                    clock_mets < nmi[1])

    malindi_stn = clock_offset_table['station'] == 'MLD'
    use_for_interpol = \
        (malindi_stn | bad_malindi_time)

    return use_for_interpol, bad_malindi_time


def rolling_window(a, window):
    """https://stackoverflow.com/a/59322185"""
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad,mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_std(a, window):
    return np.std(rolling_window(a, window), axis=-1)


def final_detrending(clock_offset_table, temptable):
    from nuclockutils.spline import spline_fit
    from scipy.interpolate import CubicSpline, UnivariateSpline, LSQUnivariateSpline

    tempcorr_idx = np.searchsorted(temptable['met'], clock_offset_table['met'])
    clock_residuals = np.array(clock_offset_table['offset'] - temptable['temp_corr'][tempcorr_idx])

    use_for_interpol, bad_malindi_time = \
        get_malindi_data_except_when_not(clock_offset_table)
    clock_residuals[bad_malindi_time] -= 0.0005

    good = (clock_residuals == clock_residuals) & ~clock_offset_table['flag'] & use_for_interpol

    lo_lim = clock_offset_table['met'][good][0]
    hi_lim = clock_offset_table['met'][good][-1]
    control_points = \
        np.linspace(
            lo_lim + 86400, hi_lim - 86400,
            clock_offset_table['met'][good].size // 5)

    detrend_fun = LSQUnivariateSpline(clock_offset_table['met'][good],
                                      clock_residuals[good],
                                      t=control_points,
                                      k=2,
                                      bbox=[lo_lim - 1000, hi_lim + 1000],
                                      )
    detrend_fun.set_smoothing_factor(0.0001)

    r_std = rolling_std(np.diff(clock_residuals[good]), 20) / np.sqrt(2)
    r_std = np.roll(r_std, -10)
    r_std[-10:] = r_std[-11]
    r_std = np.concatenate((r_std[:1], r_std, r_std[-1:], ))

    temptable['std'] = \
        r_std[np.searchsorted(clock_offset_table['met'][good], temptable['met'])]

    temptable['temp_corr_trend'] = detrend_fun(temptable['met'])

    temptable['temp_corr_detrend'] = temptable['temp_corr'] + detrend_fun(temptable['met'])

    return temptable


def get_trend_fun(table_new, cltable_new):
    good = cltable_new['met'] < np.max(table_new['met'])
    cltable_new = cltable_new[good]

    tempcorr_idx = np.searchsorted(table_new['met'],
                                   cltable_new['met'])

    clock_residuals = cltable_new['offset'] - \
                      table_new['temp_corr'][tempcorr_idx]

    use_for_interpol, bad_malindi_time = \
        get_malindi_data_except_when_not(cltable_new)
    good = (clock_residuals == clock_residuals) & ~cltable_new['flag'] & use_for_interpol

    cltable_new_filt = cltable_new[good]
    clock_residuals[bad_malindi_time] -= 0.0005

    clock_residuals_filt = clock_residuals[good]

    print('Size', cltable_new_filt['met'].size)
    if cltable_new_filt['met'].size <= 1:
        return None
    elif cltable_new_filt['met'].size > 200:
        mets = [np.mean(cltable_new_filt['met'][:50]),
                np.mean(cltable_new_filt['met'][-50:])]

        res = [np.median(clock_residuals_filt[:50]),
            np.median(clock_residuals_filt[-50:])]

        m = (res[1] - res[0]) / (mets[1] - mets[0])
        q = res[0]

        def p(times):
            return (times - mets[0]) * m + q
    else:
        fit_result = robust_linear_fit(cltable_new_filt['met'],
                                       clock_residuals_filt)
        m, q = fit_result.estimator_.coef_, \
        fit_result.estimator_.intercept_

        def p(times):
            return times * m + q

    return p


def eliminate_trends_in_residuals(temp_table, clock_offset_table,
                                  gtis):
    for g in gtis:
        log.info(f"Treating data from METs {g[0]}--{g[1]}")
        start, stop = g
        temp_idx_start, temp_idx_end = \
            np.searchsorted(temp_table['met'], g)

        cl_idx_start, cl_idx_end = \
            np.searchsorted(clock_offset_table['met'], g)

        if cl_idx_end - cl_idx_start == 0:
            continue

        table_new = temp_table[temp_idx_start:temp_idx_end]
        cltable_new = clock_offset_table[cl_idx_start:cl_idx_end]

        p_new = get_trend_fun(table_new, cltable_new)
        if p_new is not None:
            p = p_new
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(table_new['met'], table_new['temp_corr'], alpha=0.5)
        plt.scatter(cltable_new['met'], cltable_new['offset'])
        table_new['temp_corr'] += p(table_new['met'])
        plt.plot(table_new['met'], table_new['temp_corr'])
        plt.savefig(f'{int(start)}--{int(stop)}_detr.png')
        plt.close(fig)

        temp_table[temp_idx_start:temp_idx_end]
        print(f'df/f = {(p(stop) - p(start)) / (stop - start)}')


    btis = get_btis(gtis, start_time=gtis[1, 0], stop_time=gtis[-2, 1])

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

    return temp_table



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
        clock_gtis.append([current_start, jump - 1e-3])
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
    for t in gtis:
        print(t[0], t[1], t[1] - t[0])
    return gtis


def flag_bad_points(all_data, db_file='BAD_POINTS_DB.dat'):
    if not os.path.exists(db_file):
        return all_data

    ALL_BAD_POINTS = np.genfromtxt(db_file)
    ALL_BAD_POINTS.sort()
    ALL_BAD_POINTS = np.unique(ALL_BAD_POINTS)
    idxs = all_data['met'].searchsorted(ALL_BAD_POINTS)

    mask = np.array(all_data['flag'], dtype=bool)

    for idx in idxs:
        if idx >= mask.size:
            continue
        mask[idx] = True
    all_data['flag'] = mask
    return all_data


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


def aggregate(table, max_number=1000):
    N = len(table)
    if N < max_number:
        return table
    rebin_factor = int(np.ceil(len(table) / max_number))
    table['__binning__'] = np.arange(N) // rebin_factor

    if isinstance(table, Table):
        binned = table.group_by('__binning__').groups.aggregate(np.mean)
        return binned

    return table.groupby('__binning__').mean()


def aggregate_all_tables(table_list, max_number=1000):
    return [aggregate(table) for table in table_list]


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

    log.info("Final detrending...")
    t0 = time.time()
    table_new = final_detrending(clock_offset_table, table_new)
    log.info(f"Done. It took {time.time() - t0} s")
    clock_residuals_detrend = np.array(clock_offset_table['offset'] - table_new['temp_corr_detrend'][tempcorr_idx])

    all_data = pd.DataFrame({'met': clock_mets[:-1],
                             'mjd': np.array(clock_mjds[:-1], dtype=int),
                             'doy': dates.strftime("%Y:%j"),
                             'utc': dates.strftime("%Y:%m:%d"),
                             'offset': clock_offset_table['offset'][:-1],
                             'residual': clock_residuals[:-1],
                             'residual_detrend': clock_residuals_detrend[:-1],
                             'station': clock_offset_table['station'][:-1],
                             'flag': clock_offset_table['flag'][:-1]})
    temptable_data = pd.DataFrame({'met': temptable_raw['met'],
                                   'temperature': temptable_raw['temperature'],
                                   'temperature_smooth': temptable_raw['temperature_smooth'],})
    freq_change_data = freq_change_table.to_pandas()

    pickle.dump((all_data, temptable_data, freq_change_data, table_new, gtis, all_nustar_obs),
                open(outfile, 'wb'))

    calculate_stats(all_data)

    return all_data, temptable_data, freq_change_data, table_new, gtis, all_nustar_obs


def plot_dash(all_data, temptable_data, freq_change_data, table_new, gti, all_nustar_obs,
    axis_ranges=None):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import datashader.transfer_functions as tf
    from datashader.colors import inferno
    import datashader as ds

    all_data = flag_bad_points(all_data)

    temptable_data, table_new = \
        aggregate_all_tables([temptable_data, table_new], max_number=300)

    bad = all_data['flag'] == True

    hovertemplate=('MET: %{x:d}<br><br>'
                   '<b>Current observation:</b><br>'
                   '%{text}')

    all_nustar_obs['text'] = [
        (f"Source: {aobs['NAME']}<br>"+
         f"ObsID: {aobs['OBSID']}<br>"+
         f"Start: MJD {aobs['TIME']}<br>"+
         f"End: MJD {aobs['END_TIME']}<br>"+
         f"Type: {aobs['OBS_TYPE']}<br>"+
         f"Mode: {aobs['OBSERVATION_MODE']}<br>")
        for aobs in all_nustar_obs]

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
        'x': all_data['met'][bad].values,
        'y': all_data['offset'][bad].values * 1e3,
        'hovertemplate': hovertemplate,
        'text': all_nustar_obs_reindex['text'],
        'mode': 'markers',
        'name': f'Bad clock offset measurements',
        'marker': {'color': 'grey', 'symbol': "x-dot", 'size': 3}
    }), 1, 1)

    for ydata, row in zip(['residual', 'residual_detrend'], [2, 3]):
        fig.append_trace(go.Scattergl({
            'x': all_data['met'][bad].values,
            'y': all_data[ydata][bad].values * 1e3,
            'hovertemplate': hovertemplate,
            'text': all_nustar_obs_reindex['text'][bad],
            'mode': 'markers',
            'showlegend': False,
            'marker': {'color': 'grey', 'symbol': "x-dot", 'size': 3}
         }), row, 1)

    for station, color in zip(['MLD', 'SNG', 'UHI'], ['blue', 'red', 'orange']):
        good = (all_data['station'] == station) & ~bad
        fig.append_trace(go.Scattergl({
            'x': all_data['met'][good].values,
            'y': all_data['offset'][good].values * 1e3,
            'hovertemplate': hovertemplate,
            'text': all_nustar_obs_reindex['text'][good],
            'mode': 'markers',
            'marker': {'color': color, 'size': 3},
            'name': f'Clock offset - {station}'
        }), 1, 1)
        for ydata, row in zip(['residual', 'residual_detrend'], [2, 3]):
            fig.append_trace(go.Scattergl({
                'x': all_data['met'][good].values,
                'y': all_data[ydata][good].values * 1e3,
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
        if bti[-1] < all_data['met'].iloc[0]:
            continue
        if bti[0] > all_data['met'].iloc[-1]:
            continue
        shapes.append(dict(type="rect",
                # x-reference is assigned to the x-values
                xref="x",
                # y-reference is assigned to the plot paper [0,1]
                yref="paper",
                x0=max(bti[0], all_data['met'].iloc[0]),
                y0=0,
                x1=min(bti[1], all_data['met'].iloc[-1]),
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
    html.Div(id='clock-intermediate-value', style={'display': 'none'}),
    html.Div(id='freq-intermediate-value', style={'display': 'none'}),
    html.Div(id='axis-properties', style={'display': 'none'})
])


def find_idxs(array, min, max):
    imin, imax = np.searchsorted(array, [min, max])
    return slice(imin, imax)


@app.callback(
    Output('axis-properties', 'children'),
    [Input('my-figure', 'relayoutData')])
def update_x_ranges(relayoutData):
    global CURRENT_AXES
    if relayoutData is None:
        raise PreventUpdate
    if 'xaxis.autorange' in relayoutData and relayoutData['xaxis.autorange']:
        CURRENT_AXES = default_axes()
    else:
        new_ranges = relayoutData
        for key, value in new_ranges.items():
            CURRENT_AXES[key] = value
    return json.dumps(CURRENT_AXES)


@app.callback(
    Output('my-figure', 'figure'),
    [Input('axis-properties', 'children')])
def display_relayout_data(children):
    relayoutData = json.loads(children)

    all_data, temptable_data, freq_change_data, \
        table_new_complete, gti, all_nustar_obs = global_store('save_all.pickle')

    if relayoutData is not None and 'xaxis.range[0]' in relayoutData:
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
    [Input("refresh-button", "n_clicks")])
def refresh_output(n_clicks):
    global CURRENT_AXES
    log.info(f"Clicked refresh. n_clicks = {n_clicks}")
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate
    else:
        CURRENT_AXES = default_axes()
        if os.path.exists('BAD_POINTS_DB_tmp.dat'):
            os.unlink("BAD_POINTS_DB_tmp.dat")
        return [
            dcc.Graph(figure=plot_dash(*global_store('save_all.pickle'),
                axis_ranges=CURRENT_AXES), id='my-figure',
                hoverData={'points': [{'x': 2e8}]}),]

@app.callback(
    [Output("refresh-button", "n_clicks"),
     Output("clock-intermediate-value", "children"),
     Output("freq-intermediate-value", "children")],
    [Input("bad-data-button", "n_clicks")])
def bad_data_output(n_clicks):
    log.info(f"Clicked bad_data_output. n_clicks = {n_clicks}")
    table = load_and_flag_clock_table(clockfile="latest_clock.dat").to_pandas().to_json()
    freq_change_table = read_freq_changes_table("latest_freq.dat").to_pandas().to_json()

    if n_clicks is not None and n_clicks > 0:
        if os.path.exists('BAD_POINTS_DB_tmp.dat'):
            os.rename('BAD_POINTS_DB_tmp.dat', 'BAD_POINTS_DB.dat')

    return n_clicks, table, freq_change_table


@app.callback(
    Output("bad-data-button", "n_clicks"),
    [Input("recalculate-button", "n_clicks")])
def recalculate_output(n_clicks):
    log.info(f"Clicked recalculate_output. n_clicks = {n_clicks}")
    if n_clicks is None or n_clicks == 0:
        log.info("Preventing plot update")
        raise PreventUpdate

    cache.delete_memoized(global_store, 'save_all.pickle')
    if os.path.exists('save_all.pickle'):
        os.unlink('save_all.pickle')
    return n_clicks


@app.callback(
    Output('selected-data', 'children'),
    [Input('my-figure', 'selectedData')])
def display_selected_data(selectedData):
    if selectedData is None:
        return "No data selected"
    else:
        ALL_BAD_POINTS = np.genfromtxt('BAD_POINTS_DB.dat')
        ALL_BAD_POINTS = np.concatenate((ALL_BAD_POINTS,
            np.array([p['x'] for p in selectedData['points']])))
        ALL_BAD_POINTS.sort()
        ALL_BAD_POINTS = np.unique(ALL_BAD_POINTS)

        np.savetxt('BAD_POINTS_DB_tmp.dat', ALL_BAD_POINTS, fmt='%d')

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


def main():
    app.run_server(debug=True)

import cufflinks as cf
import plotly.graph_objs as go
import plotly.tools as tls
import sklearn.metrics as metrics

from plotly.tools import FigureFactory as FF
from scipy import stats


def append_traces(fig, traces, rows, cols, trace_updates=None):
    if type(traces) == go.Figure:
        traces = traces.data
    
    if type(traces) == go.graph_objs.PlotlyDict:
        traces = [traces]
        rows = [rows]
        cols = [cols]
    else:
        if type(rows) == int:
            rows = [rows] * len(traces)
        if type(cols) == int:
            cols = [cols] * len(traces)
    
    if not trace_updates:
        trace_updates = [None] * len(traces)
    
    if type(trace_updates) != list:
        trace_updates = [trace_updates] * len(traces)
    
    for trace, trace_update, row, col in zip(traces, trace_updates, rows, cols):
        if trace_update:
            trace.update(trace_update)
        fig.append_trace(trace, row, col)


def annotate_metrics(y_true, y_pred, x, y):
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    evs = metrics.explained_variance_score(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    
    return [
        dict(x=x, y=y, xref='paper', yref='paper', showarrow=False, text="Mean absolute error: %.3e" % mae),
        dict(x=x, y=y-0.05, xref='paper', yref='paper', showarrow=False, text="Mean square error: %.3e" % mse),
        dict(x=x, y=y-0.1, xref='paper', yref='paper', showarrow=False, text="Explained variance score: %.3f" % evs),
        dict(x=x, y=y-0.15, xref='paper', yref='paper', showarrow=False, text="r<sup>2</sup> score: %.3f" % r2)
    ]


def line_plot(df, secondary_y=None):
    fig = df.iplot(asFigure=True, width=2, secondary_y=secondary_y)

    fig.layout.update(
        font=dict(
            family='Open Sans'
        ),
        legend=dict(
            orientation='h',
            xanchor='center',
            yanchor='bottom',
            x=0.5,
            y=-0.2
        ),
        xaxis=dict(
            hoverformat='%Y',
            showgrid=False,
            showline=True,
            ticks='outside'
        )
    )
    
    if secondary_y:
        fig.layout.update(
            yaxis1=dict(
                fixedrange=True,
                hoverformat='.2%',
                tickformat='.2%'
            ),
            yaxis2=dict(
                fixedrange=True,
                hoverformat='d'
            )
        )
    else:
        fig.layout.update(
            yaxis=dict(
                fixedrange=True,
                hoverformat='d',
                tickformat='d'
            )
        )

    return fig


def scatter_plot_matrix(df, colours=None, y1_title='', y2_title=''):
    assert df.shape[1] == 2
    
    fig = tls.make_subplots(4, 2, specs=[[{'rowspan': 2, 'colspan': 2}, None], [None, None], [{}, {}], [{}, {}]], print_grid=False)
    append_traces(fig, df.iplot(asFigure=True, width=2), 1, 1, trace_updates=dict(showlegend=False))
    append_traces(fig, FF.create_scatterplotmatrix(df, diag='histogram'), [3, 3, 4, 4], [1, 2, 1, 2])
    
    # Add best fit lines
    if colours == None:
        colours = cf.get_colorscale(cf.get_config_file()['colorscale'])
    
    def fit(x, y):
        m, c, r, p, e = stats.linregress(x, y)
        return go.Scatter(x=x, y=m*x + c, showlegend=False, marker=dict(color=colours[2][1]),
                          name='Best fit', text='y = %.2E x + %.2E<br>r<sup>2</sup> = %.2f' % (m, c, r))
    
    df.dropna(inplace=True)
    fig.append_trace(fit(df.ix[:, 1].values, df.ix[:, 0].values), 3, 2)
    fig.append_trace(fit(df.ix[:, 0].values, df.ix[:, 1].values), 4, 1)

    # Set a shared Y axis for the top plot
    fig.layout['yaxis6'] = dict(
        overlaying='y1',
        side='right'
    )
    fig.data[1].yaxis='y6'
    
    # Set layout options
    fig.layout.update(
        font=dict(
            family='Open Sans'
        ),
        xaxis=dict(
            hoverformat='%Y',
            showgrid=False,
            showline=True,
            ticks='outside'
        ),
        xaxis2=dict(
            fixedrange=True,
            hoverformat='.2%',
            tickformat='.2%'
        ),
        xaxis3=dict(
            fixedrange=True
        ),
        xaxis4=dict(
            fixedrange=True,
            hoverformat='.2%',
            tickformat='.2%',
            title=y1_title
        ),
        xaxis5=dict(
            fixedrange=True,
            title=y2_title
        ),
        yaxis1=dict(
            fixedrange=True,
            hoverformat='.2%',
            tickformat='.2%',
            title=y1_title
        ),
        yaxis2=dict(
            fixedrange=True,
            title=y1_title
        ),
        yaxis3=dict(
            fixedrange=True,
            hoverformat='.2%',
            tickformat='.2%'
        ),
        yaxis4=dict(
            fixedrange=True,
            title=y2_title
        ),
        yaxis5=dict(
            fixedrange=True
        ),
        yaxis6=dict(
            fixedrange=True,
            hoverformat='d',
            title=y2_title
        )
    )
    
    # Match histogram colours to line colours
    fig.data[2].marker.update(color=colours[0][1])
    fig.data[3].marker.update(color='black', opacity=0.4)
    fig.data[4].marker.update(color='black', opacity=0.4)
    fig.data[5].marker.update(color=colours[1][1])
    
    # Set hover text on scatter plot matrix
    fig.data[2].update(name='Count')
    fig.data[3].update(name=df.columns[0])
    fig.data[4].update(name=df.columns[1])
    fig.data[5].update(name='Count')
    
    # Set overall size
    fig.layout.update(height=800)
    
    return fig

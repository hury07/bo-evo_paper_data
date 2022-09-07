
from posixpath import dirname
from typing import Dict, Optional, Sequence, Union
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score, auc
import matplotlib.ticker as mtick
import seaborn as sns

import sys
from os.path import dirname, join, isfile
sys.path.append(dirname(__file__))
if isfile(join(dirname(__file__), 'landscape_class.py')):
    from landscape_class import Protein_Landscape
else:
    raise ImportError


def violinplot(
    ax,
    x=None,
    y=None,
    data=None,
    face_color: Optional[Sequence[str]] = None,
    edge_color: str = "gray",
    alpha: float = 0.8,
):

    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    if (data is None) and (x is None) and (y is not None):
        data = [sorted(y)]
        labels = []
        vert = True
    elif (data is None) and (x is not None) and (y is None):
        data = [sorted(x)]
        labels = []
        vert = False
    else:
        labels = sorted(data[x].unique())
        data = [data[data[x] == i][y].values for i in labels]
        data = [sorted(item) for item in data]
        vert = True

    parts = ax.violinplot(
        data, positions=range(len(labels)),
        showmeans=False, showmedians=False, showextrema=False,
        vert=vert,
    )
    n_parts = len(parts['bodies'])
    face_color = sns.color_palette(
    )[:n_parts] if face_color is None else face_color
    for pc, face in zip(parts['bodies'], face_color):
        pc.set_facecolor(face)
        pc.set_edgecolor(edge_color)
        pc.set_alpha(alpha)
    quartiles = np.array([np.percentile(item, [25, 50, 75]) for item in data])
    quartile1, medians, quartile3 = quartiles.transpose()
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
    inds = range(len(medians))

    if vert:
        ax.scatter(inds, medians, marker='o', color='white', s=5, zorder=3)
        ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax.vlines(inds, whiskersMin, whiskersMax,
                  color='k', linestyle='-', lw=1)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
    else:
        ax.scatter(medians, inds, marker='o', color='white', s=5, zorder=3)
        ax.hlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax.hlines(inds, whiskersMin, whiskersMax,
                  color='k', linestyle='-', lw=1)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)


def barplot_count(
    ax,
    x=None,
    y=None,
    data=None
):
    if x == None:
        df = data.groupby([y]).count()
        inp1 = df.iloc[:, 0].values
        inp2 = df.index.values
        orient = 'h'
    if y == None:
        df = data.groupby([x]).count()
        inp1 = df.index.values
        inp2 = df.iloc[:, 0].values
        orient = 'v'
    #
    sns.barplot(x=inp1, y=inp2, orient=orient)
    if orient == 'h':
        for i, v in zip(inp1, inp2):
            ax.text(i, v-1, str(i))
    else:
        for i, v in zip(inp1, inp2):
            ax.text(i, v, str(v))


def read_data(run_dir, prefix="run"):
    data = []
    rep = 0
    for fname in glob.glob(f'{run_dir}/{prefix}*.csv'):
        # Skip metadata in header
        with open(fname) as f:
            next(f)
            df = pd.read_csv(f)
            df['rep'] = rep
            rep += 1
            data.append(df)
    data = pd.concat(data)
    return data


def overall(
    ax,
    names_and_dirs: Dict[Union[int, str], str],
    score: str = 'true_score',
    method: str = 'max',
    face_colors: Optional[Sequence[str]] = None,
):
    data = []
    keys = sorted(names_and_dirs.keys())
    for name in keys:
        run_dir = names_and_dirs[name]
        df = read_data(run_dir)
        df = df.groupby('rep').agg({score: method})
        df['name'] = name
        data.append(df)
    data = pd.concat(data)
    face_colors = sns.color_palette()[:len(
        names_and_dirs)] if face_colors is None else face_colors
    violinplot(ax, x='name', y=score, data=data, face_color=face_colors)


def cummax_round(
    ax,
    names_and_dirs: Dict[Union[int, str], str],
    score: str = 'true_score',
    method: str = 'max',
    xaxis_is_round: bool = True,
    linestyles: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
):
    linestyles = ['-'] * \
        len(names_and_dirs) if linestyles is None else linestyles
    colors = sns.color_palette()[:len(
        names_and_dirs)] if colors is None else colors
    keys = sorted(names_and_dirs.keys())
    for name, linestyle, color in zip(keys, linestyles, colors):
        run_dir = names_and_dirs[name]
        df = read_data(run_dir)
        df = df.groupby(['rep', 'round', 'measurement_cost']
                        ).agg({score: method}).unstack(0)
        df = df.cummax(axis=0)
        df = df.stack().reset_index(['round', 'measurement_cost'])
        if xaxis_is_round:
            sns.lineplot(ax=ax,
                         x='round', y=score, data=df, label=name, linewidth=1,
                         linestyle=linestyle, color=color,
                         )
        else:
            sns.lineplot(ax=ax,
                         x='measurement_cost', y=score, data=df, label=name, linewidth=1,
                         linestyle=linestyle, color=color,
                         )
    ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))


def score_round(
    ax,
    names_and_dirs: Dict[Union[int, str], str],
    score: str = 'true_score',
    method: str = 'max',
    max_round: int = 10,
    linestyles: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
):
    linestyles = ['-'] * \
        len(names_and_dirs) if linestyles is None else linestyles
    colors = sns.color_palette()[:len(
        names_and_dirs)] if colors is None else colors
    keys = sorted(names_and_dirs.keys())
    for name, linestyle, color in zip(keys, linestyles, colors):
        run_dir = names_and_dirs[name]
        df = read_data(run_dir)
        max_round = min(max_round, df['round'].max())
        line = []
        for round in range(max_round+1):
            data = df[df['round'] <= round]
            data = data.groupby(['rep']).agg({score: method})
            data = data.median().values[0]
            line.append([round, data])
        df = pd.DataFrame(line, columns=['round', score])
        sns.lineplot(ax=ax,
                     x='round', y=score, data=df, label=name, linewidth=1,
                     linestyle=linestyle, color=color,
                     )
    ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))


def violin_and_cummax_round(
    ax,
    path,
    score: str = 'true_score',
    print_value: bool = True,
    rep: Optional[int] = None,
):
    df = read_data(path)
    if rep is not None:
        df = df[df['rep'] == rep]
    n_round = df['round'].max()
    violinplot(ax, x='round', y=score, data=df,
               face_color=[sns.color_palette()[0]] +
               sns.color_palette()[:n_round+1],
               )

    data = df.groupby(['rep', 'round', 'measurement_cost']
                      ).agg({score: 'max'}).unstack(0)
    data = data.cummax(axis=0)
    data = data.stack().reset_index(['round', 'measurement_cost'])
    sns.lineplot(ax=ax, x='round', y=score, data=data,
                 linewidth=1, color=sns.color_palette()[6])
    if print_value:
        print(data)


def model_performance(
    ax,
    path,
    method: str = 'NDCG',
    face_colors: Optional[Sequence[str]] = None,
):
    methods = {
        'Spearman': 'spearman',
        'Pearson': 'pearson',
        'NDCG': lambda a, b: ndcg_score(np.expand_dims(a, axis=0), np.expand_dims(b, axis=0)),
    }
    df = read_data(path)
    n_round = df['round'].max()
    df = df[(df['round'] != 0) & (df['round'] != 1)]
    df = df.groupby(['rep', 'round'])[['true_score', 'model_score']].corr(
        method=methods[method]).unstack()
    df = df[('true_score', 'model_score')].reset_index('round')
    face_colors = sns.color_palette(
    )[1:n_round+1] if face_colors is None else face_colors
    violinplot(ax, x='round', y=('true_score', 'model_score'),
               data=df, face_color=face_colors)


def model_performance_curve(
    ax,
    names_and_dirs: Dict[Union[int, str], str],
    method: str = 'NDCG',
    linestyles: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
):
    methods = {
        'Spearman': 'spearman',
        'Pearson': 'pearson',
        'NDCG': lambda a, b: ndcg_score(np.expand_dims(a, axis=0), np.expand_dims(b, axis=0)),
    }
    linestyles = ['-'] * \
        len(names_and_dirs) if linestyles is None else linestyles
    colors = sns.color_palette()[:len(
        names_and_dirs)] if colors is None else colors
    keys = sorted(names_and_dirs.keys())
    for name, linestyle, color in zip(keys, linestyles, colors):
        run_dir = names_and_dirs[name]
        df = read_data(run_dir)
        df = df[(df['round'] != 0) & (df['round'] != 1)]
        df = df.groupby(['rep', 'round'])[['true_score', 'model_score']].corr(
            method=methods[method]).unstack()
        df = df[('true_score', 'model_score')]
        df = df.groupby('round').agg('median')
        sns.lineplot(ax=ax, data=df, linewidth=1, label=name,
                     linestyle=linestyle, color=color,
                     )
        # break
    ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))


def success_ratio_plot(
    ax,
    names_and_dirs: Dict[Union[int, str], str],
    level: float = 100.,
    global_max: float = 1.0,
    global_min: float = 0.0,
    linestyles: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    print_rounds: Optional[Sequence[int]] = None,
):
    level = (global_max - global_min) * level / 100 + global_min
    linestyles = ['-'] * \
        len(names_and_dirs) if linestyles is None else linestyles
    colors = sns.color_palette()[:len(
        names_and_dirs)] if colors is None else colors
    keys = sorted(names_and_dirs.keys())
    for name, linestyle, color in zip(keys, linestyles, colors):
        run_dir = names_and_dirs[name]
        df = read_data(run_dir)
        n_reps = len(df['rep'].unique())
        data = df.groupby(['rep', 'round']).agg({'true_score': 'max'})
        data = data['true_score'].apply(lambda x: 1 if x >= level else 0)
        data = data.unstack().cummax(axis=1).sum(axis=0) / n_reps * 100
        sns.lineplot(ax=ax, data=data, linewidth=1, label=name,
                     linestyle=linestyle, color=color,
                     )
        print_rounds = print_rounds if print_rounds is not None else [
            len(data) - 1]
        for i in print_rounds:
            print('Round {0}-{1}: {2:.2%}'.format(i, name, data.loc[i]/100))
    ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())


def combinations(source: list, n: int) -> list:
    if len(source) == n:
        return [source]
    if n == 1:
        ans = []
        for i in source:
            ans.append([i])
        return ans
    ans = []
    for each_list in combinations(source[1:], n-1):
        ans.append([source[0]]+each_list)
    for each_list in combinations(source[1:], n):
        ans.append(each_list)
    return ans


def RS(
    landscapes,
    files: Optional[Sequence[str]] = None,
    subgraph_size: int = 4,
    amino_acids: str = 'ACDEFGHIKLMNPQRSTVWY',
):
    def compute_rs(comps, aa, df):
        cond = pd.concat(
            [
                df['seq'].apply(lambda x: True if x[pos] == aa else False)
                for pos in comps
            ],
            axis=1
        ).all(axis=1)
        subgraph = df[cond == True].to_numpy()
        landscape_protein = Protein_Landscape(data=subgraph)
        return landscape_protein.RS_ruggedness

    rs = []
    for landscape, file in zip(landscapes, files):
        if os.path.exists(file):
            rs_subgraphs = pd.read_csv(file)
        else:
            data = landscape._sequences
            pos = list(range(len(list(data.keys())[0])))
            assert len(pos) >= subgraph_size
            ###
            if len(pos) == subgraph_size:
                subgraph = pd.DataFrame(list(data.items())).to_numpy()
                landscape_protein = Protein_Landscape(data=subgraph)
                rs_subgraphs = [landscape_protein.RS_ruggedness]
            else:
                pos_set = set(pos)
                pos_combs = combinations(pos, subgraph_size)
                pos_comps = [sorted(pos_set.difference(x)) for x in pos_combs]
                rs_subgraphs = []
                df = pd.DataFrame(list(data.items()), columns=['seq', 'score'])

                for comps in pos_comps:
                    for aa in amino_acids:
                        cond = pd.concat(
                            [
                                df['seq'].apply(
                                    lambda x: True if x[pos] == aa else False)
                                for pos in comps
                            ],
                            axis=1
                        ).all(axis=1)
                        subgraph = df[cond == True].to_numpy()
                        landscape_protein = Protein_Landscape(data=subgraph)
                        rs_subgraphs.append(landscape_protein.RS_ruggedness)
            rs_subgraphs = pd.DataFrame(rs_subgraphs)
            if files is not None:
                rs_subgraphs.to_csv(file, index=False)
        rs.append(rs_subgraphs.mean()[0])
    return np.array(rs)


def AUC(
    names_and_dirs: Dict[Union[int, str], str],
    round: int = 10,
    score: str = 'true_score',
    method: str = 'max',
):
    scores = []
    keys = sorted(names_and_dirs.keys())
    for name in keys:
        run_dir = names_and_dirs[name]
        df = read_data(run_dir)
        df = df.groupby(['rep', 'round']).agg({score: method}).unstack(0)
        df = df.cummax(axis=0)
        df = df.median(axis=1)
        baseline = df[0]
        df = df.apply(lambda x: (x - baseline) / (1.0 - baseline))
        scores.append(auc(df.index[:round+1], df[:round+1]))
    return np.array(scores)


def success_ratio(
    names_and_dirs: Dict[Union[int, str], str],
    level: float = 100.,
    global_max: float = 1.0,
    global_min: float = 0.0,
    rounds: Optional[Sequence[int]] = None,
):
    level = (global_max - global_min) * level / 100 + global_min

    ret = pd.DataFrame()
    for (name, run_dir) in names_and_dirs.items():
        df = read_data(run_dir)
        n_reps = len(df['rep'].unique())
        data = df.groupby(['rep', 'round']).agg({'true_score': 'max'})
        data = data['true_score'].apply(lambda x: 1 if x >= level else 0)
        data = data.unstack().cummax(axis=1).sum(axis=0) / n_reps * 100
        rounds = [len(data) - 1] if rounds is None else rounds
        ret[name] = data.loc[rounds]
    return ret.transpose()


def eval_model(
    ax,
    names_and_dirs: Dict[Union[int, str], str],
    prefix: str = 'eval',
    method: str = 'Spearman',
    linestyles: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
):
    methods = {
        'Spearman': 'spearman',
        'Pearson': 'pearson',
        'NDCG': lambda a, b: ndcg_score(np.expand_dims(a, axis=0), np.expand_dims(b, axis=0)),
    }
    linestyles = ['-'] * \
        len(names_and_dirs) if linestyles is None else linestyles
    colors = sns.color_palette()[:len(
        names_and_dirs)] if colors is None else colors
    for (name, run_dir), linestyle, color in zip(names_and_dirs.items(), linestyles, colors):
        df = read_data(run_dir, prefix=prefix)
        df = df[df['round'] != 0]
        df = df.groupby(['rep', 'round'])[['true_score', 'model_score']].corr(
            method=methods[method]).unstack()
        df = df[('true_score', 'model_score')]
        df = df.groupby('round').agg('median')
        df = df.fillna(0)
        sns.lineplot(ax=ax, data=df, linewidth=1, label=name,
                     linestyle=linestyle, color=color,
                     )
    ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))

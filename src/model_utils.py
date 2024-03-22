import copy
import numpy as np
import lingam
import graphviz
import netcomp as nc
from tqdm.notebook import tqdm
from lingam.utils import make_prior_knowledge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import DataStructs
import rdkit
import matplotlib.patches as patches
from matplotlib.path import Path

def causal_inference(data, target_col=-1, scale_data=False, method="DirectLiNGAM"):
    """_summary_

    Args:
        data (_type_): _description_
        target_col (int, optional): _description_. Defaults to -1.
        scale_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if scale_data:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    prior = make_prior_knowledge(n_variables=data.shape[1], sink_variables=[target_col])
    if method == "DirectLiNGAM":
        model = lingam.DirectLiNGAM(prior_knowledge=prior)
    elif method == "RESIT":
        model = lingam.RESIT(RandomForestRegressor(max_depth=15, random_state=0))
    elif method == "RESIT_PRIOR":
        model = lingam.RESIT_PRIOR(
            RandomForestRegressor(max_depth=15, random_state=0), prior_knowledge=prior
        )
    model.fit(data)
    return model


def make_graph(adjacency_matrix, labels=None):
    """_summary_

    Args:
        adjacency_matrix (_type_): _description_
        labels (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    idx = np.abs(adjacency_matrix) > 0.01
    dirs = np.where(idx)
    d = graphviz.Digraph(engine="dot")
    names = labels if labels else [f"x{i}" for i in range(len(adjacency_matrix))]
    for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
        d.edge(names[from_], names[to], label=f"{coef:.2f}")
    return d


def select_features(
    dataframe,
    nkeep=None,
    epsilon_keep=None,
    target_column="polarizability",
    ignore_columns=["smiles", "dipole"],
):
    """_summary_

    Args:
        dataframe (_type_): _description_
        nkeep (int, optional): _description_. Defaults to 6.
        target_column (str, optional): _description_. Defaults to 'polarizability'.
        ignore_columns (list, optional): _description_. Defaults to ['smiles', 'dipole'].

    Returns:
        _type_: _description_
    """
    # fabulous error checking
    if (nkeep is not None) and (epsilon_keep is not None):
        print("one of nkeep and epsilon_keep must be None. exiting.")
        return

    if (nkeep is None) and (epsilon_keep is None):
        print("one of nkeep and epsilon_keep must be specified. exiting.")
        return

    column_names = list(dataframe.columns)
    X_columns = copy.copy(column_names)
    for ignore in ignore_columns:
        X_columns.remove(ignore)
    print("Columns used for feature selection", X_columns)
    target_column_ind = X_columns.index(target_column)
    X = dataframe[X_columns].to_numpy()
    model = causal_inference(X, target_col=target_column_ind)
    ranked_features = np.argsort(np.abs(model.adjacency_matrix_.mean(axis=0)))[::-1]
    ranked_weights = np.sort(np.abs(model.adjacency_matrix_.mean(axis=0)))[::-1]
    if nkeep is not None:
        feature_inds = ranked_features[:nkeep]
    elif epsilon_keep is not None:
        ranked_weights = ranked_weights / np.sum(ranked_weights)
        print(ranked_weights)
        feature_inds = ranked_features[ranked_weights > epsilon_keep]
        print(feature_inds)
    X_columns_keep = [X_columns[ind] for ind in feature_inds]
    X_columns_keep.append("dipole")
    print("Columns kept after feature selection", X_columns_keep)
    return (
        dataframe[X_columns_keep],
        X_columns_keep,
        model.adjacency_matrix_,
        ranked_weights,
    )


def fit_random_forest(x, fit_small_values=True, return_pred_values=True):
    """_summary_

    Args:
        dataframe (_type_): _description_
        nkeep (int, optional): _description_. Defaults to 6.
        target_column (str, optional): _description_. Defaults to 'polarizability'.
        ignore_columns (list, optional): _description_. Defaults to ['smiles', 'dipole'].

    Returns:
        _type_: _description_
    """
    np.random.seed = 0
    x_fit = x[:, :-1]
    y_fit = x[:, -1]

    if fit_small_values:
        ind_small = y_fit < 10
        x_fit = x_fit[ind_small, :]
        y_fit = y_fit[ind_small]

    inds = np.arange(x_fit.shape[0])
    ntrain = int(0.8 * len(inds))
    np.random.shuffle(inds)
    train_inds = inds[:ntrain]
    test_inds = inds[ntrain:]

    x_train = x_fit[train_inds, :]
    y_train = y_fit[train_inds]

    regr = RandomForestRegressor(max_depth=15, random_state=0)
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_fit[test_inds, :])
    test_scores = score_regressor(y_pred, y_fit[test_inds])
    if return_pred_values:
        return regr, test_scores, y_pred, y_fit[test_inds]

    return regr, test_scores


def score_regressor(y_pred, y_true):
    """_summary_

    Args:
        data (_type_): _description_
        target_col (int, optional): _description_. Defaults to -1.
        scale_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    return {
        "R^2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(np.mean((y_pred - y_true) ** 2)),
    }


def score_dataframe(df, global_adj):
    """_summary_

    Args:
        data (_type_): _description_
        target_col (int, optional): _description_. Defaults to -1.
        scale_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    target_column_ind = list(df.drop(columns="smiles").columns).index("dipole")
    X = df.drop(columns="smiles").to_numpy()
    causal_model = causal_inference(X, target_col=target_column_ind)
    _, score, _, _ = fit_random_forest(X)
    gscore = nc.lambda_dist(causal_model.adjacency_matrix_, global_adj)
    return score, gscore, causal_model.adjacency_matrix_


def concatenate_samples(source, destination, nsamples):
    """_summary_

    Args:
        data (_type_): _description_
        target_col (int, optional): _description_. Defaults to -1.
        scale_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    source.reset_index(drop=True)
    rand_id = np.arange(len(source))
    np.random.shuffle(rand_id)
    id0 = rand_id[:nsamples]
    destination = destination.copy()
    return pd.concat([destination, source.loc[id0]], ignore_index=True)


def run_random_loop(
    all_df, random_df, n_iter, global_adj, n_samples=100, mode="subset"
):
    """_summary_

    Args:
        data (_type_): _description_
        target_col (int, optional): _description_. Defaults to -1.
        scale_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    scores_, gscores_, choices_, adjs_, subsets_ = [], [], [], [], []
    for _ in tqdm(range(n_iter)):
        if mode == "subset":
            rand_ind = np.random.randint(1, 4)
            random_df = concatenate_samples(all_df[rand_ind], random_df, n_samples)
        elif mode == "full":
            random_df = concatenate_samples(all_df[0], random_df, n_samples)
        score, gscore, adj = score_dataframe(random_df, global_adj)
        scores_.append(score)
        adjs_.append(adj)
        gscores_.append(gscore)
    return scores_, gscores_, choices_, adjs_, subsets_, random_df


def run_active_loop(all_df, active_df, n_iter, global_adj, n_samples=100):
    """_summary_

    Args:
        data (_type_): _description_
        target_col (int, optional): _description_. Defaults to -1.
        scale_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # get initial scores
    scores_, gscores_, choices_, adjs_, subsets_ = [], [], [], [], []
    for _ in tqdm(range(n_iter)):
        test_dfs, adj_mats, r2_scores, graph_scores = [], [], [], []
        for k in range(1, len(all_df)):
            df_now = all_df[k]
            test_df = concatenate_samples(df_now, active_df, n_samples)
            score, gscore, adj = score_dataframe(test_df, global_adj)
            test_dfs.append(test_df)
            adj_mats.append(adj)
            r2_scores.append(score)
            graph_scores.append(gscore)

        # get the scores.
        idx = np.argmin(graph_scores)
        active_df = test_dfs[idx]
        scores_.append(r2_scores[idx])
        adjs_.append(adj_mats[idx])
        gscores_.append(graph_scores[idx])
        subsets_.append(idx)
    return scores_, gscores_, choices_, adjs_, subsets_, active_df


def get_data_fingerprints(data, smiles_column):
    """_summary_

    Args:
        data (_type_): _description_
        target_col (int, optional): _description_. Defaults to -1.
        scale_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    all_fp = []
    for mol in data[smiles_column].to_list():
        all_fp.append(ECFP_from_smiles(mol))
    return all_fp

def get_tanimoto_similarities(fingerprintsA, fingerprintsB):
    """_summary_

    Args:
        data (_type_): _description_
        target_col (int, optional): _description_. Defaults to -1.
        scale_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    return [DataStructs.TanimotoSimilarity( fingerprintsA[i], fingerprintsB[i]) for i in range(len(fingerprintsA))]

def ECFP_from_smiles(smiles, R=2, L=2**10, use_features=False, use_chirality=False):
    """_summary_

    Args:
        data (_type_): _description_
        target_col (int, optional): _description_. Defaults to -1.
        scale_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    molecule = AllChem.MolFromSmiles(smiles)
    feature_list = AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        radius=R,
        nBits=L,
        useFeatures=use_features,
        useChirality=use_chirality,
    )
    return feature_list

def get_top_k_close_molecules(k, distances, reference_df, original_df, include_dipole=True):
    """_summary_

    Args:
        data (_type_): _description_
        target_col (int, optional): _description_. Defaults to -1.
        scale_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    closest_smiles = []
    closest_smiles_dipole = []
    for molecule_id in range(distances.shape[0]):
        closest_k_distances = np.sort(distances[molecule_id,:])[:k]
        closest_k = np.argsort(distances[molecule_id,:])[:k]
        if include_dipole:
            closest_smiles.append( [original_df.loc[molecule_id].smiles] +
                            [ reference_df.loc[j].smiles for j in closest_k] + [ reference_df.loc[j].dipole for j in closest_k] + list(closest_k_distances) )
            column_titles = ['original'] + [f'closest_{j}' for j in range(k)] + [f'closest_{j}_dipole' for j in range(k)] + [f'closest_{j}_distance' for j in range(k)]
    
    return pd.DataFrame(closest_smiles, columns=column_titles)

def plot_path_between(pointA, pointB, ax):
    before = pointA
    after = pointB

    path_data = [
        (Path.MOVETO, pointA),
        (Path.CURVE4, pointB )
        ]

    verts = [pointA,
             (pointA+pointB)/2+np.random.randn()*.5,
            pointB] 

    codes = [Path.MOVETO, 
             Path.CURVE3,
             Path.CURVE3
            ]
             
    # codes, verts = zip(*path_data)
    path = Path(verts, codes)
    patch = patches.FancyArrowPatch(path=path, facecolor='k', arrowstyle="-|>,head_length=5,head_width=3")
    ax.add_patch(patch)

    return ax

def getcolordensity_contour(xdata, ydata):
    """_summary_

    Args:
        data (_type_): _description_
        target_col (int, optional): _description_. Defaults to -1.
        scale_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    nbin = 20
    hist2d, xbins_edge, ybins_edge = np.histogram2d(x=xdata, y=ydata, bins=[nbin, nbin])
    xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])
    ybin_cen = 0.5 * (ybins_edge[0:-1] + ybins_edge[1:])
    BCTY, BCTX = np.meshgrid(ybin_cen, xbin_cen)
    hist2d = hist2d / np.amax(hist2d)

    return BCTX, BCTY, hist2d

def draw_molecules(molecules, prefix, molsPerRow=3, maxMols=100):
    """_summary_

    Args:
        data (_type_): _description_
        target_col (int, optional): _description_. Defaults to -1.
        scale_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    best_molecules = [rdkit.Chem.MolFromSmiles(smiles) for smiles in molecules]
    best_svg = rdkit.Chem.Draw.MolsToGridImage(
        best_molecules,
        molsPerRow=molsPerRow,
        subImgSize=(300, 300),
        useSVG=True,
        maxMols=maxMols,
    )
    with open(f"{prefix}.svg", "w") as f:
        f.write(best_svg.data)

    best_png = rdkit.Chem.Draw.MolsToGridImage(
        best_molecules,
        molsPerRow=molsPerRow,
        subImgSize=(300, 300),
        returnPNG=True,
        maxMols=maxMols,
    )
    with open(f"{prefix}.png", "wb") as f:
        f.write(best_png.data)

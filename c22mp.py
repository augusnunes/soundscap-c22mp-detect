"""This code is from Matrix Profile XXIX: C22MP: Fusing catch22 and 
the Matrix Profile to Produce an Efficient and Interpretable Anomaly Detector. 
Sadaf Tafazoli, Yue Lu, Renjie Wu, Thirumalai Srinivas, Hannah Dela Cruz, Ryan Mercer, 
and Eamonn Keogh.ICDM 2023."""

import numpy as np
import pycatch22
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


def scoring_function(loc, anomaly_start, anomaly_end, pad=100):
    L = anomaly_end - anomaly_start + 1
    lower_bound = min(anomaly_start - L, anomaly_start - pad)
    upper_bound = max(anomaly_end + L, anomaly_end + pad)
    if lower_bound < loc and loc < upper_bound:
        return True
    else:
        return False


def cal_catch22(x):
    transformed_data = pycatch22.catch22_all(x)["values"]
    return transformed_data


def cal_feature_profiles(ts, win, step=1):
    transformed_seq = []
    for i in np.arange(0, len(ts) - win + 1, step):
        # for i in np.arange(0,len(ts)-win+1, step):
        new_subseq = ts[i : i + win]
        tr_new_subseq = cal_catch22(new_subseq)
        transformed_seq.append(tr_new_subseq)
    transformed_seq = np.asarray(transformed_seq)
    return transformed_seq


def ffill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out


def bfill(arr):
    return ffill(arr[:, ::-1])[:, ::-1]


def get_feature_profiles(ts, win, step=1):
    feature_profiles = cal_feature_profiles(ts, win, step)
    feature_profiles = ffill(feature_profiles)
    feature_profiles = bfill(feature_profiles)
    feature_profiles_scaled = MinMaxScaler().fit_transform(feature_profiles)
    return feature_profiles_scaled


def eudis1(v1, v2):
    return np.linalg.norm(v1 - v2)


def init_bsf(x, start_idx, exclude_zone):
    rep = x[start_idx]
    normals = []
    for i in np.arange(start_idx - 1 - exclude_zone, -1, -1):
        d = eudis1(rep, x[i])
        normals.append(d)
    return np.quantile(normals, 0.05)  # np.min(normals)


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def left_c22_mp(
    X,
    start_idx,
    win,
    earlly_abaondon=True,
    get_c22_mp_idxs=False,
    you_can_do_better=False,
    dynamic_bsf_update=False,
    look_back_window=None,
    weights=None,
    mad=None,
    exclude_zone=None,
    verbose=False,
):
    """Left_C22_MP.

    Parameters
    ----------
    X : array
      the feature profile array to calculate its left_c22_MP.

    start_idx : int
      The index of the first sample in test data.
      note- since we use all the samples before start_idx as traing data, so this number also tells us the size of the traing data.

    exclude_zone : int
      It is used to avoid trivial matches.

    win:  int
    window size is used to calculate feature profiles.

    earlly_abaondon : boolean
      Switch to set if we want to have earlly abondon or not.
      if set True ORR algorithm will be run otherwise the Brute Force algorithm will be run.

    get_c22_mp_idxs : boolean
      if set true it returns the left_c22_mp indexs. the index of the subsequnce that its distance is used to fill left_c22_mp.

    you_can_do_better : boolean
      if set true, it forces to:
      for the current subsequnc we compare it atleast with n previouse subsequnces.
      by default n is set to be eqaul to the size of the training data.

    dynamic_bsf_update : boolean
      if set false it updates the bsf as follow:
      if for the current subsequence we had to go back all the way to the begining to find a neighbor for it that
      has distance less than bsf, then we use the value of the 1NN of curr subsequnec to fill left_c22_mp and update bsf.
      if set true it updates the bsf dynamically:
      if for the current subsequence we had to go back all the way to the begining to find a neighbor for it that
      has distance less than bsf, then we use the value of the 1NN of curr subsequnec to only fill left_c22_mp.
      We use the median of the n recent values in the left_c22_mp that we have made so far to update the bsf.
      the n is equal to win*10.

    look_back_window: int
      it is used to set the look back window.
      note - set look_back_window to None if you do not want to have speed up for earlly abondoning.
      if set to x, for each subsequence we allow the algorithm to go back only x steps to find a neighbor (that
      has distance less than bsf).
      intuitivly - we are saying you dont need to go all the way to begining to to proof this subseq is discordia,
      we say go back enough and if we didnt see any matches then that is enough proof. use the lbsf distance to fill c22mp.

    weights : array
      the array used to weight features.


    mad: int
      minmum alarm duration

    verbose : boolean
      Verbosity mode.


    Returns
    -------
    left_C22_MP : list (1D array)
      left_c22_mp of the give feature profiles.

    anomaly_idxs : list (1D array)
      indexes of the subsequneces that we had to go back all the way to begining to find a neighbor for them that has distance less than bsf.
      note -  we consider these subsequnse anomaly as we could not find easily/fast a neighbor for them.

    n_skipped : int
      number of times that we had earlly abondon.
      or in other world the number of subsequnces that we could find a neighbor for them that has distance less than bsf without the need to
      go back all the way to begining.

    n_full_back : int
      number of times that we did not have earlly abondon.
      or in other world the number of subsequnces that we had to go all the way to begining to find a neighbor for them which has distance less bsf.

    bsf_mem : list
      a list containing all the values of bsfs used during the left_c22_mp calculation.

    left_c22_mp_idxs : list (1D array)
      the left_c22_mp indexs. the index of the subsequnce that its distance is used to fill left_c22_mp.

    bsf_mem_idx: list
      a list containing the index of subsquences that made the algorithm to update the bsf value.

    """
    if exclude_zone is None:
        exclude_zone = win // 2

    if weights is not None:
        if np.ndim(weights) == 1:
            not_important_features_idx = []
            for i, w in enumerate(weights):
                if np.isclose(w, 0.0):
                    not_important_features_idx.append(i)
            important_features_idx = np.arange(weights.shape[0])
            important_features_idx = np.delete(
                important_features_idx, not_important_features_idx
            )
            X = np.multiply(
                X[:, important_features_idx], weights[important_features_idx]
            )
        else:
            X = np.multiply(X, weights)

    n_subseq = len(X)
    bsf = init_bsf(X, start_idx, exclude_zone)
    if verbose:
        print("init bsf", bsf)
    look_back_to_this_point = 0
    left_C22_MP = np.zeros((X.shape[0],))
    n_skipped = 0
    n_full_back = 0
    anomaly_idxs = {}
    bsf_mem = [bsf]
    bsf_mem_idx = [0]
    len_train = n_subseq - start_idx
    left_c22_mp_idxs = np.zeros((X.shape[0],))
    if earlly_abaondon:  # Table 1 ORR
        for i in np.arange(start_idx, n_subseq):
            curr = X[i]
            lbsf = np.inf
            lbsf_idx = None
            for j in np.arange(i - 1 - exclude_zone, -1, -1):
                if look_back_window is not None:
                    look_back_to_this_point = i - (look_back_window)

                d = eudis1(curr, X[j])

                if d < lbsf:
                    lbsf = d
                    lbsf_idx = j

                if d < bsf and j > look_back_to_this_point:
                    n_skipped += 1
                    if you_can_do_better:
                        continue
                    break

                elif d >= bsf and j > look_back_to_this_point:
                    continue

                elif d >= bsf and j <= look_back_to_this_point:
                    n_full_back += 1
                    if (
                        dynamic_bsf_update and i > start_idx + 2
                    ):  # we check if lenght left_C22_MP that we have  made so far is bigger than 2 then we calculate its median
                        temp = np.median(left_C22_MP[-(win * 10) :])
                        if temp <= np.quantile(
                            bsf_mem, 0.7
                        ):  # we want to have earlly abaondon, if we dont check that temp is big enough, we will end up with a very small bsf value, and basically we will have full left mp
                            pass
                        else:
                            bsf = temp
                    else:
                        bsf = lbsf
                    anomaly_idxs[i] = j
                    bsf_mem.append(bsf)
                    bsf_mem_idx.append(i)
                    if verbose:
                        print("found anomaly at {}, update bsf to:{}".format(i, bsf))
                    break

            left_C22_MP[i] = lbsf
            if get_c22_mp_idxs:
                left_c22_mp_idxs[i] = lbsf_idx

    else:  # Table 3 - Brute Force
        for i in np.arange(start_idx, n_subseq):
            curr = X[i]
            lbsf = np.inf
            for j in np.arange(i - 1 - exclude_zone, -1, -1):
                d = eudis1(curr, X[j])
                if d < lbsf:
                    lbsf = d
                    if get_c22_mp_idxs:
                        left_c22_mp_idxs[i] = j

            left_C22_MP[i] = lbsf

    anomaly_idxs = np.asarray([[k, v] for k, v in anomaly_idxs.items()])

    if mad is not None:
        left_C22_MP = np.min(rolling_window(left_C22_MP, mad), 1)
    left_C22_MP = np.pad(left_C22_MP, (win // 2, 0), "constant")
    return (
        left_C22_MP,
        anomaly_idxs,
        n_skipped,
        n_full_back,
        bsf_mem,
        left_c22_mp_idxs,
        bsf_mem_idx,
    )


def feature_search_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    weights = np.abs(model.coef_)
    return weights


def prepare_data_to_get_weight(normal, anomaly):
    feature_for_all_anomaly_ts = []
    for i in range(len(anomaly)):
        feature_for_one_ts = pycatch22.catch22_all(anomaly[i])["values"]
        feature_for_all_anomaly_ts.append(feature_for_one_ts)

    feature_for_all_normal_ts = []
    for i in range(len(normal)):
        feature_for_one_ts = pycatch22.catch22_all(normal[i])["values"]
        feature_for_all_normal_ts.append(feature_for_one_ts)

    normal_labels = [0] * len(normal)
    anomaly_labels = [1] * len(anomaly)

    anomaly_labels.extend(normal_labels)
    original_labels = anomaly_labels
    original_labels = np.array(original_labels)

    feature_for_all_anomaly_ts.extend(feature_for_all_normal_ts)
    original_features = feature_for_all_anomaly_ts
    original_features = np.array(original_features)

    (
        original_train_features,
        original_test_features,
        original_train_labels,
        original_test_labels,
    ) = train_test_split(
        original_features, original_labels, test_size=0.1, random_state=42
    )
    return (
        original_features,
        original_labels,
        original_train_features,
        original_test_features,
        original_train_labels,
        original_test_labels,
    )


def feature_search_classification(
    original_features,
    original_labels,
    original_train_features,
    original_test_features,
    original_train_labels,
    original_test_labels,
):
    model = DecisionTreeClassifier()
    model.fit(original_train_features, original_train_labels)
    predictions = model.predict(original_test_features)
    errors = abs(predictions - original_test_labels)
    # print('Average model error:', round(np.mean(errors), 2), 'degrees.')
    model.fit(original_features, original_labels)
    weights = model.feature_importances_
    return weights


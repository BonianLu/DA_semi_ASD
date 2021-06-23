from __future__ import print_function

import sys
import numpy as np

sys.path.insert(0, "/mnt/sda/DA_semi_ASD/Model_")
import util


def features_plot_supp(filepath, features_model, test_model, source_data, target_data,supp_data, description):
    train_fts_source, train_labels_source = source_data[0]
    valid_fts_source, valid_labels_source = source_data[1]
    test_fts_source, test_labels_source = source_data[2]

    train_fts_target, train_labels_target = target_data[0]
    valid_fts_target, valid_labels_target = target_data[1]
    test_fts_target, test_labels_target = target_data[2]

    train_fts_supp, train_labels_supp = supp_data[0]
    valid_fts_target, valid_labels_target = supp_data[1]
    test_fts_target, test_labels_target = supp_data[2]

    y_dim = np.shape(train_labels_source)[1]
    S_labels = train_labels_source
    T_labels = train_labels_target
    P_labels = train_labels_supp

    #
    zy_S = features_model()[0]
    zy_T = features_model()[1]
    zy_P = features_model()[2]
    zy_S, zy_T,zy_P = util.feature_tsne_supp(zy_S, zy_T,zy_P)

    label_zy_S = []
    label_zy_T = []
    label_zy_P = []

    for i in range(y_dim):
        label_zy_S.append(zy_S[np.where(S_labels[:, i] == 1)[0], :])
        label_zy_T.append(zy_T[np.where(T_labels[:, i] == 1)[0], :])
        label_zy_P.append(zy_P[np.where(P_labels[:, i] == 1)[0], :])

    # Source zy feature
    title = 'Source_zy_feature_%s' % (description)
    fts = ()
    for i in range(y_dim):
        fts = fts + (label_zy_S[i][:, 0], label_zy_S[i][:, 1])
    label = ['Controls', 'Aspergers', 'Autism']
    color = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    marker = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    line = False
    legend = True
    util.data2plot(filepath=filepath, title=title, fts=fts, label=label, color=color, marker=marker, line=line,
                   legend=legend)

    # Target zy feature
    title = 'Target_zy_feature_%s' % (description)
    fts = ()
    for i in range(y_dim):
        fts = fts + (label_zy_T[i][:, 0], label_zy_T[i][:, 1])
    label = ['Controls', 'Aspergers', 'Autism']
    color = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    marker = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    line = False
    legend = True
    util.data2plot(filepath=filepath, title=title, fts=fts, label=label, color=color, marker=marker, line=line,
                   legend=legend)

    # Supp zy feature
    title = 'Additional_zy_feature_%s' % (description)
    fts = ()
    for i in range(y_dim):
        fts = fts + (label_zy_P[i][:, 0], label_zy_P[i][:, 1])
    label = ['Controls', 'Aspergers', 'Autism']
    color = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    marker = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    line = False
    legend = True
    util.data2plot(filepath=filepath, title=title, fts=fts, label=label, color=color, marker=marker, line=line,
                   legend=legend)

    # Both source, target zy feature
    title = 'Zy_feature_%s' % (description)
    fts = ()
    tmp = ()
    for i in range(y_dim):
        fts = fts + (label_zy_S[i][:, 0], label_zy_S[i][:, 1])
        fts = fts + (label_zy_T[i][:, 0], label_zy_T[i][:, 1])
        fts = fts + (label_zy_P[i][:, 0], label_zy_P[i][:, 1])
    label = ['Source:Controls', 'Target:Controls', 'Additional:Controls',
             'Source:Aspergers', 'Target:Aspergers', 'Additional:Aspergers',
             'Source:Autism','Target:Autism','Additional:Autism']
    color = [1, 1, 1, 2, 2, 2,3,3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10]
    marker = [1, 2, 3, 1, 2, 3,1, 2, 3,1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    line = False
    legend = True
    util.data2plot(filepath=filepath, title=title, fts=fts, label=label, color=color, marker=marker, line=line,
                   legend=legend)

    # Both source, target zy feature
    title = 'Test_zy_feature_%s' % (description)
    fts = ()
    tmp = ()
    for i in range(y_dim):
        fts = fts + (label_zy_S[i][:, 0], label_zy_S[i][:, 1])
        fts = fts + (label_zy_T[i][:, 0], label_zy_T[i][:, 1])
        fts = fts + (label_zy_P[i][:, 0], label_zy_P[i][:, 1])
    label = ['Source:Controls', 'Target:Controls', 'Additional:Controls',
             'Source:Aspergers', 'Target:Aspergers', 'Additional:Aspergers',
             'Source:Autism','Target:Autism', 'Additional:Autism']
    color = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10]
    marker = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    line = False
    legend = True
    util.data2plot(filepath=filepath, title=title, fts=fts, label=label, color=color, marker=marker, line=line,
                   legend=legend)

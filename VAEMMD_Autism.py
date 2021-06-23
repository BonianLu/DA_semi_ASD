# %matplotlib inline

import sys
import pickle

import numpy as np
import theano.tensor as T

sys.path.insert(0, "/mnt/sda/DA_semi_ASD/Model_")
import Model_.nnet_supp as nn
import Model_.VAEMMD_SEMI_supp as VFAE
import VAEMMD_FeaturePlot as fp
import math
import pandas as pd

'''Model Construct'''

def data_shuffle(ip_array, ip_label):
    shuffle_id = np.random.permutation(ip_array.shape[0])
    shuffle_id = shuffle_id.tolist()
    new_array = ip_array[shuffle_id,:]
    new_label = ip_label[shuffle_id,:]

    return new_array, new_label

def oneHot_reverse(array, labels_size):
    result = np.empty((0), int)

    for temp in array:
        temp_result = np.where(temp == 1)
        result = np.concatenate((result, list(temp_result[0])))

    return result

def groups_count(one_hot_array, labels_size):
    a = oneHot_reverse(np.array(one_hot_array), labels_size)
    return [np.sum(a == 0),np.sum(a == 1),np.sum(a == 2)]



def generate_data(source_data, target_data,miniclass = True,  filepath = None):

    train_fts_source, train_labels_source = source_data[0]
    train_fts_target, train_labels_target = target_data[0]

    asp_id = np.nonzero(train_labels_source[:,1])[0].tolist()
    asp_num= len(asp_id)
    aut_id = np.nonzero(train_labels_source[:,2])[0].tolist()
    aut_num= len(aut_id)
    con_id = np.nonzero(train_labels_source[:,0])[0].tolist()
    con_num= len(con_id)

    y_dim = np.shape(train_labels_source)[1]
    S_labels = train_labels_source
    T_labels = train_labels_target

    asp_misnum = con_num - asp_num
    aut_misnum = con_num - aut_num

    asp_data = train_fts_source[asp_id,:]
    aut_data = train_fts_source[aut_id, :]
    con_data = train_fts_source[con_id, :]

    asp_label = train_labels_source[asp_id,:]
    aut_label = train_labels_source[aut_id, :]
    con_label = train_labels_source[con_id, :]

    if miniclass :
        asp_data_rep = np.tile(asp_data, (4, 1))
        asp_label_rep = np.tile(asp_label, (4, 1))

        aut_data_rep = aut_data[0:int(math.floor(aut_num / 2)), :]
        aut_label_rep = aut_label[0:int(math.floor(aut_num / 2)), :]

        Train_result_miniclass = [np.concatenate((asp_data_rep, aut_data_rep), axis=0),
                                  np.concatenate((asp_label_rep, aut_label_rep), axis=0)]
        Train_result_generate = Train_result_miniclass

    else:
        asp_data_rep = np.tile(asp_data, (4, 1))
        asp_data_rep_combined = np.concatenate((asp_data, asp_data_rep), axis=0)
        aut_data_rep = aut_data[0:int(math.floor(aut_num / 2)), :]
        aut_data_rep_combined = np.concatenate((aut_data, aut_data_rep), axis=0)

        asp_label_rep = np.tile(asp_label, (4, 1))
        asp_label_rep_combined = np.concatenate((asp_label, asp_label_rep), axis=0)
        aut_label_rep = aut_label[0:int(math.floor(aut_num / 2)), :]
        aut_label_rep_combined = np.concatenate((aut_label, aut_label_rep), axis=0)

        Train_result_miniclass = [np.concatenate((asp_data_rep_combined, aut_data_rep_combined), axis=0),
                                  np.concatenate((asp_label_rep_combined, aut_label_rep_combined), axis=0)]
        Train_result_generate = [np.concatenate((con_data, Train_result_miniclass[0]), axis=0),
                                 np.concatenate((con_label, Train_result_miniclass[1]), axis=0)]

    return Train_result_generate




if __name__ == '__main__':

    #  Load the pickle file of source domain data (ABIDE I)
    ## Change data directory here:
    data_dir= '/mnt/sda/DA_semi_ASD/data/'
    pickle_file1 = data_dir+'ABIDEI_COMBAT_3group_sitesplit.pickle'
    with open(pickle_file1, 'rb') as f1:
        save = pickle.load(f1)
        Train_result1 = save['Train_result']
        Test_result1 = save['Test_result']
        del save

    test_sub_num = Test_result1[0].shape[0]
    Valid_result1 = [Test_result1[0][:int(np.floor(test_sub_num/2)),: ],Test_result1[1][:int(np.floor(test_sub_num/2)),: ]]
    Test_result1 = [Test_result1[0][int(np.floor(test_sub_num / 2)):, :],
                     Test_result1[1][int(np.floor(test_sub_num / 2)):, :]]

    #  Load the pickle file of target domain data (ABIDE II)
    pickle_file2 = data_dir+'ABIDEII_COMBAT_3group_sitesplit.pickle'
    with open(pickle_file2, 'rb') as f2:
        save = pickle.load(f2)

        Train_result2 = save['Train_result']
        Test_result2 = save['Test_result']

        del save

    test_sub_num = Test_result2[0].shape[0]
    a = oneHot_reverse(np.array(Test_result2[1]), 3)
    Valid_result2 = [Test_result2[0][:int(np.floor(test_sub_num / 2)), :],
                     Test_result2[1][:int(np.floor(test_sub_num / 2)), :]]
    Test_result2 = [Test_result2[0][int(np.floor(test_sub_num / 2)):, :],
                    Test_result2[1][int(np.floor(test_sub_num / 2)):, :]]

    # Load the pickle file of additional source domain data (AOMIC)
    pickle_file_AOMIC= data_dir+'AOMIC_COMBAT_3group_sitesplit.pickle'
    with open(pickle_file_AOMIC, 'rb') as f_AOMIC:
        save = pickle.load(f_AOMIC)
        PIOP1_result = save['PIOP1_result']
        PIOP2_result = save['PIOP2_result']
        del save

    AOMIC_data = np.concatenate((PIOP1_result[0],PIOP2_result[0]), axis = 0)
    control_array = np.array([[1, 0, 0]])
    AOMIC_label = np.repeat(control_array, AOMIC_data.shape[0], axis=0)
    AOMIC_result = [AOMIC_data, AOMIC_label]


    pickle_file_HBN = data_dir+'HBN_COMBAT_3group_sitesplit.pickle'
    with open(pickle_file_HBN, 'rb') as f_HBN:
        save = pickle.load(f_HBN)
        HBN_SI_result = save['HBN_SI_result']
        HBN_CBIC_result = save['HBN_CBIC_result']
        del save

    HBN_data = np.concatenate((HBN_SI_result[0], HBN_CBIC_result[0]), axis=0)
    control_array = np.array([[1, 0, 0]])
    HBN_label = np.repeat(control_array, HBN_data.shape[0], axis=0)
    HBN_result = [HBN_data, HBN_label]
    np.random.seed(1234)

    ADD_result = [np.concatenate((AOMIC_data,HBN_data), axis = 0),
                  np.concatenate((AOMIC_label,HBN_label), axis = 0)]

    # Train_result1 = [np.concatenate((Train_result1[0],ADD_result[0]), axis = 0),
    #               np.concatenate((Train_result1[1],ADD_result[1]), axis = 0)]

    ADD_more = True
    if ADD_more:
        Train_result3 = [ADD_result[0], ADD_result[1]]
        ADD_str = '_ADD'
    else:
        ADD_str = '_noADD'

    sub_num_supp = Train_result3[0].shape[0]
    Train_result3_ = [Train_result3[0][:int(np.floor(sub_num_supp/2)),: ],
                     Train_result3[1][:int(np.floor(sub_num_supp/2)),: ]]
    Valid_result3 = [Train_result3[0][int(np.floor(sub_num_supp/2)):int(np.floor(3*sub_num_supp/4)),: ],
                     Train_result3[1][int(np.floor(sub_num_supp/2)):int(np.floor(3*sub_num_supp/4)),: ]]
    Test_result3 = [Train_result3[0][int(np.floor(3*sub_num_supp/4)):,: ],
                     Train_result3[1][int(np.floor(3*sub_num_supp/4)):,: ]]
    # Train_result3 = Train_result3_

    source_data = [Train_result1,Valid_result1,Test_result1]
    supp_data = [Train_result3, Valid_result3, Test_result3]
    target_data = [Train_result2, Valid_result2, Test_result2  ]


    ########################################################################
    ###                        Coefficient Initial                       ###
    ########################################################################
    coef = VFAE.VFAE_coef(
        alpha=500,
        beta=500,
        chi=1,
        D=500,
        L=10,
        optimize='Adam_update'
        # optimize = 'SGD'
    )

    ip_dim = 19900
    x_dim = 19900
    y_dim = 3
    d_dim = 2

    z_dim = 2000  # dimension of latent feature
    a_dim = 1000  # dimension of prior of latent feature
    h_zy_dim = 500  # dimension of hidden unit
    h_ay_dim = 300
    h_y_dim = 300
    learning_rate = 0.0001

    activation = T.nnet.relu

    struct = VFAE.VFAE_struct()
    encoder_template = nn.NN_struct()

    struct.encoder1.share.layer_dim = [x_dim + d_dim, h_zy_dim]
    struct.encoder1.share.activation = [activation]
    struct.encoder1.share.learning_rate = [learning_rate, learning_rate]
    struct.encoder1.share.decay = [1, 1]

    struct.encoder1.mu.layer_dim = [h_zy_dim, z_dim]
    struct.encoder1.mu.activation = [None]
    struct.encoder1.mu.learning_rate = [learning_rate]
    struct.encoder1.mu.decay = [1, 1]
    struct.encoder1.sigma.layer_dim = [h_zy_dim, z_dim]
    struct.encoder1.sigma.activation = [None]
    struct.encoder1.sigma.learning_rate = [learning_rate, learning_rate]
    struct.encoder1.sigma.decay = [1, 1]

    struct.encoder2.share.layer_dim = [z_dim + y_dim, h_ay_dim]
    struct.encoder2.share.activation = [activation]
    struct.encoder2.share.learning_rate = [learning_rate, learning_rate]
    struct.encoder2.share.decay = [1, 1]
    struct.encoder2.mu.layer_dim = [h_ay_dim, a_dim]
    struct.encoder2.mu.activation = [None]
    struct.encoder2.mu.learning_rate = [learning_rate, learning_rate]
    struct.encoder2.mu.decay = [1, 1]
    struct.encoder2.sigma.layer_dim = [h_ay_dim, a_dim]
    struct.encoder2.sigma.activation = [None]
    struct.encoder2.sigma.learning_rate = [learning_rate, learning_rate]
    struct.encoder2.sigma.decay = [1, 1]

    '''
    struct.encoder3.layer_dim = [z_dim, h_y_dim, y_dim]
    struct.encoder3.activation = [activation, T.nnet.softmax]   
    struct.encoder3.learning_rate = [learning_rate, learning_rate]
    struct.encoder3.decay = [1, 1]     
    '''

    struct.encoder3.layer_dim = [z_dim, y_dim]
    struct.encoder3.activation = [T.nnet.softmax]
    struct.encoder3.learning_rate = [learning_rate, learning_rate]
    struct.encoder3.decay = [1, 1]

    struct.decoder1.share.layer_dim = [z_dim + d_dim, h_zy_dim]
    struct.decoder1.share.activation = [activation]
    struct.decoder1.share.learning_rate = [learning_rate, learning_rate]
    struct.decoder1.share.decay = [1, 1]
    struct.decoder1.mu.layer_dim = [h_zy_dim, x_dim]
    struct.decoder1.mu.activation = [None]
    struct.decoder1.mu.learning_rate = [learning_rate, learning_rate]
    struct.decoder1.mu.decay = [1, 1]
    struct.decoder1.sigma.layer_dim = [h_zy_dim, x_dim]
    struct.decoder1.sigma.activation = [None]
    struct.decoder1.sigma.learning_rate = [learning_rate, learning_rate]
    struct.decoder1.sigma.decay = [1, 1]

    struct.decoder2.share.layer_dim = [a_dim + y_dim, h_ay_dim]
    struct.decoder2.share.activation = [activation]
    struct.decoder2.share.learning_rate = [learning_rate, learning_rate]
    struct.decoder2.share.decay = [1, 1]
    struct.decoder2.mu.layer_dim = [h_ay_dim, z_dim]
    struct.decoder2.mu.activation = [None]
    struct.decoder2.mu.learning_rate = [learning_rate, learning_rate]
    struct.decoder2.mu.decay = [1, 1]
    struct.decoder2.sigma.layer_dim = [h_ay_dim, z_dim]
    struct.decoder2.sigma.activation = [None]
    struct.decoder2.sigma.learning_rate = [learning_rate, learning_rate]
    struct.decoder2.sigma.decay = [1, 1]

    isMMD = 'MMD_'
    if coef.beta == 0:
        isMMD = 'no_MMD_'
    description = 'Autism_VFAE_%s%s%s' % (isMMD, coef.optimize, ADD_str)

    recon_result,features_model, test_model, trained_param, export_para = VFAE.VFAE_training(
        source_data=source_data,
        supp_data=supp_data,
        target_data=target_data,
        n_train_batches=20,
        n_epochs=20,
        struct=struct,
        coef=coef,
        description=description
    )

    # Customize the file path to save result
    filepath = '/mnt/sda/DA_semi_ASD/cache/'

    target_predict = features_model()[4]
    supp_predict = features_model()[5]

    fp.features_plot_supp(filepath, features_model, test_model, source_data,  target_data,supp_data,  description)

    pickle_file = filepath + 'Recon_combat.pickle'
    save = 0
    if save:
        try:

            f = open(pickle_file, 'wb')
            save = {
                'Train_result_final': recon_result,
                'Valid_result_final': target_data,
                'Test_result_final': target_data

            }
            # pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(save, f, 2)
            print('Success to save data to', pickle_file)
            f.close()
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

    sort_array_EC1 = np.array(export_para[0])
    sort_array_DC2 = np.array(export_para[1])

    save_para = 0
    if save_para:
        try:

          df_EC1 = pd.DataFrame(sort_array_EC1)
          df_EC1.to_csv(filepath +'EC1_vae.csv',index=False)
          # df_DC2 = pd.DataFrame(sort_array_DC2)
          # df_DC2.to_csv('./cache_Result/DC2_vae.csv',index=False)

        except Exception as e:
            print('Unable to save data')
            raise

import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import lil_matrix
import random
import pickle as pk
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, accuracy_score, precision_score, average_precision_score, recall_score
import relplot as rp

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size != k:
        raise ValueError('Ranking List length < k')    
    return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    sort_r = sorted(r,reverse = True)
    idcg = dcg_at_k(sort_r, k)
    if not idcg:
        #print('.', end=' ')
        return 0.
    return dcg_at_k(r, k) / idcg

def find_dp_ndcg_age(data, adj_mtx):
    m = 121
    n = len(data)
    data1 = np.zeros((m,n))
    
    embedding_np = pk.load(open('../../../../DeBayes/cne_dump/nba_seed_0_dim_8_trp_bias-deg_age.emb', 'rb'))
    #pk.load(open('nba_gat_entropy_0.01_800_uge-r_age_0.1.bin', 'rb'))

    print(embedding_np.shape)
    sizes = np.zeros(m, dtype=int)
    M = embedding_np.shape[0]

    gr = [ [] for _ in range(m) ]
    grf= [ [] for _ in range(m) ]

    train_array = np.loadtxt('../../../../multicalibration_link_prediction/train_nba.txt')
    test_array = np.loadtxt('../../../../multicalibration_link_prediction/test_nba.txt')
    val_array = np.loadtxt('../../../../multicalibration_link_prediction/val_nba.txt')

    predictions = np.loadtxt('../../../../multicalibration_link_prediction/HKRR_nba.txt', delimiter=',')
    #print(train_array, test_array, val_array)
    data_hkrr = np.concatenate((train_array, test_array, val_array))
    hkrr_dict = {}
    for i in range(predictions.shape[0]):
      hkrr_dict[data_hkrr[i]] = i

    b_old = []
    n = len(data)
    print(n)
    n_old = n
    for i in range(len(data)):
      x = int(data.iloc[i,0])
      y = int(data.iloc[i,1])
      s1 = predictions[hkrr_dict[x]][2]
      s2 = predictions[hkrr_dict[y]][2]
      z = abs(s1 - s2)
      if(z < 0.1):
        b_old.append(0.9)
      else:
        b_old.append(0.1)
    #emb = np.array(l)
    b_old = np.array(b_old)
    fair_scores = b_old

    #count = np.zeros(r, dtype=int)
    #fair_scores = []
    DP_list = []
    DPf_list = []
    sizes_edges = np.zeros(m, dtype=int)
    for i in range(n):
        m1 = int(data.iloc[i,0])
        m2 = int(data.iloc[i,3])
        #if m1<M and m2<M:
        #  fair_scores.append(sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,3])])))
        #else:
        #  fair_scores.append(sigmoid(data.iloc[i,6]))
        '''
        for j in range(2):
          for k in range(2):
            if data.iloc[i,1] == j and data.iloc[i,4] == k:
              gr[2*j+k].append(data.iloc[i,6])
              grf[2*j+k].append(fair_scores[i])
              data1[2*j+k][i] = 1
        '''
        for j in range(11):
          for k in range(11):
            if data.iloc[i,2]==j+1 and data.iloc[i,5]==k+1 :
              gr[11*j+k].append(data.iloc[i,6])   #begins at 4
              grf[11*j+k].append(fair_scores[i])
              data1[11*j+k][i] = 1
            if data.iloc[i,2]==j+1 and data.iloc[i,5]==k+1 and adj_mtx[int(data.iloc[i,0]), int(data.iloc[i,3])] > 0:
              sizes_edges[11*j+k] = sizes_edges[11*j+k] + 1
    #print(fair_scores)            
    max_size = 0
    for i in range(m):
      count=0
      for j in range(n):
        if data1[i][j]==1:
          count=count+1 
      if count>max_size:
        max_size=count
      sizes[i]=count
    print(sizes) 

    for i in range(m):
      if sizes[i]>0:
        DP_list.append(np.sum(sigmoid(np.asarray(gr[i])))/sizes[i])
        DPf_list.append(np.sum(np.asarray(grf[i]))/sizes[i])

    print(max(DP_list), min(DP_list))
    print(max(DPf_list), min(DPf_list))
    DP = max(DP_list)-min(DP_list)
    DPf = max(DPf_list)-min(DPf_list)
    print("Demogrphic Parity is without/with fair", DP, DPf)

    c = []
    for i in range(n):
      x = int(data.iloc[i,0])
      y = int(data.iloc[i,3])
      if(adj_mtx[x,y] > 0):
        c.append(1)
      else:
        c.append(0)

    c = np.array(c)

    c1 = []
    for i in range(n):
      if sigmoid(data.iloc[i,6]) > 0.7:
        c1.append(1)
      else:
        c1.append(0)
    c1 = np.array(c1)

    b = data.iloc[:,6].to_numpy()
    fair = np.zeros(n)
    for i in range(n):
      if fair_scores[i] > 0.7:
        fair[i] = 1
      else:
        fair[i] = 0    
    print("Micro F1 score for unfair score (binarised) is", f1_score(c,c1, average='micro'))
    print("Micro F1 score for fair score (binarised) is", f1_score(c,fair, average='micro'))
    print("Macro F1 score for unfair score (binarised) is", f1_score(c,c1, average='macro'))
    print("Macro F1 score for fair score (binarised) is", f1_score(c,fair, average='macro'))
    print("Binary F1 score for unfair score (binarised) is", f1_score(c,c1, average='binary'))
    print("Binary F1 score for fair score (binarised) is", f1_score(c,fair, average='binary'))
    print("Weighted F1 score for unfair score (binarised) is", f1_score(c,c1, average='weighted'))
    print("Weighted F1 score for fair score (binarised) is", f1_score(c,fair, average='weighted'))
    print("Accuracy score for unfair score (binarised) is", accuracy_score(c,c1))
    print("Accuracy score for fair score (binarised) is", accuracy_score(c,fair))
    print("Micro Average Precision score for unfair score is", average_precision_score(c,b, average='micro'))
    print("Micro Average Precision score for fair score is", average_precision_score(c,fair_scores, average='micro'))
    print("Micro Precision score for unfair score (binarised) is", precision_score(c,c1, average='micro'))
    print("Mirco Precision score for fair score (binarised) is", precision_score(c,fair, average='micro'))
    print("Mirco Recall score for unfair score (binarised) is", recall_score(c,c1, average='micro'))
    print("Mirco Recall score for fair score (binarised) is", recall_score(c,fair, average='micro'))
    print("Macro Average Precision score for unfair score is", average_precision_score(c,b, average='macro'))
    print("Macro Average Precision score for fair score is", average_precision_score(c,fair_scores, average='macro'))
    print("Binary Precision score for unfair score (binarised) is", precision_score(c,c1, average='binary'))
    print("Binary Precision score for fair score (binarised) is", precision_score(c,fair, average='binary'))

    gr_eo = [ [] for _ in range(m) ]
    grf_eo= [ [] for _ in range(m) ]

        #count = np.zeros(r, dtype=int)

    EO_list = []
    EOf_list = []

    for i in range(n):
          '''
          for j in range(2):
            for k in range(2):
              if data.iloc[i,1] == j and data.iloc[i,4] == k:
                gr_eo[2*j+k].append(sigmoid(data.iloc[i,6])*c[i])
                grf_eo[2*j+k].append(fair_scores[i]*c[i])
          '''
          for j in range(11):
            for k in range(11):
              if data.iloc[i,2]==j+1 and data.iloc[i,5]==k+1 :
                gr_eo[11*j+k].append(sigmoid(data.iloc[i,6])*c[i])   #begins at 4
                grf_eo[11*j+k].append(fair_scores[i]*c[i])

    for i in range(m):
      if sizes_edges[i] != 0:
        EO_list.append(np.sum(np.asarray(gr_eo[i]))/sizes_edges[i])
        EOf_list.append(np.sum(np.asarray(grf_eo[i]))/sizes_edges[i])

    print(max(EO_list), min(EO_list))
    print(max(EOf_list), min(EOf_list))
    EO = max(EO_list)-min(EO_list)
    EOf = max(EOf_list)-min(EOf_list)
    print("Equal Opportunity is without/with fair", EO, EOf)

    '''

    M = 403
    k = 10
    accum_ndcg = 0
    node_cnt = 0
    accum_ndcg_u = 0
    node_cnt_u = 0    
    adj_mtx_fair = lil_matrix((M,M))
    adj_mtx_unfair = lil_matrix((M,M))
    selected_pairs = lil_matrix((M,M))

    for i in range(n):
        adj_mtx_unfair[int(data.iloc[i,0]),int(data.iloc[i,3])] = sigmoid(data.iloc[i,6])
        selected_pairs[int(data.iloc[i,0]),int(data.iloc[i,3])] = 1
        adj_mtx_fair[int(data.iloc[i,0]),int(data.iloc[i,3])] = fair_scores[i]

    #print(adj_mtx_fair)
    #print(np.count_nonzero(adj_mtx_fair))
    print('Utility evaluation (link prediction)')
    s = random.sample(range(M),400)
    for node_id in s:
        node_edges = adj_mtx[node_id]
        test_pos_nodes = []
        neg_nodes = []
        for i in range(M):
            if selected_pairs[node_id,i]==1:
                 if adj_mtx[node_id,i]>0:
                     test_pos_nodes.append(i)
                 else:
                     neg_nodes.append(i)
        num_pos = len(test_pos_nodes)
        all_eval_nodes = np.concatenate((test_pos_nodes, neg_nodes)) 
        all_eval_edges = np.zeros(500)  # because each node has 20 neighbors in the set


        all_eval_edges[:num_pos] = 1
        #print(all_eval_edges)


        #in pred_edges all positive edges should be before and then negative edge sores
        edges = []
        pred_edges_fair_pos = []
        pred_edges_fair_neg = []

        pred_edges_unfair_pos = []
        pred_edges_unfair_neg = []

        for i in range(M):
            if int(selected_pairs[node_id,i])==1:
                 if adj_mtx[node_id,i]>0:
                     pred_edges_fair_pos.append(adj_mtx_fair[node_id,i])
                     pred_edges_unfair_pos.append(adj_mtx_unfair[node_id,i])
                 else:
                     pred_edges_fair_neg.append(adj_mtx_fair[node_id,i])
                     pred_edges_unfair_neg.append(adj_mtx_unfair[node_id,i])

        #print(pred_edges_fair_pos)
        pred_edges_fair = np.concatenate((np.array(pred_edges_fair_pos),np.array(pred_edges_fair_neg)))
        pred_edges_unfair = np.concatenate((np.array(pred_edges_unfair_pos),np.array(pred_edges_unfair_neg)))
        if len(pred_edges_unfair) >=k:
            #pred_edges_unfair = np.array(pred_edges_unfair)
            rank_pred_keys = np.argsort(pred_edges_unfair)[::-1]
            ranked_node_edges = all_eval_edges[rank_pred_keys]
            ndcg_u = ndcg_at_k(ranked_node_edges, k)
            if ndcg_u != 0.0:
                 accum_ndcg_u += ndcg_u
                 print(ndcg_u, node_cnt_u)
                 node_cnt_u += 1
 
        if len(pred_edges_fair) >=k:
            #pred_edges_fair = np.array(pred_edges_fair)
            rank_pred_keys = np.argsort(pred_edges_fair)[::-1]
            ranked_node_edges = all_eval_edges[rank_pred_keys]
            ndcg = ndcg_at_k(ranked_node_edges, k)
            if ndcg != 0.0:
                 accum_ndcg += ndcg
                 node_cnt += 1

    score = accum_ndcg/node_cnt
    score_u = accum_ndcg_u/node_cnt_u

    # now compute accuracy as well and dp

    print('-- ndcg of link prediction for LP score:{}'.format(score))
    print('-- ndcg of link prediction for unfair score:{}'.format(score_u))

    '''


def find_dp_ndcg(data, adj_mtx):
    m = 4
    n = len(data)
    data1 = np.zeros((m,n))
    
    embedding_np = pk.load(open('../../../../DeBayes/cne_dump/nba_seed_0_dim_8_trp_bias-deg_country.emb', 'rb'))
    sizes_edges = np.zeros(m, dtype=int)
    print(embedding_np.shape)
    sizes = np.zeros(m, dtype=int)
    M = embedding_np.shape[0]
    sizes_edges = np.zeros(m, dtype=int)

    gr = [ [] for _ in range(m) ]
    grf= [ [] for _ in range(m) ]
    
    #count = np.zeros(r, dtype=int)
    fair_scores = []
    DP_list = []
    DPf_list = []
    
    for i in range(n):
        m1 = int(data.iloc[i,0])
        m2 = int(data.iloc[i,3])
        if m1<M and m2<M:
          fair_scores.append(sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,3])])))
        else:
          fair_scores.append(sigmoid(data.iloc[i,6]))
       
        for j in range(2):
          for k in range(2):
            if data.iloc[i,1] == j and data.iloc[i,4] == k:
              gr[2*j+k].append(data.iloc[i,6])
              grf[2*j+k].append(fair_scores[i])
              data1[2*j+k][i] = 1
            if data.iloc[i,1] == j and data.iloc[i,4] == k and adj_mtx[int(data.iloc[i,0]), int(data.iloc[i,3])]>0:
              sizes_edges[2*j+k] = sizes_edges[2*j+k] + 1

        '''
        for j in range(11):
          for k in range(11):
            if data.iloc[i,2]==j+1 and data.iloc[i,5]==k+1 :
              gr[11*j+k].append(data.iloc[i,6])   #begins at 4
              grf[11*j+k].append(fair_scores[i])
              data1[11*j+k][i] = 1
            if data.iloc[i,2] == j+1 and data.iloc[i,5] == k+1 and adj_mtx[int(data.iloc[i,0]), int(data.iloc[i,3])]>0:
              sizes_edges[11*j+k] = sizes_edges[11*j+k] + 1
        '''
    #print(fair_scores)            
    max_size = 0
    for i in range(m):
      count=0
      for j in range(n):
        if data1[i][j]==1:
          count=count+1 
      if count>max_size:
        max_size=count
      sizes[i]=count
    print(sizes) 

    for i in range(m):
      if sizes[i]>0:
        DP_list.append(np.sum(sigmoid(np.asarray(gr[i])))/sizes[i])
        DPf_list.append(np.sum(np.asarray(grf[i]))/sizes[i])

    print(max(DP_list), min(DP_list))
    print(max(DPf_list), min(DPf_list))
    DP = max(DP_list)-min(DP_list)
    DPf = max(DPf_list)-min(DPf_list)
    print("Demogrphic Parity is without/with fair", DP, DPf)


    c = []
    for i in range(n):
      x = int(data.iloc[i,0])
      y = int(data.iloc[i,3])
      if(adj_mtx[x,y] > 0):
        c.append(1)
      else:
        c.append(0)

    c = np.array(c)
    c1 = []
    for i in range(n):
      if sigmoid(data.iloc[i,6]) > 0.7:
        c1.append(1)
      else:
        c1.append(0)
    c1 = np.array(c1)

    b = data.iloc[:,6].to_numpy()
    fair = np.zeros(n)
    for i in range(n):
      if fair_scores[i] > 0.7:
        fair[i] = 1
      else:
        fair[i] = 0    
    print("Micro F1 score for unfair score (binarised) is", f1_score(c,c1, average='micro'))
    print("Micro F1 score for fair score (binarised) is", f1_score(c,fair, average='micro'))
    print("Macro F1 score for unfair score (binarised) is", f1_score(c,c1, average='macro'))
    print("Macro F1 score for fair score (binarised) is", f1_score(c,fair, average='macro'))
    print("Binary F1 score for unfair score (binarised) is", f1_score(c,c1, average='binary'))
    print("Binary F1 score for fair score (binarised) is", f1_score(c,fair, average='binary'))
    print("Weighted F1 score for unfair score (binarised) is", f1_score(c,c1, average='weighted'))
    print("Weighted F1 score for fair score (binarised) is", f1_score(c,fair, average='weighted'))
    print("Accuracy score for unfair score (binarised) is", accuracy_score(c,c1))
    print("Accuracy score for fair score (binarised) is", accuracy_score(c,fair))
    print("Micro Average Precision score for unfair score is", average_precision_score(c,b, average='micro'))
    print("Micro Average Precision score for fair score is", average_precision_score(c,fair_scores, average='micro'))
    print("Micro Precision score for unfair score (binarised) is", precision_score(c,c1, average='micro'))
    print("Mirco Precision score for fair score (binarised) is", precision_score(c,fair, average='micro'))
    print("Mirco Recall score for unfair score (binarised) is", recall_score(c,c1, average='micro'))
    print("Mirco Recall score for fair score (binarised) is", recall_score(c,fair, average='micro'))
    print("Macro Average Precision score for unfair score is", average_precision_score(c,b, average='macro'))
    print("Macro Average Precision score for fair score is", average_precision_score(c,fair_scores, average='macro'))
    print("Binary Precision score for unfair score (binarised) is", precision_score(c,c1, average='binary'))
    print("Binary Precision score for fair score (binarised) is", precision_score(c,fair, average='binary'))

    gr_eo = [ [] for _ in range(m) ]
    grf_eo = [ [] for _ in range(m) ]

        #count = np.zeros(r, dtype=int)

    EO_list = []
    EOf_list = []

    for i in range(n):
          for j in range(2):
            for k in range(2):
              if data.iloc[i,1] == j and data.iloc[i,4] == k:
                gr_eo[2*j+k].append(sigmoid(data.iloc[i,6])*c[i])
                grf_eo[2*j+k].append(fair_scores[i]*c[i])
          '''
          for j in range(11):
            for k in range(11):
              if data.iloc[i,2]==j+1 and data.iloc[i,5]==k+1 :
                gr_eo[11*j+k].append(sigmoid(data.iloc[i,6])*c[i])   #begins at 4
                grf_eo[11*j+k].append(fair_scores[i]*c[i])
          '''
    for i in range(m):
      if sizes_edges[i] != 0:
        EO_list.append(np.sum(np.asarray(gr_eo[i]))/sizes_edges[i])
        EOf_list.append(np.sum(np.asarray(grf_eo[i]))/sizes_edges[i])


    print(max(EO_list), min(EO_list))
    print(max(EOf_list), min(EOf_list))
    EO = max(EO_list)-min(EO_list)
    EOf = max(EOf_list)-min(EOf_list)
    print("Equal Opportunity is without/with fair", EO, EOf)


    '''
    M = 403
    k = 10
    accum_ndcg = 0
    node_cnt = 0
    accum_ndcg_u = 0
    node_cnt_u = 0    
    adj_mtx_fair = lil_matrix((M,M))
    adj_mtx_unfair = lil_matrix((M,M))
    selected_pairs = lil_matrix((M,M))

    for i in range(n):
        adj_mtx_unfair[int(data.iloc[i,0]),int(data.iloc[i,3])] = sigmoid(data.iloc[i,6])
        selected_pairs[int(data.iloc[i,0]),int(data.iloc[i,3])] = 1
        adj_mtx_fair[int(data.iloc[i,0]),int(data.iloc[i,3])] = fair_scores[i]

    #print(adj_mtx_fair)
    #print(np.count_nonzero(adj_mtx_fair))
    print('Utility evaluation (link prediction)')
    s = random.sample(range(M),400)
    for node_id in s:
        node_edges = adj_mtx[node_id]
        test_pos_nodes = []
        neg_nodes = []
        for i in range(M):
            if selected_pairs[node_id,i]==1:
                 if adj_mtx[node_id,i]>0:
                     test_pos_nodes.append(i)
                 else:
                     neg_nodes.append(i)
 
        #pred_edges_fair.append(adj_mtx_fair[node_id][i])
        #pred_edges_unfair.append(adj_mtx_fair[node_id][i])

        #pos_nodes = np.where(node_edges>0)[0]
        #num_pos = len(pos_nodes)
        #num_test_pos = int(len(pos_nodes) / 10) + 1
        #test_pos_nodes = pos_nodes[:num_test_pos]
        num_pos = len(test_pos_nodes)
        all_eval_nodes = np.concatenate((test_pos_nodes, neg_nodes)) 
        all_eval_edges = np.zeros(500)  # because each node has 20 neighbors in the set


        all_eval_edges[:num_pos] = 1
        #print(all_eval_edges)


        #in pred_edges all positive edges should be before and then negative edge sores
        edges = []
        pred_edges_fair_pos = []
        pred_edges_fair_neg = []

        pred_edges_unfair_pos = []
        pred_edges_unfair_neg = []

        for i in range(M):
            if int(selected_pairs[node_id,i])==1:
                 if adj_mtx[node_id,i]>0:
                     pred_edges_fair_pos.append(adj_mtx_fair[node_id,i])
                     pred_edges_unfair_pos.append(adj_mtx_unfair[node_id,i])
                 else:
                     pred_edges_fair_neg.append(adj_mtx_fair[node_id,i])
                     pred_edges_unfair_neg.append(adj_mtx_unfair[node_id,i])

        #print(pred_edges_fair_pos)
        pred_edges_fair = np.concatenate((np.array(pred_edges_fair_pos),np.array(pred_edges_fair_neg)))
        pred_edges_unfair = np.concatenate((np.array(pred_edges_unfair_pos),np.array(pred_edges_unfair_neg)))
        if len(pred_edges_unfair) >=k:
            #pred_edges_unfair = np.array(pred_edges_unfair)
            rank_pred_keys = np.argsort(pred_edges_unfair)[::-1]
            ranked_node_edges = all_eval_edges[rank_pred_keys]
            ndcg_u = ndcg_at_k(ranked_node_edges, k)
            if ndcg_u != 0.0:
                 accum_ndcg_u += ndcg_u
                 print(ndcg_u, node_cnt_u)
                 node_cnt_u += 1
 
        if len(pred_edges_fair) >=k:
            #pred_edges_fair = np.array(pred_edges_fair)
            rank_pred_keys = np.argsort(pred_edges_fair)[::-1]
            ranked_node_edges = all_eval_edges[rank_pred_keys]
            ndcg = ndcg_at_k(ranked_node_edges, k)
            if ndcg != 0.0:
                 accum_ndcg += ndcg
                 node_cnt += 1

    score = accum_ndcg/node_cnt
    score_u = accum_ndcg_u/node_cnt_u

    # now compute accuracy as well and dp

    print('-- ndcg of link prediction for LP score:{}'.format(score))
    print('-- ndcg of link prediction for unfair score:{}'.format(score_u))
    '''
def find_dp_ndcg_all(data, adj_mtx):
    m = 125
    n = len(data)
    data1 = np.zeros((m,n))
    sizes_edges = np.zeros(m, dtype=int)
    #graph_embedding = np.genfromtxt("region_job.embedding",skip_header=0,dtype=float)
    #embedding_df = pd.DataFrame(graph_embedding)
    #embedding_df = embedding_df.rename(index=int, columns={0:"user_id"})
    #print(embedding_df)
    #user_ids = embedding_df['user_id']
    #embedding_df = embedding_df.drop(['user_id'],axis=1)
    embedding_np = pk.load(open('../../../../DeBayes/cne_dump/nba_seed_0_dim_8_trp_bias-deg_age.emb', 'rb'))

    print(embedding_np.shape)
    sizes = np.zeros(m, dtype=int)
    M = embedding_np.shape[0]
    train_array = np.loadtxt('../../../../multicalibration_link_prediction/train_nba.txt')
    test_array = np.loadtxt('../../../../multicalibration_link_prediction/test_nba.txt')
    val_array = np.loadtxt('../../../../multicalibration_link_prediction/val_nba.txt')

    predictions = np.loadtxt('../../../../multicalibration_link_prediction/HKRR_nba.txt', delimiter=',')
    #print(train_array, test_array, val_array)
    data_hkrr = np.concatenate((train_array, test_array, val_array))
    hkrr_dict = {}
    for i in range(predictions.shape[0]):
      hkrr_dict[data_hkrr[i]] = i

    b_old = []
    n = len(data)
    print(n)
    n_old = n
    for i in range(len(data)):
      x = int(data.iloc[i,0])
      y = int(data.iloc[i,1])
      s1 = predictions[hkrr_dict[x]][2]
      s2 = predictions[hkrr_dict[y]][2]
      z = abs(s1 - s2)
      if(z < 0.1):
        b_old.append(0.9)
      else:
        b_old.append(0.1)
    #emb = np.array(l)
    b_old = np.array(b_old)
    fair_scores = b_old

    gr = [ [] for _ in range(m) ]
    grf= [ [] for _ in range(m) ]
    
    #count = np.zeros(r, dtype=int)
    #fair_scores = []
    DP_list = []
    DPf_list = []
    
    for i in range(n):
        m1 = int(data.iloc[i,0])
        m2 = int(data.iloc[i,3])
        #if m1<M and m2<M:
        #  fair_scores.append(sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,3])])))
        #else:
        #  fair_scores.append(sigmoid(data.iloc[i,6]))
      
        for j in range(2):
          for k in range(2):
            if data.iloc[i,1] == j and data.iloc[i,4] == k:
              gr[2*j+k].append(data.iloc[i,6])
              grf[2*j+k].append(fair_scores[i])
              data1[2*j+k][i] = 1
            if data.iloc[i,1] == j and data.iloc[i,4] == k and adj_mtx[int(data.iloc[i,0]), int(data.iloc[i,3])]>0:
              sizes_edges[2*j+k] = sizes_edges[2*j+k] + 1


     
        for j in range(11):
          for k in range(11):
            if data.iloc[i,2]==j+1 and data.iloc[i,5]==k+1 :
              gr[11*j+k+4].append(data.iloc[i,6])   #begins at 4
              grf[11*j+k+4].append(fair_scores[i])
              data1[11*j+k+4][i] = 1
            if data.iloc[i,2] == j+1 and data.iloc[i,5] == k+1 and adj_mtx[int(data.iloc[i,0]), int(data.iloc[i,3])]>0:
              sizes_edges[11*j+k+4] = sizes_edges[11*j+k+4] + 1


    #print(fair_scores)            
    max_size = 0
    for i in range(m):
      count=0
      for j in range(n):
        if data1[i][j]==1:
          count=count+1 
      if count>max_size:
        max_size=count
      sizes[i]=count
    print(sizes) 

    for i in range(m):
      if sizes[i]>0:
        DP_list.append(np.sum(sigmoid(np.asarray(gr[i])))/sizes[i])
        DPf_list.append(np.sum(np.asarray(grf[i]))/sizes[i])

    print(max(DP_list), min(DP_list))
    print(max(DPf_list), min(DPf_list))
    DP = max(DP_list)-min(DP_list)
    DPf = max(DPf_list)-min(DPf_list)
    print("Demogrphic Parity is without/with fair", DP, DPf)

    c = []
    for i in range(n):
      x = int(data.iloc[i,0])
      y = int(data.iloc[i,3])
      if(adj_mtx[x,y] > 0):
        c.append(1)
      else:
        c.append(0)
    c1 = []
    for i in range(n):
      if sigmoid(data.iloc[i,6]) > 0.5:
        c1.append(1)
      else:
        c1.append(0)
    c1 = np.array(c1)


    c = np.array(c)
    gr_eo = [ [] for _ in range(m) ]
    grf_eo = [ [] for _ in range(m) ]

        #count = np.zeros(r, dtype=int)

    EO_list = []
    EOf_list = []

    for i in range(n):
          for j in range(2):
            for k in range(2):
              if data.iloc[i,1] == j and data.iloc[i,4] == k:
                gr_eo[2*j+k].append(sigmoid(data.iloc[i,6])*c[i])
                grf_eo[2*j+k].append(fair_scores[i]*c[i])

          for j in range(11):
            for k in range(11):
              if data.iloc[i,2]==j+1 and data.iloc[i,5]==k+1 :
                gr_eo[11*j+k+4].append(sigmoid(data.iloc[i,6])*c[i])   #begins at 4
                grf_eo[11*j+k+4].append(fair_scores[i]*c[i])

    for i in range(m):
      if sizes_edges[i] != 0:
        EO_list.append(np.sum(np.asarray(gr_eo[i]))/sizes_edges[i])
        EOf_list.append(np.sum(np.asarray(grf_eo[i]))/sizes_edges[i])

    print(max(EO_list), min(EO_list))
    print(max(EOf_list), min(EOf_list))
    EO = max(EO_list)-min(EO_list)
    EOf = max(EOf_list)-min(EOf_list)
    print("Equal Opportunity is without/with fair", EO, EOf)

    b = data.iloc[:,6].to_numpy()
    fair = np.zeros(n)
    for i in range(n):
      if fair_scores[i] > 0.5:
        fair[i] = 1
      else:
        fair[i] = 0    

    b = fair_scores

    scores = [[] for _ in range(m)]
    actual = [[] for _ in range(m)]


    brier = np.zeros(m)
    for i in range(n):
      x = int(data.iloc[i,0])
      y = int(data.iloc[i,3])
      for j in range(2):
        for k in range(2):
          if data.iloc[i,1] == j and data.iloc[i,4] == k:
            brier[2*j+k] = brier[2*j+k] + (b[i]-adj_mtx[x,y])**2
            scores[2*j+k].append(b[i])
            actual[2*j+k].append(adj_mtx[x,y])

      for j in range(11):
        for k in range(11):
          if data.iloc[i,2] == j+1 and data.iloc[i,5] == k+1:
            brier[11*j+k+4] = brier[11*j+k+4] + (b[i]-adj_mtx[x,y])**2
            scores[11*j+k+4].append(b[i])
            actual[11*j+k+4].append(adj_mtx[x,y])
    for i in range(m):
      if sizes[i] != 0:
        brier[i] = brier[i]/sizes[i]

    brier_gap = brier.max()-np.min(brier[brier>0])
    print("The Brier Scores are", brier)
    print("The Brier Score Gap is", brier_gap)

    thr = 0.5
    ECE = np.zeros(m)
    bin_size = 0.1

    countB_m = np.zeros((m,10), dtype=int)
    accB_m = np.zeros((m,10), dtype=int)
    confB_m = np.zeros((m,10), dtype=int)

    for i in range(n):
      x = int(data.iloc[i,0])
      y = int(data.iloc[i,3])
      for j in range(2):
        for k in range(2):
          if data.iloc[i,1] == j and data.iloc[i,4] == k:
            for l in range(10):
              if b[i] > l*0.1 and b[i] <= (l+1)*0.1:
                countB_m[2*j+k][l] = countB_m[2*j+k][l] + 1
                confB_m[2*j+k][l] = confB_m[2*j+k][l] + b[i]
                if (b[i] >= thr and adj_mtx[x,y]>0) or (b[i] < thr and adj_mtx[x,y]==0):
                  accB_m[2*j+k][l] = accB_m[2*j+k][l] + 1


      for j in range(11):
        for k in range(11):
          if data.iloc[i,2] == j+1 and data.iloc[i,5] == k+1:
            for l in range(10):
              if b[i] > l*0.1 and b[i] <= (l+1)*0.1:
                countB_m[11*j+k+4][l] = countB_m[11*j+k+4][l] + 1
                confB_m[11*j+k+4][l] = confB_m[11*j+k+4][l] + b[i]
                if (b[i] >= thr and adj_mtx[x,y]>0) or (b[i] < thr and adj_mtx[x,y]==0):
                  accB_m[11*j+k+4][l] = accB_m[11*j+k+4][l] + 1

    for i in range(m):
      for j in range(10):
        if countB_m[i][j] != 0:
          confB_m[i][j] = confB_m[i][j]/countB_m[i][j]
          accB_m[i][j] = accB_m[i][j]/countB_m[i][j]

    for i in range(m):
      if (sizes[i] != 0):
        for j in range(10):
          ECE[i] = ECE[i] + (countB_m[i][j]/sizes[i])*abs(confB_m[i][j]-accB_m[i][j])

    print("ECE scores of the groups are", ECE)
    print("Max ECE for the unfair scores is", ECE.max())

    #now compute the smooth ECE score for each group
    #for that we need the true labels (from adj_mtx) and the scores (b[i]s)

    smcalib_error = np.zeros(m)
    ece = np.zeros(m)

    for i in range(m):
      if sizes[i] != 0:
        smcalib_error[i] = rp.smECE(np.array(scores[i]),np.array(actual[i]))
        #ece[i] = expected_calibration_error(np.array(actual[i]), np.array(scores[i]))

    print("Smooth Calibration error is", smcalib_error)
    print("Max smooth calibration error is", smcalib_error.max())
    #print("Max expected calibration error after recomputation is", ece.max()-np.min(ece[ece>0]))



    b = sigmoid(data.iloc[:,6].to_numpy())

    for i in range(len(data)):
      if b[i] >= 0.5: c1[i]=1
      else: c1[i]=0

    print("AUC value for unfair score is", roc_auc_score(c,b))
    print("AUC value for the fair score is",roc_auc_score(c,fair_scores))
    print("Micro F1 score for unfair score (binarised) is", f1_score(c,c1, average='micro'))
    print("Micro F1 score for fair score (binarised) is", f1_score(c,fair, average='micro'))
    print("Macro F1 score for unfair score (binarised) is", f1_score(c,c1, average='macro'))
    print("Macro F1 score for fair score (binarised) is", f1_score(c,fair, average='macro'))
    print("Binary F1 score for unfair score (binarised) is", f1_score(c,c1, average='binary'))
    print("Binary F1 score for fair score (binarised) is", f1_score(c,fair, average='binary'))
    print("Weighted F1 score for unfair score (binarised) is", f1_score(c,c1, average='weighted'))
    print("Weighted F1 score for fair score (binarised) is", f1_score(c,fair, average='weighted'))
    print("Accuracy score for unfair score (binarised) is", accuracy_score(c,c1))
    print("Accuracy score for fair score (binarised) is", accuracy_score(c,fair))
    print("Micro Average Precision score for unfair score is", average_precision_score(c,b, average='micro'))
    print("Micro Average Precision score for fair score is", average_precision_score(c,fair_scores, average='micro'))
    print("Micro Precision score for unfair score (binarised) is", precision_score(c,c1, average='micro'))
    print("Mirco Precision score for fair score (binarised) is", precision_score(c,fair, average='micro'))
    print("Mirco Recall score for unfair score (binarised) is", recall_score(c,c1, average='micro'))
    print("Mirco Recall score for fair score (binarised) is", recall_score(c,fair, average='micro'))
    print("Macro Average Precision score for unfair score is", average_precision_score(c,b, average='macro'))
    print("Macro Average Precision score for fair score is", average_precision_score(c,fair_scores, average='macro'))
    print("Binary Precision score for unfair score (binarised) is", precision_score(c,c1, average='binary'))
    print("Binary Precision score for fair score (binarised) is", precision_score(c,fair, average='binary'))

    '''
    M = 403
    k = 10
    accum_ndcg = 0
    node_cnt = 0
    accum_ndcg_u = 0
    node_cnt_u = 0    
    adj_mtx_fair = lil_matrix((M,M))
    adj_mtx_unfair = lil_matrix((M,M))
    selected_pairs = lil_matrix((M,M))

    for i in range(n):
        adj_mtx_unfair[int(data.iloc[i,0]),int(data.iloc[i,3])] = sigmoid(data.iloc[i,6])
        selected_pairs[int(data.iloc[i,0]),int(data.iloc[i,3])] = 1
        adj_mtx_fair[int(data.iloc[i,0]),int(data.iloc[i,3])] = fair_scores[i]

    #print(adj_mtx_fair)
    #print(np.count_nonzero(adj_mtx_fair))
    print('Utility evaluation (link prediction)')
    s = random.sample(range(M),400)
    for node_id in s:
        node_edges = adj_mtx[node_id]
        test_pos_nodes = []
        neg_nodes = []
        for i in range(M):
            if selected_pairs[node_id,i]==1:
                 if adj_mtx[node_id,i]>0:
                     test_pos_nodes.append(i)
                 else:
                     neg_nodes.append(i)
 
        #pred_edges_fair.append(adj_mtx_fair[node_id][i])
        #pred_edges_unfair.append(adj_mtx_fair[node_id][i])

        #pos_nodes = np.where(node_edges>0)[0]
        #num_pos = len(pos_nodes)
        #num_test_pos = int(len(pos_nodes) / 10) + 1
        #test_pos_nodes = pos_nodes[:num_test_pos]
        num_pos = len(test_pos_nodes)
        all_eval_nodes = np.concatenate((test_pos_nodes, neg_nodes)) 
        all_eval_edges = np.zeros(500)  # because each node has 20 neighbors in the set


        all_eval_edges[:num_pos] = 1
        #print(all_eval_edges)


        #in pred_edges all positive edges should be before and then negative edge sores
        edges = []
        pred_edges_fair_pos = []
        pred_edges_fair_neg = []

        pred_edges_unfair_pos = []
        pred_edges_unfair_neg = []

        for i in range(M):
            if int(selected_pairs[node_id,i])==1:
                 if adj_mtx[node_id,i]>0:
                     pred_edges_fair_pos.append(adj_mtx_fair[node_id,i])
                     pred_edges_unfair_pos.append(adj_mtx_unfair[node_id,i])
                 else:
                     pred_edges_fair_neg.append(adj_mtx_fair[node_id,i])
                     pred_edges_unfair_neg.append(adj_mtx_unfair[node_id,i])

        #print(pred_edges_fair_pos)
        pred_edges_fair = np.concatenate((np.array(pred_edges_fair_pos),np.array(pred_edges_fair_neg)))
        pred_edges_unfair = np.concatenate((np.array(pred_edges_unfair_pos),np.array(pred_edges_unfair_neg)))
        if len(pred_edges_unfair) >=k:
            #pred_edges_unfair = np.array(pred_edges_unfair)
            rank_pred_keys = np.argsort(pred_edges_unfair)[::-1]
            ranked_node_edges = all_eval_edges[rank_pred_keys]
            ndcg_u = ndcg_at_k(ranked_node_edges, k)
            if ndcg_u != 0.0:
                 accum_ndcg_u += ndcg_u
                 print(ndcg_u, node_cnt_u)
                 node_cnt_u += 1
 
        if len(pred_edges_fair) >=k:
            #pred_edges_fair = np.array(pred_edges_fair)
            rank_pred_keys = np.argsort(pred_edges_fair)[::-1]
            ranked_node_edges = all_eval_edges[rank_pred_keys]
            ndcg = ndcg_at_k(ranked_node_edges, k)
            if ndcg != 0.0:
                 accum_ndcg += ndcg
                 node_cnt += 1

    score = accum_ndcg/node_cnt
    score_u = accum_ndcg_u/node_cnt_u

    # now compute accuracy as well and dp

    print('-- ndcg of link prediction for LP score:{}'.format(score))
    print('-- ndcg of link prediction for unfair score:{}'.format(score_u))
    '''

M = 403
adj_mtx = lil_matrix((M,M))

print('loading data ...')
with open('nba_edge.csv', 'r') as fin:
    lines = fin.readlines()
    idx = 0 
    for line in lines:
        if idx == 0: 
            idx += 1
            continue
        eachline = line.strip().split(',')
        scr_node = int(eachline[0])
        dst_node = int(eachline[1])
        weight = float(eachline[2])
        adj_mtx[scr_node, dst_node] = weight 
        adj_mtx[dst_node, scr_node] = weight 
        idx += 1

data = pd.read_csv('allNBA_country_age.csv', sep=',')
#find_dp_ndcg_age(data,adj_mtx)
#find_dp_ndcg(data,adj_mtx)
find_dp_ndcg_all(data,adj_mtx)

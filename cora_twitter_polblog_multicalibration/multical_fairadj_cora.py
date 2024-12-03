import numpy as np
import pandas as pd
import pulp as p 
import math
import random
import cvxpy as cp
import networkx as nx
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, accuracy_score, precision_score, average_precision_score, recall_score
import relplot as rp

def loss(z):
    return sum((z-b)**2.0)


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
    a = dcg_at_k(r,k)
    print("dcg of node is", a)
    return dcg_at_k(r, k) / idcg

def find_dp_ndcg_all(data, adj_mtx):
    m = 49
    n = len(data)
    data1 = np.zeros((m,n))
    
    graph_embedding = np.genfromtxt("../../../FairAdj/src/fairadj_cora.embedding",skip_header=0,dtype=float)
    embedding_df = pd.DataFrame(graph_embedding)
    #embedding_df = embedding_df.rename(index=int, columns={0:"user_id"})
    #print(embedding_df)
    #user_ids = embedding_df['user_id']
    #embedding_df = embedding_df.drop(['user_id'],axis=1)
    embedding_np = embedding_df.to_numpy()
    print(embedding_np.shape)
    sizes = np.zeros(m, dtype=int)
    sizes_edges = np.zeros(m, dtype=int)
    M = len(embedding_df)

    gr = [ [] for _ in range(m) ]
    grf= [ [] for _ in range(m) ]
    
    #count = np.zeros(r, dtype=int)
    fair_scores = []
    DP_list = []
    DPf_list = []

    c = []
    for i in range(n):
      x = int(data.iloc[i,0])
      y = int(data.iloc[i,2])
      if(adj_mtx[x,y] > 0):
        c.append(1)
      else:
        c.append(0)

    c = np.array(c)
    b = sigmoid(data.iloc[:,4].to_numpy())
    print("AUC value for unfair score is", roc_auc_score(c,b))

    for i in range(n):
        m1 = int(data.iloc[i,0])
        m2 = int(data.iloc[i,2])
        if m1<M and m2<M:
          fair_scores.append(sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,2])])))
        else:
         fair_scores.append(data.iloc[i,4])
        for j in range(7):
          for k in range(7):
            if int(data.iloc[i,1]) == j and int(data.iloc[i,3]) == k:
              gr[7*j+k].append(data.iloc[i,4])
              grf[7*j+k].append(fair_scores[i])
              data1[7*j+k][i] = 1
            if int(data.iloc[i,1]) == j and int(data.iloc[i,3]) == k and adj_mtx[int(data.iloc[i,0]),int(data.iloc[i,2])]>0:
              sizes_edges[7*j+k] = sizes_edges[7*j+k] + 1


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
      DP_list.append(np.sum(sigmoid(np.asarray(gr[i])))/sizes[i])
      DPf_list.append(np.sum(np.asarray(grf[i]))/sizes[i])

    print(max(DP_list), min(DP_list))
    print(max(DPf_list), min(DPf_list))
    DP = max(DP_list)-min(DP_list)
    DPf = max(DPf_list)-min(DPf_list)
    print("Demogrphic Parity is without/with fair", DP, DPf)


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
          for j in range(7):
            for k in range(7):
              if int(data.iloc[i,1])==j and int(data.iloc[i,3])==k :
                gr_eo[7*j+k].append(sigmoid(data.iloc[i,4])*c[i])   #begins at 4
                grf_eo[7*j+k].append(fair_scores[i]*c[i])

    for i in range(m):
      if sizes_edges[i] != 0:
        EO_list.append(np.sum(np.asarray(gr_eo[i]))/sizes_edges[i])
        EOf_list.append(np.sum(np.asarray(grf_eo[i]))/sizes_edges[i])

    print(max(EO_list), min(EO_list))
    print(max(EOf_list), min(EOf_list))
    EO = max(EO_list)-min(EO_list)
    EOf = max(EOf_list)-min(EOf_list)
    print("Equal Opportunity is without/with fair", EO, EOf)

    t = 0.5
    c1 = []
    for i in range(n):
      if(sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,2])])) > t):
        c1.append(1)
      else:
        c1.append(0)
    c1 = np.array(c1)


    fair = np.zeros(n)
    for i in range(n):
      if fair_scores[i] > t:
        fair[i] = 1
      else:
        fair[i] = 0

    b = fair_scores
    #b = sigmoid(data.iloc[:,4].to_numpy())

    scores = [[] for _ in range(m)]
    actual = [[] for _ in range(m)]


    brier = np.zeros(m)
    for i in range(n):
      x = int(data.iloc[i,0])
      y = int(data.iloc[i,2])
      for j in range(7):
        for k in range(7):
          if int(data.iloc[i,1])==j and int(data.iloc[i,3])==k :
            brier[7*j+k] = brier[7*j+k] + (b[i]-adj_mtx[x,y])**2
            scores[7*j+k].append(b[i])
            actual[7*j+k].append(adj_mtx[x,y])

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
      y = int(data.iloc[i,2])
      for j in range(7):
        for k in range(7):
          if int(data.iloc[i,1])==j and int(data.iloc[i,3])==k :
            for l in range(10):
              if b[i] > l*0.1 and b[i] <= (l+1)*0.1:
                countB_m[7*j+k][l] = countB_m[7*j+k][l] + 1
                confB_m[7*j+k][l] = confB_m[7*j+k][l] + b[i]
                if (b[i] >= thr and adj_mtx[x,y]>0) or (b[i] < thr and adj_mtx[x,y]==0):
                  accB_m[7*j+k][l] = accB_m[7*j+k][l] + 1
    for i in range(m):
      for j in range(10):
        if countB_m[i][j] != 0:
          confB_m[i][j] = confB_m[i][j]/countB_m[i][j]
          accB_m[i][j] = accB_m[i][j]/countB_m[i][j]

    for i in range(m):
      if (sizes[i] != 0):
        for j in range(10):
          ECE[i] = ECE[i] + (countB_m[i][j]/sizes[i])*abs(confB_m[i][j]-accB_m[i][j])

    avg_ECE = 0
    ECE_t = 0
    delta_ECE = 0
    max_ECE = -2
    min_ECE = 2
    max_ECE_t = -2
    min_ECE_t = 2
    count_av = 0
    count_t = 0
    for i in range(m):
      if (sizes[i] != 0):
        avg_ECE = avg_ECE + ECE[i]
        count_av = count_av + 1
        if ECE[i] >= max_ECE: max_ECE = ECE[i]
        if ECE[i] <= min_ECE: min_ECE = ECE[i]
        if ECE[i] > 0.1:
           ECE_t = ECE_t + ECE[i]
           count_t = count_t + 1
           if ECE[i]>= max_ECE_t: max_ECE_t = ECE[i]
           if ECE[i]<= min_ECE_t: min_ECE_t = ECE[i]

    avg_ECE = avg_ECE/count_av
    ECE_t = ECE_t/count_t
    nabla = 0.2
    FCE = nabla*avg_ECE + (1-nabla)*(max_ECE - min_ECE)
    ECE_t = nabla*ECE_t + (1-nabla)*(max_ECE_t - min_ECE_t)
    print("ECE scores of the groups unfair are", ECE)
    print("Max ECE for the unfair scores is", ECE.max())
    print("Delta ECE for the unfair scores is", max_ECE-min_ECE)
    print("FCE for the unfair scores is", FCE)
    print("t-ECE for the unfair scores is", ECE_t)

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


    b = sigmoid(data.iloc[:,4].to_numpy())

    for i in range(len(data)):
      if b[i] >= 0.5: c1[i]=1
      else: c1[i]=0



    #b = data.iloc[:,4].to_numpy()

    #n = len(data)
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
    print("Binary Recall score for unfair score (binarised) is", recall_score(c,c1, average='binary'))
    print("Binary Recall score for fair score (binarised) is", recall_score(c,fair, average='binary'))


    '''
    M = 2708
    k = 10
    accum_ndcg = 0
    node_cnt = 0
    accum_ndcg_u = 0
    node_cnt_u = 0    
    adj_mtx_fair = np.zeros((M,M))
    adj_mtx_unfair = np.zeros((M,M))
    selected_pairs = np.zeros((M,M))

    for i in range(n):
        adj_mtx_unfair[int(data.iloc[i,0])][int(data.iloc[i,2])] = sigmoid(data.iloc[i,4])
        selected_pairs[int(data.iloc[i,0])][int(data.iloc[i,2])] = 1
        adj_mtx_fair[int(data.iloc[i,0])][int(data.iloc[i,2])] = fair_scores[i]

        #print(adj_mtx_fair)
        #print(np.count_nonzero(adj_mtx_fair))
    print('Utility evaluation (link prediction)')
    s = random.sample(range(M),1000)
    for node_id in s:
        node_edges = adj_mtx[node_id]
        test_pos_nodes = []
        neg_nodes = []
        for i in range(M):
            if selected_pairs[node_id][i]==1:
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
          #num_pos = len(test_pos_nodes)
          #print(num_pos)

          #if num_pos == 0 or num_pos >= 100:
          #    continue
          #neg_nodes = np.random.choice(np.where(node_edges == 0)[0], 100-num_pos, replace=False)
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
            if selected_pairs[node_id][i]==1:
                if adj_mtx[node_id,i]>0:
                    pred_edges_fair_pos.append(adj_mtx_fair[node_id][i])
                    pred_edges_unfair_pos.append(adj_mtx_unfair[node_id][i])
                else:
                    pred_edges_fair_neg.append(adj_mtx_fair[node_id][i])
                    pred_edges_unfair_neg.append(adj_mtx_unfair[node_id][i])

          #print(pred_edges_fair_pos)
        pred_edges_fair = np.concatenate((np.array(pred_edges_fair_pos),np.array(pred_edges_fair_neg)))
        pred_edges_unfair = np.concatenate((np.array(pred_edges_unfair_pos),np.array(pred_edges_unfair_neg)))
        
        if len(pred_edges_unfair) >=k:
            #pred_edges_unfair = np.array(pred_edges_unfair)
          rank_pred_keys = np.argsort(pred_edges_unfair)[::-1]
          ranked_node_edges = all_eval_edges[rank_pred_keys]
          ndcg_u = ndcg_at_k(ranked_node_edges, k)
              #if ndcg_u>0: 
              #  print("edgelist is", ranked_node_edges[:20])
              #  print("Top 20 scores for unfair", sorted(list(pred_edges_unfair), reverse=True))
          if ndcg_u != 0.0:
               accum_ndcg_u += ndcg_u
               print(ndcg_u, node_cnt_u)
               node_cnt_u += 1
 
        if len(pred_edges_fair) >=k:
            #pred_edges_fair = np.array(pred_edges_fair)
          rank_pred_keys = np.argsort(pred_edges_fair)[::-1]
          ranked_node_edges = all_eval_edges[rank_pred_keys]
          ndcg = ndcg_at_k(ranked_node_edges, k)
              #if ndcg>0:
              #  print("edgelist is", ranked_node_edges[:20])
              #  print("Top 20 scores for fair", sorted(list(pred_edges_fair), reverse=True))
          if ndcg != 0.0:
               accum_ndcg += ndcg
               node_cnt += 1


    score = accum_ndcg/node_cnt
    score_u = accum_ndcg_u/node_cnt_u

    print('-- ndcg of link prediction for QP score:{}'.format(score))
    print('-- ndcg of link prediction for unfair score:{}'.format(score_u))

    '''

M = 2708
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
adj_mtx = lil_matrix((M,M))
# load the adjacency matrix of size MxM of training or testing set

print('loading data ...')
with open('../../../UGE-Unbiased-Graph-Embedding/processed_data/cora_edge.csv', 'r') as fin:
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


data = pd.read_csv('allcora_topic.csv', sep=',')
b = sigmoid(data.iloc[:,4].to_numpy())
epsilon = [0.001,0.01,0.02,0.1,0.2,0.3,0.4]
#for i in epsilon:
#  print("------------------This is for beta=",i,"------------------------")
  #LPG(data,adj_mtx,i)

find_dp_ndcg_all(data,adj_mtx)
#    LPG(data,adj_mtx,i)

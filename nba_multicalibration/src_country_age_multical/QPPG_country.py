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
    return dcg_at_k(r, k) / idcg



def QPG_EO(data,adj_mtx, beta_avg):     #here we are taking two sensitive attrib # make the data here  
    m = 4 # total 40 groups are there 
    n = len(data)
    data1 = np.zeros((m,n))
    sizes_edges = np.zeros(m)

    adamic_adar = np.zeros(n, dtype=float)
    t = 0.7
    M = adj_mtx.shape[0]
    nbr = np.zeros(M, dtype=int)
    
    #G = nx.from_numpy_matrix(adj_mtx)
   

    for i in range(M):
      for j in range(M):
        if adj_mtx[i,j] > 0:
          nbr[i] = nbr[i] + 1

    '''
    for i in range(n):
      u = int(data.iloc[i,0])
      v = int(data.iloc[i,3])
      for j in list(set(G[u]) & set(G[v])):
        adamic_adar[i] = adamic_adar[i] + 1/math.log2(nbr[j])
    '''
    a_adar_gr = np.zeros(m, dtype=float)

    sizes=np.zeros(m,dtype=int)
#     report_index(index,data1,e):  

    sizes_edges_actual = np.zeros(m, dtype=int) 

    for i in range(len(data)):
        for j in range(2):
          for k in range(2):
            if data.iloc[i,1]==j and data.iloc[i,4]==k :
              data1[2*j+k][i] = 1
              a_adar_gr[2*j+k] =  a_adar_gr[2*j+k] + adamic_adar[i]
            if data.iloc[i,1]==j and data.iloc[i,4]==k and sigmoid(data.iloc[i,6]) > t:
              sizes_edges[2*j+k] = sizes_edges[2*j+k]+1
            if data.iloc[i,1]==j and data.iloc[i,4]==k and adj_mtx[int(data.iloc[i,0]),int(data.iloc[i,3])] > 0:
              sizes_edges_actual[2*j+k] = sizes_edges_actual[2*j+k]+1


        '''
        for j in range(11):
          for k in range(11):
            if data.iloc[i,2]==j+1 and data.iloc[i,5]==k+1 :
              data1[11*j+k][i] = 1   #begins at 4
              a_adar_gr[11*j+k] =  a_adar_gr[11*j+k] + adamic_adar[i]
            if data.iloc[i,2]==j+1 and data.iloc[i,5]==k+1 and sigmoid(data.iloc[i,6]) > 0.5:
              sizes_edges[11*j+k] = sizes_edges[11*j+k]+1
            if data.iloc[i,2]==j+1 and data.iloc[i,5]==k+1 and adj_mtx[int(data.iloc[i,0]),int(data.iloc[i,3])] > 0:
              sizes_edges_actual[11*j+k] = sizes_edges_actual[11*j+k]+1

        '''

    max_size=0
    for i in range(m):
        count=0
        for j in range(n):
            if data1[i][j]==1:
                count=count+1 
        if count>max_size:
            max_size=count
        sizes[i]=count
    print(sizes)    
    print(sizes_edges/sizes)
    edge_density = np.zeros(m)
    for i in range(m):
      if sizes[i]==0:
        edge_density[i] = 0
      else:
        edge_density[i] = sizes_edges[i]*400000/sizes[i]

    print("data is")
    print(data1)
    print(m,n)

    group = np.zeros(n)
    '''
    for i in range(n):
      for j in range(2):
        for k in range(2):
          if data.iloc[i,1] == j and data.iloc[i,4] == k:
            group[i] = group[i] + edge_density[2*j+k]
     
      for j in range(11):
        for k in range(11):
          if data.iloc[i,2]==j+1 and data.iloc[i,5]==k+1 :
            group[i] = group[i] + edge_density[11*j+k+4]
    '''
    base_all = [0.5,0.6,0.7,0.8,0.9]
    indicator_all = [0] 
    for base in base_all:
      print("--------------This is for base=",base,"---------------------------")
      for indicator in indicator_all:
        print("--------------This is for indicator=",indicator,"---------------------------")
        gr = [ [] for _ in range(m) ]

        #count = np.zeros(r, dtype=int)

        random_beta = np.random.rand(m)

        DP_list = []
      
        for i in range(n):
              for j in range(2):
                for k in range(2):
                  if data.iloc[i,1] == j and data.iloc[i,4] == k:
                    gr[2*j+k].append(data.iloc[i,6])
              '''            
              for j in range(11):
                for k in range(11):
                  if data.iloc[i,2]==j and data.iloc[i,5]==k :
                    gr[11*j+k].append(data.iloc[i,6])   #begins at 4
              '''
        for i in range(m):
          if sizes[i] != 0:
            DP_list.append(np.sum(sigmoid(np.asarray(gr[i])))/sizes[i])

      
          
        
        b = sigmoid(data.iloc[:,6].to_numpy())
        
        c1 = []
        for i in range(n):
          x = int(data.iloc[i,0])
          y = int(data.iloc[i,3])
          score = sigmoid(data.iloc[i,6])
          if(score > t):
            c1.append(1)
          else:
            c1.append(0)

        c1 = np.array(c1)
            

        c = []
        for i in range(n):
          x = int(data.iloc[i,0])
          y = int(data.iloc[i,3])
          if(adj_mtx[x,y] > 0):
            c.append(1)
          else:
            c.append(0)
        c = np.array(c)

        '''
        def loss(z):
          return sum((z-b)**2.0)
        bnds = []
        upper = []
        lower = []
        for i in range(m):
          lower.append(DP_list[i]*sizes[i])
          upper.append((DP_list[i]+0.05)*sizes[i])
          bnds.append((0,1))

        linear_constraint = LinearConstraint(data1, lower, upper)
        x0 = np.ones(n)
        res = minimize(loss, x0, method='trust-constr', bounds=bnds, constraints = linear_constraint,options={'verbose': 1})
        print(res.x)
        print("Scipy optimization done")
        '''

        # Define and solve the CVXPY problem.
        Dem = np.array(DP_list)
        #Beta = beta*np.ones(m)
        beta = 0.01
        x = cp.Variable(n)
        y = cp.Variable(m)
        data_np = data1
        constraints = []
        cost1 = cp.sum_squares(cp.multiply(group,(x - b)))
        cost2 = cp.sum_squares(x - b)
        cost3 = cp.sum(cp.kl_div(x,b))

        constraints += [data1 @ cp.multiply(c1, x) <= (base + beta)*sizes_edges]
        constraints += [data1 @ cp.multiply(c1, x) >= base*sizes_edges]
        constraints += [x>=0, x<=1]

        #constraints += [data1 @ x <= (base + beta)*sizes]
        #constraints += [data1 @ x >= base*sizes]
        #constraints += [x>=0, x<=1]
        #print(constraints)
        #print(cost)
        if indicator==0: #LP
          prob = cp.Problem(cp.Minimize(cost2), constraints)
          prob.solve(solver=cp.SCS)
        elif indicator==1: #QP 
          prob = cp.Problem(cp.Minimize(cost1), constraints)
          prob.solve(solver=cp.SCS)
        else: # KL
          prob = cp.Problem(cp.Minimize(cost3), constraints)
          prob.solve(solver=cp.SCS)        
        # Print result.
        print("\nThe optimal value is", prob.value)
        print("The optimal x is")
        print(x.value)
        print("CVXPY optimization done")
        fair_scores = x.value

        fair = np.zeros(n)
        for i in range(n):
          if fair_scores[i] > t:
            fair[i] = 1
          else:
            fair[i] = 0
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


        #fair_scores = res.x
      
        #print(sizes_edges)
        #print(a_adar_gr/sizes_edges)
        #print(sizes_edges/sizes)


        # # Printing the final solution
        ## compute accuracy, new DP and old DP and NDCG
        count1 = 0
        count2 = 0
        count3 = 0

        #fair_scores = []
        for i in range(n):
          #fair_scores.append(p.value(X[i]))
          if fair_scores[i]==1.0: count1 = count1 + 1
          elif fair_scores[i]==0.0: count2 = count2 + 1
          else: count3 = count3 + 1

        print(count1,count2,count3)
        #print(np.array(fair_scores))
        print(data['score'].to_numpy())
      
        gr = [ [] for _ in range(m) ]
        grf= [ [] for _ in range(m) ]
        
        #count = np.zeros(r, dtype=int)
        
        DP_list = []
        DPf_list = []
      
        for i in range(n):
              for j in range(2):
                for k in range(2):
                  if data.iloc[i,1] == j and data.iloc[i,4] == k:
                    gr[2*j+k].append(data.iloc[i,6])
                    grf[2*j+k].append(fair_scores[i])
              '''
              for j in range(11):
                for k in range(11):
                  if data.iloc[i,2]==j+1 and data.iloc[i,5]==k+1 :
                    gr[11*j+k].append(sigmoid(data.iloc[i,6]))   #begins at 4
                    grf[11*j+k].append(fair_scores[i])
              '''
        for i in range(m):
          if sizes[i] != 0:
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
              for j in range(2):
                for k in range(2):
                  if data.iloc[i,1] == j and data.iloc[i,4] == k:
                    gr_eo[2*j+k].append(sigmoid(data.iloc[i,6])*c[i])
                    grf_eo[2*j+k].append(fair_scores[i]*c[i])
              '''
              for j in range(11):
                for k in range(11):
                  if data.iloc[i,2]==j+1 and data.iloc[i,5]==k+1 :
                    gr[11*j+k].append(sigmoid(data.iloc[i,6])*c[i])   #begins at 4
                    grf[11*j+k].append(fair_scores[i]*c[i])
              '''
        for i in range(m):
          if sizes_edges[i] != 0:
            EO_list.append(np.sum(np.asarray(gr_eo[i]))/sizes_edges_actual[i])
            EOf_list.append(np.sum(np.asarray(grf_eo[i]))/sizes_edges_actual[i])

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
        adj_mtx_fair = np.zeros((M,M))
        adj_mtx_unfair = np.zeros((M,M))
        selected_pairs = np.zeros((M,M))

        for i in range(n):
            adj_mtx_unfair[int(data.iloc[i,0])][int(data.iloc[i,3])] = sigmoid(data.iloc[i,6])
            selected_pairs[int(data.iloc[i,0])][int(data.iloc[i,3])] = 1
            adj_mtx_fair[int(data.iloc[i,0])][int(data.iloc[i,3])] = fair_scores[i]

        #print(adj_mtx_fair)
        #print(np.count_nonzero(adj_mtx_fair))
        print('Utility evaluation (link prediction)')
        s = random.sample(range(M),M)
        counter = 0
        counter1 = 0
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
        
          #print(len(pred_edges_unfair))
          #print(len(pred_edges_fair))

            
            if len(pred_edges_unfair) >=k:
              #pred_edges_unfair = np.array(pred_edges_unfair)
                rank_pred_keys = np.argsort(pred_edges_unfair)[::-1]
                ranked_node_edges = all_eval_edges[rank_pred_keys]
                ndcg_u = ndcg_at_k(ranked_node_edges, k)
                
                if ndcg_u != 0.0:
                    #print("Top edges unfair are", ranked_node_edges[:10])
                    accum_ndcg_u += ndcg_u
                    #print(ndcg_u, node_cnt_u)
                    node_cnt_u += 1
  
            if len(pred_edges_fair) >=k:
              #pred_edges_fair = np.array(pred_edges_fair)
                rank_pred_keys = np.argsort(pred_edges_fair)[::-1]
                ranked_node_edges = all_eval_edges[rank_pred_keys]
                ndcg = ndcg_at_k(ranked_node_edges, k)
                if ndcg != 0.0:
                    #print("Top edges fair are", ranked_node_edges[:10])
                    accum_ndcg += ndcg
                    #print(ndcg, node_cnt)
                    node_cnt += 1
            if ndcg_u > ndcg: counter = counter+1
            if ndcg > ndcg_u: counter1 = counter1+1
        score = accum_ndcg/node_cnt
        score_u = accum_ndcg_u/node_cnt_u
        #print("unfair scores are better in", counter)
        #print("fair scores are better in", counter1)
        print('-- ndcg of link prediction for QP score:{}'.format(score))
        print('-- ndcg of link prediction for unfair score:{}'.format(score_u))
        '''



M = 403
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
adj_mtx = lil_matrix((M,M))
# load the adjacency matrix of size MxM of training or testing set

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
b = sigmoid(data.iloc[:,6].to_numpy())
epsilon = [0.001,0.01,0.02,0.1,0.2,0.3,0.4]
#for i in epsilon:
#  print("------------------This is for beta=",i,"------------------------")
  #LPG(data,adj_mtx,i)

QPG_EO(data,adj_mtx,0.5)
#    LPG(data,adj_mtx,i)

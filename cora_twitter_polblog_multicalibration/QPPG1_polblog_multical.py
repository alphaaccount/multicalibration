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
import time
import relplot as rp
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pickle as pk

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

def write_lists_to_file(filename, list_of_lists):
    with open(filename, 'w') as f:
        for lst in list_of_lists:
            # Join the float numbers as strings with spaces between them
            line = ' '.join(map(str, lst))
            f.write(line + '\n')  # Write each list on a new line


def read_lists_from_file(filename):
    list_of_lists = []
    with open(filename, 'r') as f:
        for line in f:
            # Split each line by spaces and convert to floats
            list_of_lists.append(list(map(float, line.split())))
    return list_of_lists

def convert_list_of_lists(input_list):
    return [[1 if val > 0.5 else 0 for val in sublist] for sublist in input_list]



def QPG_EO(data,adj_mtx, beta_avg):     #here we are taking two sensitive attrib # make the data here  
    m = 4 # total 40 groups are there 
    n = len(data)
    data1 = np.zeros((m,n))
    sizes_edges = np.zeros(m)
    data = data.sample(frac=1).reset_index(drop=True)
    adamic_adar = np.zeros(n, dtype=float)

    M = adj_mtx.shape[0]
    nbr = np.zeros(M, dtype=int)
    l = []
    #G = nx.from_numpy_matrix(adj_mtx)
    graph_embedding = pk.load(open("../../../UGE-Unbiased-Graph-Embedding/embeddings/pol-blog_gat_entropy_0.01_800_embedding.bin",'rb'))
    embedding_df = pd.DataFrame(graph_embedding)
    embedding_np = embedding_df.to_numpy()
    for i in range(len(data)):                                                                                                           
      x = int(data.iloc[i,0])                                                                                                            
      y = int(data.iloc[i,2])
      l.append(np.concatenate([embedding_np[x],embedding_np[y]], axis=0))
    emb = np.array(l)
    '''
    for i in range(M):
      for j in range(M):
        if adj_mtx[i,j] > 0:
          nbr[i] = nbr[i] + 1

   
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
            if data.iloc[i,1]==j and data.iloc[i,3]==k :
              data1[2*j+k][i] = 1
              a_adar_gr[2*j+k] =  a_adar_gr[2*j+k] + adamic_adar[i]
            if data.iloc[i,1]==j and data.iloc[i,3]==k and adj_mtx[int(data.iloc[i,0]),int(data.iloc[i,2])] > 0:
              sizes_edges_actual[2*j+k] = sizes_edges_actual[2*j+k]+1
            if data.iloc[i,1]==j and data.iloc[i,3]==k and sigmoid(data.iloc[i,4]) > 0.5:
              sizes_edges[2*j+k] = sizes_edges[2*j+k]+1


        '''
        for j in range(6):
          for k in range(6):
            if data.iloc[i,2]==j and data.iloc[i,5]==k :
              data1[6*j+k+4][i] = 1   #begins at 4
              a_adar_gr[6*j+k+4] =  a_adar_gr[6*j+k+4] + adamic_adar[i]
            if data.iloc[i,2]==j and data.iloc[i,5]==k and adj_mtx[int(data.iloc[i,0]),int(data.iloc[i,3])] > 0:
              sizes_edges[6*j+k+4] = sizes_edges[6*j+k+4]+1
        '''

    #np.savetxt("data1", data1, delimiter=" ")
    print("group information written to file data1")
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
     
      for j in range(6):
        for k in range(6):
          if data.iloc[i,2]==j and data.iloc[i,5]==k :
            group[i] = group[i] + edge_density[6*j+k+4]
    '''
    c = []
    for i in range(n):
      x = int(data.iloc[i,0])
      y = int(data.iloc[i,2])
      if(adj_mtx[x,y] > 0):
        c.append(1)
      else:
        c.append(0)

    c = np.array(c)

    t  = 0.5
    c1 = []
    for i in range(n):
      x = int(data.iloc[i,0])
      y = int(data.iloc[i,2])
      score = sigmoid(data.iloc[i,4])
      if(score > t):
        c1.append(1)
      else:
        c1.append(0)

    c1 = np.array(c1)
    b = sigmoid(data.iloc[:,4].to_numpy())
    b_old = b
    c_old = c
    c1_old = c1
    data_old = data
    data1_old = data1
    sizes_old = sizes
    n_old = n
    #base_all = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    #indicator_all = [0] 
    #for base in base_all:
    SMECE = []
    ECE_l = []
    EO_l = []
    DP_l = []
    BR = []
    AUC = []
    F1 = []
    Acc = []
    indicator_all = [0]
    Alpha = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 1] #[
    for alpha in Alpha:
      print("--------------This is for Alpha=",alpha,"---------------------------")
      for indicator in indicator_all:
        print("--------------This is for indicator=",indicator,"---------------------------")
        gr = [ [] for _ in range(m) ]

        #count = np.zeros(r, dtype=int)
        p = 50000
        data = data_old[:p]
        data1 = np.zeros((m,p), dtype=int)
        random_beta = np.random.rand(m)
        for i in range(len(data)):
            for j in range(2):
              for k in range(2):
                if data.iloc[i,1]==j and data.iloc[i,3]==k :
                  data1[2*j+k][i] = 1
        DP_list = []

        b = sigmoid(data.iloc[:,4].to_numpy())
        max_size=0
        n = len(data)
        for i in range(m):
            count=0
            for j in range(n):
                if data1[i][j]==1:
                    count=count+1
            if count>max_size:
                max_size=count
            sizes[i]=count

        print("Sizes before optimization", sizes)
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

        c = []
        for i in range(n):
          x = int(data.iloc[i,0])
          y = int(data.iloc[i,2])
          if(adj_mtx[x,y] > 0):
            c.append(1)
          else:
            c.append(0)

        c = np.array(c)

        t  = 0.5
        c1 = []
        for i in range(n):
          x = int(data.iloc[i,0])
          y = int(data.iloc[i,2])
          score = sigmoid(data.iloc[i,4])
          if(score > t):
            c1.append(1)
          else:
            c1.append(0)

        c1 = np.array(c1)
        #b = data.iloc[:,4].to_numpy()
        

        # Define and solve the CVXPY problem.
        Dem = np.array(DP_list)
        #Beta = beta*np.ones(m)
        beta = 0.01
        x = cp.Variable(p)
        y = cp.Variable(m)
        data_np = data1
        constraints = []
        t1 = time.time()
        #cost1 = cp.sum_squares(cp.multiply(group,(x - b)))
        cost2 = cp.sum_squares(x - b)
        #cost3 = cp.sum(cp.kl_div(x,b))
        #constraints += [data1 @ x <= (base + beta)*sizes]
        #constraints += [data1 @ x >= base*sizes]
        #constraints += [x>=0, x<=1]
        #print(constraints)
        #print(cost)
        #constraints += [data1 @ cp.multiply(c1, x) <= (base + beta)*sizes_edges]
        #constraints += [data1 @ cp.multiply(c1, x) >= base*sizes_edges]
        #constraints += [x>=0, x<=1]
        for i in range(m):
          constraints.append(cp.sum_squares(cp.multiply(data1[i],(x-c))) <= alpha*sizes[i])

        constraints += [x>=0, x<=1]

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
        t2 = time.time()
        print("Time Taken in this iteration is", t2-t1, "seconds")
        print("\nThe optimal value is", prob.value)
        print("The optimal x is")
        print(x.value)
        print("CVXPY optimization done")
        fair_scores = x.value

        X1 = emb[:p,:]
        X2 = emb[p:,:]
        reg = LinearRegression().fit(X1, fair_scores)
        f = reg.predict(X2)
        fair_scores = np.concatenate([fair_scores,sigmoid(f)], axis=0)
        print(X1.shape)
        print(X2.shape)
        print("-------------DONE with Regression---------------")
        fair = np.zeros(n_old)
        for i in range(n_old):
          if fair_scores[i] > t:
            fair[i] = 1
          else:
            fair[i] = 0

        data = data_old
        data_new = np.zeros((m,len(data)), dtype=int)
        n = len(data)
        for i in range(len(data)):
            for j in range(2):
              for k in range(2):
                if int(data.iloc[i,1])==j and int(data.iloc[i,3])==k :
                  data_new[2*j+k][i] = 1
        max_size=0
        for i in range(m):
            count=0
            for j in range(n):
                if data_new[i][j]==1:
                    count=count+1
            if count>max_size:
                max_size=count
            sizes[i]=count

        #sizes = sizes_old
        print("Sizes after Optimization", sizes)
        b = fair_scores
        print("fair_scores are", b)
        scores = [[] for _ in range(m)]
        actual = [[] for _ in range(m)]

        n = len(data)
        brier = np.zeros(m)
        for i in range(n):
          x = int(data.iloc[i,0])
          y = int(data.iloc[i,2])
          for j in range(2):
            for k in range(2):
              if data.iloc[i,1] == j and data.iloc[i,3] == k:
                brier[2*j+k] = brier[2*j+k] + (b[i]-adj_mtx[x,y])**2
                scores[2*j+k].append(b[i])
                actual[2*j+k].append(adj_mtx[x,y])

        for i in range(m):
          if sizes[i] != 0:
            brier[i] = brier[i]/sizes[i]
        write_lists_to_file('y_true_group_polblog.txt',actual)
        write_lists_to_file('y_prob_group_fair_polblog'+str(alpha)+'.txt',scores)
        write_lists_to_file('y_pred_group_fair_polblog'+str(alpha)+'.txt',convert_list_of_lists(scores))


        brier_gap = brier.max()-np.min(brier[brier>0])
        print("The Brier Scores are", brier)
        print("The Brier Score Gap is", brier_gap)
        BR.append(brier_gap)
        thr = 0.5
        ECE = np.zeros(m)
        bin_size = 0.1

        countB_m = np.zeros((m,10), dtype=int)
        accB_m = np.zeros((m,10), dtype=int)
        confB_m = np.zeros((m,10), dtype=int)

        for i in range(n):
          x = int(data.iloc[i,0])
          y = int(data.iloc[i,2])
          for j in range(2):
            for k in range(2):
              if data.iloc[i,1] == j and data.iloc[i,3] == k:
                for l in range(10):
                  if b[i] > l*0.1 and b[i] <= (l+1)*0.1:
                    countB_m[2*j+k][l] = countB_m[2*j+k][l] + 1
                    confB_m[2*j+k][l] = confB_m[2*j+k][l] + b[i]
                    if (b[i] >= thr and adj_mtx[x,y]>0) or (b[i] < thr and adj_mtx[x,y]==0):
                      accB_m[2*j+k][l] = accB_m[2*j+k][l] + 1

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
        ECE_l.append(ECE.max())

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
            if ECE[i] > 0.1*ECE.max():
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
        SMECE.append(smcalib_error.max())
        #print("Max expected calibration error after recomputation is", ece.max()-np.min(ece[ece>0]))


        #b = #sigmoid(data.iloc[:,4].to_numpy())
        #c1 = c1_old
        #for i in range(len(data)):
        #  if b[i]>=0.5: c1[i]=1
        #  else: c1[i]=0
        b = b_old
        c = c_old
        c1 = c1_old
        print("AUC value for unfair score is", roc_auc_score(c,b))
        print("AUC value for the fair score is",roc_auc_score(c,fair_scores))
        AUC.append(roc_auc_score(c,fair_scores))
        print("Micro F1 score for unfair score (binarised) is", f1_score(c,c1, average='micro'))
        print("Micro F1 score for fair score (binarised) is", f1_score(c,fair, average='micro'))
        print("Macro F1 score for unfair score (binarised) is", f1_score(c,c1, average='macro'))
        print("Macro F1 score for fair score (binarised) is", f1_score(c,fair, average='macro'))
        F1.append(f1_score(c,fair, average='macro'))
        print("Binary F1 score for unfair score (binarised) is", f1_score(c,c1, average='binary'))
        print("Binary F1 score for fair score (binarised) is", f1_score(c,fair, average='binary'))
        print("Weighted F1 score for unfair score (binarised) is", f1_score(c,c1, average='weighted'))
        print("Weighted F1 score for fair score (binarised) is", f1_score(c,fair, average='weighted'))
        print("Accuracy score for unfair score (binarised) is", accuracy_score(c,c1))
        print("Accuracy score for fair score (binarised) is", accuracy_score(c,fair))
        Acc.append(accuracy_score(c,fair))
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
                  if data.iloc[i,1] == j and data.iloc[i,3] == k:
                    gr[2*j+k].append(data.iloc[i,4])
                    grf[2*j+k].append(fair_scores[i])
              '''            
              for j in range(6):
                for k in range(6):
                  if data.iloc[i,2]==j and data.iloc[i,5]==k :
                    gr[6*j+k+4].append(data.iloc[i,6])   #begins at 4
                    grf[6*j+k+4].append(fair_scores[i])
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
        DP_l.append(DPf)
        gr_eo = [ [] for _ in range(m) ]
        grf_eo= [ [] for _ in range(m) ]

        #count = np.zeros(r, dtype=int)

        EO_list = []
        EOf_list = []



        for i in range(n):
              for j in range(2):
                for k in range(2):
                  if data.iloc[i,1] == j and data.iloc[i,3] == k:
                    gr_eo[2*j+k].append(sigmoid(data.iloc[i,4])*c[i])
                    grf_eo[2*j+k].append(fair_scores[i]*c[i])
              '''
              for j in range(6):
                for k in range(6):
                  if data.iloc[i,2]==j and data.iloc[i,5]==k :
                    gr[6*j+k+4].append(sigmoid(data.iloc[i,6])*c[i])   #begins at 4
                    grf[6*j+k+4].append(fair_scores[i]*c[i])
              '''
        for i in range(m):
          if sizes_edges_actual[i] != 0:
            EO_list.append(np.sum(np.asarray(gr_eo[i]))/sizes_edges_actual[i])
            EOf_list.append(np.sum(np.asarray(grf_eo[i]))/sizes_edges_actual[i])

        print(max(EO_list), min(EO_list))
        print(max(EOf_list), min(EOf_list))
        EO = max(EO_list)-min(EO_list)
        EOf = max(EOf_list)-min(EOf_list)
        print("Equal Opportunity is without/with fair", EO, EOf)
        EO_l.append(EOf)
        print("SMECE =",SMECE)
        print("ECE=", ECE_l)
        print("EO_l=", EO_l)
        print("DP_l=", DP_l)
        print("BR=", BR)
        print("AUC_list=", AUC)
        print("F1_list=", F1)
        print("Acc_list=", Acc)

        '''
        M = 1222
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
        s = random.sample(range(M),500)
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

M = 1222
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
adj_mtx = lil_matrix((M,M))
# load the adjacency matrix of size MxM of training or testing set

print('loading data ...')
with open('../../../UGE-Unbiased-Graph-Embedding/processed_data/pol-blog_edge.csv', 'r') as fin:
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


data = pd.read_csv('allpolblog_party.csv', sep=',')
b = sigmoid(data.iloc[:,4].to_numpy())
epsilon = [0.001,0.01,0.02,0.1,0.2,0.3,0.4]
#for i in epsilon:
#  print("------------------This is for beta=",i,"------------------------")
  #LPG(data,adj_mtx,i)
#QPG_DP(data,adj_mtx,0.5)
QPG_EO(data,adj_mtx,0.5)
#    LPG(data,adj_mtx,i)

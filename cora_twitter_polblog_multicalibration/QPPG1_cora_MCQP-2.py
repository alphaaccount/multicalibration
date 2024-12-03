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
import pickle as pk
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
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



def ndcg_at_k(r, k):
    sort_r = sorted(r,reverse = True)
    idcg = dcg_at_k(sort_r, k)
    if not idcg:
        #print('.', end=' ')
        return 0.
    a = dcg_at_k(r,k)
    print("dcg of node is", a)
    return dcg_at_k(r, k) / idcg




def QPG_EO(data,adj_mtx):     #here we are taking two sensitive attrib # make the data here  
    m = 49 # total 40 groups are there 
    n = len(data)
    data1 = np.zeros((m,n))
    sizes_edges = np.zeros(m)

    adamic_adar = np.zeros(n, dtype=float)

    M = adj_mtx.shape[0]
    nbr = np.zeros(M, dtype=int)
    l = []
    graph_embedding = pk.load(open("../../../UGE-Unbiased-Graph-Embedding/embeddings/cora_gat_entropy_0.01_800_embedding.bin",'rb'))
    embedding_df = pd.DataFrame(graph_embedding)
    embedding_np = embedding_df.to_numpy()
    print(embedding_np.shape)
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
        for j in range(7):
          for k in range(7):
            if int(data.iloc[i,1])==j and int(data.iloc[i,3])==k :
              data1[7*j+k][i] = 1
              a_adar_gr[7*j+k] =  a_adar_gr[7*j+k] + adamic_adar[i]
            if data.iloc[i,1]==j and data.iloc[i,3]==k and adj_mtx[int(data.iloc[i,0]),int(data.iloc[i,2])] > 0:
              sizes_edges_actual[7*j+k] = sizes_edges_actual[7*j+k]+1
            if data.iloc[i,1]==j and data.iloc[i,3]==k and sigmoid(data.iloc[i,4]) > 0.7:
              sizes_edges[7*j+k] = sizes_edges[7*j+k]+1
 

    #np.savetxt("data1", data1, delimiter=" ")
    #print("group information written to file data1")
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
#    base_all = [0.5, 0.6, 0.7, 0.8, 0.9]
#    indicator_all = [0] 
#    for base in base_all:
    data_old = data
    t = 0.5
    c = []
    for i in range(n):
      x = int(data.iloc[i,0])
      y = int(data.iloc[i,2])
      if(adj_mtx[x,y] > 0):
        c.append(1)
      else:
        c.append(0)
    c = np.array(c)


    c1 = []
    for i in range(n):
      x = int(data.iloc[i,0])
      y = int(data.iloc[i,2])
      score = sigmoid(data.iloc[i,4])
      if(score > 0.5):
        c1.append(1)
      else:
        c1.append(0)

    c1 = np.array(c1)
    b = sigmoid(data.iloc[:,4].to_numpy())

    c1_old = c1
    b_old = b
    c_old = c
    n_old = len(data)
    sizes_old = sizes
    data1_old = data1
    SMECE = []
    ECE_l = []
    EO_l = []
    DP_l = []
    BR = []
    AUC_list = []
    F1_list = []
    Acc_list = []
    print("OLD sizes are", sizes_old)
    indicator_all = [0]
    Alpha = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 1]
    for alpha in Alpha:
      print("--------------This is for alpha=",alpha,"---------------------------")
      for indicator in indicator_all:
        print("--------------This is for indicator=",indicator,"---------------------------")
        gr = [ [] for _ in range(m) ]
        p = 50000
        data = data_old.iloc[:p]
        #count = np.zeros(r, dtype=int)
        data1=np.zeros((m,p), dtype=int)
        random_beta = np.random.rand(m)

        DP_list = []
        data1 = np.zeros((m,p))
        

        for i in range(len(data)):
            for j in range(7):
              for k in range(7):
                if int(data.iloc[i,1])==j and int(data.iloc[i,3])==k :
                  data1[7*j+k][i] = 1
      
        t = 0.5
        c = []
        n = len(data)
        for i in range(n):
          x = int(data.iloc[i,0])
          y = int(data.iloc[i,2])
          if(adj_mtx[x,y] > 0):
            c.append(1)
          else:
            c.append(0)
        c = np.array(c)


        c1 = []
        for i in range(n):
          x = int(data.iloc[i,0])
          y = int(data.iloc[i,2])
          score = sigmoid(data.iloc[i,4])
          if(score > 0.5):
            c1.append(1)
          else:
            c1.append(0)

        c1 = np.array(c1)

        b = sigmoid(data.iloc[:,4].to_numpy())
        max_size=0
        for i in range(m):
            count=0
            for j in range(p):
                if data1[i][j]==1:
                    count=count+1
            if count>max_size:
                max_size=count
            sizes[i]=count
        print("Sizes before optimization", sizes)

        p_x = []
        for i in range(n):
          val = sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,1])]))
          for j in range(11):
            if(val >= j*0.1 and val < (j+1)*0.1):
              p_x.append(j*0.1)


        p_x = np.array(p_x)

        #p_x = np.array([0.3, 0.2, 0.4, 0.1])  # Probabilities of p(x_i)
        v_j = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])  # Values of v_j for each support element
        y_i = c  # Example values for y_i

        z = cp.Variable((n, 11))  # z_ij variables

        # Build the objective function with the corrected nested summations
        objective_terms = []

        # For each v_j, sum over x_i such that p(x_i) = v_j
        for j in range(11):
            indices = [i for i in range(n) if p_x[i] == v_j[j]]  # Indices where p(x_i) = v_j
            if indices:  # Only include if there are such indices
                objective_terms.append(cp.sum([(y_i[i] - z[i, j] * v_j[j])**2 for i in indices]))

        objective = cp.Minimize(cp.sum(objective_terms))

        # Define the constraints
        constraints = []
        S = range(n)
        C = range(11)
        # Constraint 1: (p(xi) - vj)^2 * zij <= alpha * sum(zij) for each xi in S and vj in C
        for i in range(m):
          for j in range(11):
              constraints.append(cp.multiply(data1[i],cp.multiply((p_x - v_j[j])**2, z[:, j])) <= alpha * cp.multiply(data1[i],z[:, j]))
        for i in S:
            constraints.append(cp.sum(z[i, :]) == 1)

        # Constraint 3: 0 <= zij <= 1
        constraints += [z >= 0, z <= 1]

        # Formulate and solve the problem
        prob = cp.Problem(objective, constraints)
        prob.solve()

        # Print the results
        print("Optimal value:", prob.value)
        print("Optimal z values:\n", z.value)
        z_optimal = z.value
        max_indices = np.argmax(z_optimal, axis=1)
        fair_scores = 0.1*max_indices

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
        '''
        # Define and solve the CVXPY problem.
        Dem = np.array(DP_list)
        #Beta = beta*np.ones(m)
        beta = 0.01
        x = cp.Variable(p)
        y = cp.Variable(m)
        data_np = data1
        constraints = []
        #cost1 = cp.sum_squares(cp.multiply(group,(x - b)))
        #cost2 = cp.sum_squares(x - b)
        #cost3 = cp.sum(cp.kl_div(x,b))
        #constraints += [data1 @ x <= (base + beta)*sizes]
        #constraints += [data1 @ x >= base*sizes]
        #constraints += [x>=0, x<=1]
        #print(constraints)
        #print(cost)
        cost1 = cp.sum_squares(x - b)  #cp.sum_squares(cp.multiply(group,(x - b)))
        cost2 = cp.sum_squares(x - b)
        cost3 = cp.sum_squares(x - b)  #cp.sum(cp.kl_div(x,b))
        #constraints += [data1 @ cp.multiply(c, x) <= beta*sizes_edges_actual + base*sizes_edges_actual]
        #constraints += [data1 @ cp.multiply(c, x) >= base*sizes_edges_actual]
        #constraints += [x>=0, x<=1]
        #constraints += [base>=r1, base<=(r1+0.01)]
        #print(constraints)
        #print(cost)
        for i in range(m):
          constraints.append(cp.sum_squares(cp.multiply(data1[i],(x-c))) <= alpha*sizes[i])

        constraints += [x>=0, x<=1]
        t1 = time.time()
        #constraints += [data1 @ cp.multiply(c1, x) <= (base + beta)*sizes_edges]
        #constraints += [data1 @ cp.multiply(c1, x) >= base*sizes_edges]
        #constraints += [x>=0, x<=1]



        if indicator==0: #LP
          prob = cp.Problem(cp.Minimize(cost2), constraints)
          prob.solve(solver=cp.SCS)
        elif indicator==1: #QP 
          prob = cp.Problem(cp.Minimize(cost1), constraints)
          prob.solve(solver=cp.SCS)
        else: # KL
          prob = cp.Problem(cp.Minimize(cost3), constraints)
          prob.solve(solver=cp.SCS)

        t2 = time.time()
        print("Time Taken is",t2-t1,"seconds")
        # Print result.
        print("\nThe optimal value is", prob.value)
        print("The optimal x is")
        print(x.value)
        print("CVXPY optimization done")
        fair_scores = x.value
        '''
        t = 0.5
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
        data1_new = np.zeros((m, len(data)), dtype=int)
        for i in range(len(data)):
            for j in range(7):
              for k in range(7):
                if int(data.iloc[i,1])==j and int(data.iloc[i,3])==k :
                  data1_new[7*j+k][i] = 1

        max_size=0
        for i in range(m):
            count=0
            for j in range(len(data)):
                if data1_new[i][j]==1:
                    count=count+1
            if count>max_size:
                max_size=count
            sizes[i]=count
        #sizes = sizes_old
        b = fair_scores
        print(b)
        print("Sizes before metrics computation", sizes)
        #print("data before metrics computation", data)
        scores = [[] for _ in range(m)]
        actual = [[] for _ in range(m)]
        n = len(data)

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

        write_lists_to_file('y_prob_cora_group_fair_MCQP-1_'+str(alpha)+'.txt',scores)
        write_lists_to_file('y_pred_cora_group_fair_MCQP-1_'+str(alpha)+'.txt',convert_list_of_lists(scores))

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

        smcalib_error = np.zeros(m)
        ece = np.zeros(m)

        for i in range(m):
          if sizes[i] != 0:
            smcalib_error[i] = rp.smECE(np.array(scores[i]),np.array(actual[i]))
        

        print("Smooth Calibration error is", smcalib_error)
        print("Max smooth calibration error is", smcalib_error.max())
        SMECE.append(smcalib_error.max())
        c = c_old
        c1 = c1_old
        b = b_old #sigmoid(data.iloc[:,4].to_numpy())
        b1 = b
        for i in range(len(data)):
          if b1[i] >= 0.5: b[i]=1
          else: b[i]=0

        score = f1_score(c,fair, average='macro')
        print("AUC value for unfair score is", roc_auc_score(c,b))
        print("AUC value for the fair score is",roc_auc_score(c,fair_scores))
        AUC_list.append(roc_auc_score(c,fair_scores))
        print("Micro F1 score for unfair score (binarised) is", f1_score(c,c1, average='micro'))
        print("Micro F1 score for fair score (binarised) is", f1_score(c,fair, average='micro'))

        print("Macro F1 score for unfair score (binarised) is", f1_score(c,c1, average='macro'))
        print("Macro F1 score for fair score (binarised) is", f1_score(c,fair, average='macro'))
        F1_list.append(f1_score(c,fair, average='macro'))

        print("Binary F1 score for unfair score (binarised) is", f1_score(c,c1, average='binary'))
        print("Binary F1 score for fair score (binarised) is", f1_score(c,fair, average='binary'))
        print("Weighted F1 score for unfair score (binarised) is", f1_score(c,c1, average='weighted'))
        print("Weighted F1 score for fair score (binarised) is", f1_score(c,fair, average='weighted'))
        print("Accuracy score for unfair score (binarised) is", accuracy_score(c,c1))
        print("Accuracy score for fair score (binarised) is", accuracy_score(c,fair))
        Acc_list.append(accuracy_score(c,fair))


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
              for j in range(7):
                for k in range(7):
                  if data.iloc[i,1] == j and data.iloc[i,3] == k:
                    gr[7*j+k].append(data.iloc[i,4])
                    grf[7*j+k].append(fair_scores[i])
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
        grf_eo = [ [] for _ in range(m) ]

        #count = np.zeros(r, dtype=int)

        EO_list = []
        EOf_list = []

        for i in range(n):
              for j in range(7):
                for k in range(7):
                  if data.iloc[i,1] == j and data.iloc[i,3] == k:
                    gr_eo[7*j+k].append(sigmoid(data.iloc[i,4])*c[i])
                    grf_eo[7*j+k].append(fair_scores[i]*c[i])

        for i in range(m):
          if sizes_edges_actual[i] != 0:
            EO_list.append(np.sum(np.asarray(gr_eo[i]))/sizes_edges_actual[i])
            EOf_list.append(np.sum(np.asarray(grf_eo[i]))/sizes_edges_actual[i])

        print(max(EO_list), min(EO_list))
        print(max(EOf_list), min(EOf_list))
        #EO = max(EO_list)-min(EO_list)
        EO_list = np.array(EO_list)
        EOf_list = np.array(EOf_list)
        #EOf = max(EOf_list)-min(EOf_list)
        EOf = EOf_list.max()-np.min(EOf_list[EOf_list>0])
        EO = EO_list.max()-np.min(EO_list[EO_list>0])
        print("Equal Opportunity is without/with fair", EO, EOf)
        EO_l.append(EOf)
        print("SMECE =",SMECE)
        print("ECE=", ECE_l)
        print("EO_l=", EO_l)
        print("DP_l=", DP_l)
        print("BR=", BR)
        print("AUC_list=", AUC_list)
        print("F1_list=", F1_list)
        print("Acc_list=", Acc_list)




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

QPG_EO(data,adj_mtx)
#    LPG(data,adj_mtx,i)
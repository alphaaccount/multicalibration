import numpy as np
import pandas as pd
import pulp as p 
import math
import random
import cvxpy as cp
import networkx as nx
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import pickle as pk
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, accuracy_score, precision_score, average_precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import relplot as rp
import json

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
    #print("dcg of node is", a)
    return dcg_at_k(r, k) / idcg





def QPG_EO(data,adj_mtx, beta_avg, t):     #here we are taking two sensitive attrib # make the data here  
    #data_initial = pd.read_csv('test_pairs', sep=',')
    m = 40 # total 40 groups are there
    #n = len(data_initial)
    M = adj_mtx.shape[0]
    '''
    present = np.zeros((M,M))
    file = open("test_pairs_final","w+")
    for i in range(len(data_initial)):
      x = int(data_initial.iloc[i,0])
      y = int(data_initial.iloc[i,1])
      if present[x,y]==0 and present[y,x]==0:
        present[x,y]=1
        present[y,x]=1
        file.write(str(x)+','+str(y)+'\n')

    file.close()
    print("Original number of pairs=", n)
    


    data_initial = pd.read_csv('test_pairs_final', sep=',')
    n = len(data_initial)
    print("Old number of pairs=", n)
    total = 0
    #file = open("test_pairs_final","w+")
    file = open("test_pairs_final2", "w+")
    for i in range(len(data_initial)):
      x = int(data_initial.iloc[i,0])
      y = int(data_initial.iloc[i,1])
      if adj_mtx[x,y] > 0:
        file.write(str(x)+','+str(y)+'\n')
        total = total + 1


    for i in range(len(data_initial)):
      x = int(data_initial.iloc[i,0])
      y = int(data_initial.iloc[i,1])
      if adj_mtx[x,y] == 0 and total < 1000001:
        file.write(str(x)+','+str(y)+'\n')
        total = total + 1

    file.close()
    
    df1 = pd.read_csv('test_pos_final2.txt', sep=',', header=None)
    df2 = pd.read_csv('test_neg_final2.txt', sep=',', header=None)
    present = np.zeros((M,M), dtype=int)
    print(len(df1))
    print(len(df2))
    for i in range(len(df2)):
      x = df2.iloc[i,0]
      y = df2.iloc[i,1]
      present[x,y] = 1
      present[y,x] = 1
    for i in range(M):
      present[i,i] = 1
    N = 1000000 - 2*len(df2)
    test_neg_more = []

    Y = pd.read_csv("../../../../UGE-Unbiased-Graph-Embedding/processed_data/deleted_nodes.csv")
    deleted_nodes = set(Y["Index"])
    for i in range(N):
      x = random.randrange(0,M)
      y = random.randrange(0,M)
      if(present[x,y]==0 and adj_mtx[x,y]==0 and (x not in deleted_nodes) and (y not in deleted_nodes)):
        test_neg_more.append((x,y))
        present[x,y] = 1
        present[y,x] = 1
    df3 = pd.DataFrame(test_neg_more)
    print(len(df3))
    data = pd.concat([df1,df2,df3], ignore_index=True)
    #####   DO SOME PREPROESSING HERE SO THAT RANDOM K PAIRS ARE AT THE TOP OF THE DATA ##############
    index_list = []
    non_index_list = []
    index = random.sample(range(len(data)), 10000)
    for i in range(len(data)):
      x = int(data.iloc[i,0])
      y = int(data.iloc[i,1])
      if i in index:
        index_list.append([x,y])
      else:
        non_index_list.append([x,y])
    dataf1 = pd.DataFrame(index_list)
    dataf2 = pd.DataFrame(non_index_list)
    data = pd.concat([dataf1, dataf2], ignore_index=True)
    data.to_csv('data_new.csv', header=False, index=False)
    '''
    ##################################################################################################
    data = pd.read_csv('data_new.csv')
    dict = {}
    with open("../../../../UGE-Unbiased-Graph-Embedding/processed_data/node_to_index.txt", 'r') as f:
      for line in f:
        # Split the line into key and value at the colon and strip any extra whitespace
        key, value = line.strip().split(': ')
        # Convert the key to an integer and add the key-value pair to the dictionary
        dict[int(key)] = int(value)

    data_old = data
    l = []
    b_old = []
    graph_embedding = pk.load(open("../../../../UGE-Unbiased-Graph-Embedding/embeddings/pokec-z_gat_entropy_0.01_800_embedding.bin", 'rb'))
    embedding_df = pd.DataFrame(graph_embedding)
    embedding_np = embedding_df.to_numpy()
    print(embedding_np.shape)
    n = len(data)
    print(n)
    n_old = n
    for i in range(len(data)):
      x = int(data.iloc[i,0])
      y = int(data.iloc[i,1])
      l.append(np.concatenate([embedding_np[x],embedding_np[y]], axis=0))
      b_old.append(sigmoid(np.inner(embedding_np[x],embedding_np[y])))

    emb = np.array(l)
    b_old = np.array(b_old)

    c1_old = []
    for i in range(n):
      x = int(data.iloc[i,0])
      y = int(data.iloc[i,1])
      if(sigmoid(np.inner(embedding_np[x],embedding_np[y])) > 0.5):
        c1_old.append(1)
      else:
        c1_old.append(0)
    c1_old = np.array(c1_old)

    c_old = []
    for i in range(n):
      x = int(data.iloc[i,0])
      y = int(data.iloc[i,1])
      if(adj_mtx[x,y] > 0):
        c_old.append(1)
      else:
        c_old.append(0)

    c_old = np.array(c_old)

    data1_old = np.zeros((m,n_old), dtype=int)
    sizes_old = np.zeros(m, dtype=int)
    sizes_edges_actual_old = np.zeros(m, dtype=int)

    X = pd.read_csv("../../../../UGE-Unbiased-Graph-Embedding/processed_data/pokec-z_node_attribute6.csv")

    b = b_old
    np.savetxt('org_scores_links.txt',c1_old)
    count_of_ones = np.count_nonzero(c1_old == 1)
    count_of_zeros = np.count_nonzero(c1_old == 0)
    print("count of 1s is", count_of_ones, " and count of 0s is", count_of_zeros)
    print("The Original AUC score is",roc_auc_score(c_old,b_old))
    print("The Original F1-score is",f1_score(c_old,c1_old, average='macro'))
    #dict = json.load(open("../../../../UGE-Unbiased-Graph-Embedding/processed_data/node_to_index.txt"))
    for i in range(n_old):
        x = dict[int(data_old.iloc[i,0])]
        y = dict[int(data_old.iloc[i,1])]
        x1 = int(data_old.iloc[i,0])
        y1 = int(data_old.iloc[i,1])
        #b.append(sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,1])])))
        for j in range(2):
          for k in range(2):
            if X.iloc[x,1]==j and X.iloc[y,1]==k :
              data1_old[2*j+k][i] = 1
            if X.iloc[x,1]==j and X.iloc[y,1]==k and adj_mtx[x1, y1] > 0:
              sizes_edges_actual_old[2*j+k] = sizes_edges_actual_old[2*j+k]+1


        for j in range(6):
          for k in range(6):
            if X.iloc[x,2]==j and X.iloc[y,2]==k :
              data1_old[6*j+k+4][i] = 1
            if X.iloc[x,2]==j and X.iloc[y,2]==k and adj_mtx[x1, y1] > 0:
              sizes_edges_actual_old[6*j+k+4] = sizes_edges_actual_old[6*j+k+4]+1


    max_size=0
    for i in range(m):
      count=0
      for j in range(n_old):
          if data1_old[i][j]==1:
              count=count+1
      if count>max_size:
          max_size=count
      sizes_old[i]=count
    print("===sizes in the original data are=====", sizes_old)
    old_size = sizes_old
    sizes = sizes_old
    scores = [[] for _ in range(m)]
    actual = [[] for _ in range(m)]

    brier = np.zeros(m)
    for i in range(n):
      x = dict[int(data.iloc[i,0])]
      y = dict[int(data.iloc[i,1])]
      x1 = int(data_old.iloc[i,0])
      y1 = int(data_old.iloc[i,1])
      for j in range(2):
        for k in range(2):
          if int(X.iloc[x,1]) == j and int(X.iloc[y,1]) == k:
            brier[2*j+k] = brier[2*j+k] + (b[i]-adj_mtx[x1,y1])**2
            scores[2*j+k].append(b[i])
            actual[2*j+k].append(adj_mtx[x1,y1])

      for j in range(6):
        for k in range(6):
          if int(X.iloc[x,2])==j and int(X.iloc[y,2])==k :
            brier[6*j+k+4] = brier[6*j+k+4] + (b[i]-adj_mtx[x1,y1])**2
            scores[6*j+k+4].append(b[i])
            actual[6*j+k+4].append(adj_mtx[x1,y1])
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
      x = dict[int(data.iloc[i,0])]
      y = dict[int(data.iloc[i,1])]
      x1 = int(data_old.iloc[i,0])
      y1 = int(data_old.iloc[i,1])
      for j in range(2):
        for k in range(2):
          if int(X.iloc[x,1]) == j and int(X.iloc[y,1]) == k:
            for l in range(10):
              if b[i] > l*0.1 and b[i] <= (l+1)*0.1:
                countB_m[2*j+k][l] = countB_m[2*j+k][l] + 1
                confB_m[2*j+k][l] = confB_m[2*j+k][l] + b[i]
                if (b[i] >= thr and adj_mtx[x1,y1]>0) or (b[i] < thr and adj_mtx[x1,y1]==0):
                  accB_m[2*j+k][l] = accB_m[2*j+k][l] + 1


      for j in range(6):
        for k in range(6):
          if int(X.iloc[x,2])==j and int(X.iloc[y,2])==k :
            for l in range(10):
              if b[i] > l*0.1 and b[i] <= (l+1)*0.1:
                countB_m[6*j+k+4][l] = countB_m[6*j+k+4][l] + 1
                confB_m[6*j+k+4][l] = confB_m[6*j+k+4][l] + b[i]
                if (b[i] >= thr and adj_mtx[x1,y1]>0) or (b[i] < thr and adj_mtx[x1,y1]==0):
                  accB_m[6*j+k+4][l] = accB_m[6*j+k+4][l] + 1

    for i in range(m):
      for j in range(10):
        if countB_m[i][j] != 0:
          confB_m[i][j] = confB_m[i][j]/countB_m[i][j]
          accB_m[i][j] = accB_m[i][j]/countB_m[i][j]

    for i in range(m):
      if (sizes[i] != 0):
        for j in range(10):
          ECE[i] = ECE[i] + (countB_m[i][j]/sizes[i])*abs(confB_m[i][j]-accB_m[i][j])

    print("ECE scores of the groups unfair are", ECE)
    print("Max ECE for the unfair scores is", ECE.max())

    #now compute the smooth ECE score for each group
    #for that we need the true labels (from adj_mtx) and the scores (b[i]s)

    smcalib_error = np.zeros(m)
    ece = np.zeros(m)

    for i in range(m):
      if sizes[i] != 0:
        smcalib_error[i] = rp.smECE(np.array(scores[i]),np.array(actual[i]))
        #ece[i] = expected_calibration_error(np.array(actual[i]), np.array(scores[i]))

    print("Smooth Calibration error for unfair scores is", smcalib_error)
    print("Max smooth calibration error for unfair scores is", smcalib_error.max())


    data = data.iloc[:10000]
    data_mod = data
    n = len(data)
    print(n)
    print("New number of pairs=", n)
    list_of_edges = []
    for i in range(n):
      x = data.iloc[i,0]
      y = data.iloc[i,1]
      if(adj_mtx[x,y] > 0):
        list_of_edges.append((x,y))

    g = nx.Graph()
    g.add_edges_from(list_of_edges)
    print("Number of Connected Components", nx.number_connected_components(g))
    print("connected", nx.is_connected(g))
    data1 = np.zeros((m,n))

    #sizes = np.zeros(m, dtype=int)
    sizes_edges = np.zeros(m, dtype=int)
    sizes_edges_actual = np.zeros(m, dtype=int)
    adamic_adar = np.zeros(n, dtype=float)
    scores = [ [] for _ in range(m) ]
    a_adar_gr = np.zeros(m)

    nbr = np.zeros(M, dtype=int)


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

    #graph_embedding = pk.load(open("../../../../UGE/UGE-Unbiased-Graph-Embedding/embeddings/pokec-z_gat_entropy_0.01_800_embedding.bin", 'rb'))    
    #embedding_df = pd.DataFrame(graph_embedding)
    #embedding_df = embedding_df.rename(index=int, columns={0:"user_id"})
    #print(embedding_df)
    #user_ids = embedding_df['user_id']
    #embedding_df = embedding_df.drop(['user_id'],axis=1)
    #embedding_np = embedding_df.to_numpy()
    #print(embedding_np.shape)

    b = []
    present = np.zeros((M,M))
    for i in range(len(data)):
        x = dict[int(data.iloc[i,0])]
        y = dict[int(data.iloc[i,1])]
        x1 = int(data_old.iloc[i,0])
        y1 = int(data_old.iloc[i,1])
        b.append(sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,1])])))
        for j in range(2):
          for k in range(2):
            if X.iloc[x,1]==j and X.iloc[y,1]==k :
              data1[2*j+k][i] = 1
              a_adar_gr[2*j+k] =  a_adar_gr[2*j+k] + adamic_adar[i]
            if X.iloc[x,1]==j and X.iloc[y,1]==k and sigmoid(np.inner(embedding_np[x1],embedding_np[y1])) > t:
              sizes_edges[2*j+k] = sizes_edges[2*j+k]+1
              #scores[2*j+k].append(sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,1])])))
            if X.iloc[x,1]==j and X.iloc[y,1]==k and adj_mtx[x1, y1] > 0:
              sizes_edges_actual[2*j+k] = sizes_edges_actual[2*j+k]+1


        for j in range(6):
          for k in range(6):
            if X.iloc[x,2]==j and X.iloc[y,2]==k :
              data1[6*j+k+4][i] = 1   #begins at 4
              a_adar_gr[6*j+k+4] =  a_adar_gr[6*j+k+4] + adamic_adar[i]
            if X.iloc[x,2]==j and X.iloc[y,2]==k and sigmoid(np.inner(embedding_np[x1],embedding_np[y1])) > t:
              sizes_edges[6*j+k+4] = sizes_edges[6*j+k+4]+1
              #scores[6*j+k+4].append(sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,1])])))
            if X.iloc[x,2]==j and X.iloc[y,2]==k and adj_mtx[x1, y1] > 0:
              sizes_edges_actual[6*j+k+4] = sizes_edges_actual[6*j+k+4]+1
    print(sizes_edges_actual)
    sizes_edges_old = sizes_edges
    sizes_edges_actual_old = sizes_edges_actual
    max_size=0
    for i in range(m):
      count=0
      for j in range(n):
          if data1[i][j]==1:
              count=count+1
      if count>max_size:
          max_size=count
      sizes[i]=count
    print("sizes in the validation set are", sizes)
    b_new = b
    sizes_new = sizes


    #print(sizes)    
    #print(sizes_edges/sizes)
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
    SMECE_list = []
    ECE_list = []
    DEO_list = []
    DDP_list = []
    BR_list = []
    AUC_list = []
    F1_list = []
    Acc_list = []

    indicator_all = [0]
    Alpha = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 1]
    for alpha in Alpha:
      print("--------------This is for alpha =",alpha,"---------------------------")
      for indicator in indicator_all:
        print("--------------This is for indicator=",indicator,"---------------------------")
        gr = [ [] for _ in range(m) ]
        data = data_mod
        n = len(data)
        #sizes_edges = sizes_edges_old
        #sizes_edges_actual = sizes_edges_actual_old
        #count = np.zeros(r, dtype=int)
        r2 = []
        r1 = np.random.rand()
        for i in range(m):
          r2.append(r1)

        r1 = np.array(r2)
        DP_list = []
        group_size = {}
        group_number = {}

        topk = 90000
        c2 = []
        b = np.array(b_new)
        sorted_index = np.argsort(np.argsort(b))
        for i in range(n):
          if(sorted_index[i] > n-topk):
            c2.append(1)
          else:
            c2.append(0)
        c2 = np.array(c2)


        c1 = []
        for i in range(n):
          if(sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,1])])) > t):
            c1.append(1)
          else:
            c1.append(0)
        c1 = np.array(c1)

        c = []
        for i in range(n):
          x = int(data.iloc[i,0])
          y = int(data.iloc[i,1])
          if(adj_mtx[x,y] > 0):
            c.append(1)
          else:
            c.append(0)
        
        c = np.array(c)
        print("Total number of pairs are", n)
        print("Total number of edges are", np.count_nonzero(c > 0))

        '''
        c_adamic = []

        print("Max AA score is", adamic_adar.max())
        print("Min AA score is", adamic_adar.min())
        thresh = 0
        hist = np.zeros(4, dtype=int)
        for i in range(n):
          if adamic_adar[i] > thresh:
            c_adamic.append(1)
          else:
            c_adamic.append(0)

        c_adamic = np.array(c_adamic)

        for i in range(n):
          if adamic_adar[i] >= 0 and adamic_adar[i] < 5:
            hist[0] = hist[0] + 1
          elif adamic_adar[i] >= 5 and adamic_adar[i] < 10:
            hist[1] = hist[1] + 1
          elif adamic_adar[i] >= 10 and adamic_adar[i] < 15:
            hist[2] = hist[2] + 1
          elif adamic_adar[i] >= 15 and adamic_adar[i] < 20:
            hist[3] = hist[3] + 1

        print("distribution of AA scores is", hist)
        '''

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
        #Dem = np.array(DP_list)
        #Beta = beta*np.ones(m)
        #r1 = random.rand(m)
        #m = 38
        # Example values (adjust these to your actual data)

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

        # Constraint 2: sum(z_ij) = 1 for each xi
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
        #fair_scores = []
        #for i in range(n):
        #  for j in C:
        #    if int(z[i,j])==1:
        #      val = j
        #  fair_scores.append(val)
        #fair_scores = np.array(fair_scores)
        z_optimal = z.value  # This is a numpy 2D array of shape (n, m)

        # Find the `j` that maximizes each row `i`
        max_indices = np.argmax(z_optimal, axis=1)
        fair_scores = 0.1*max_indices
        np.savetxt('MCQP-2_fair-scores.txt',fair_scores)
        '''
        beta = 0.01
        x = cp.Variable(n)
        sizes = sizes_new
        #base = cp.Variable(m)
        data_np = data1
        #data1 = np.delete(data1, (32), axis=0)
        #sizes_edges = np.delete(sizes_edges, 32)
        constraints = []
        b = b_new
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
        #print(base.value)
        #print(r1)
        '''
        print("CVXPY optimization done")
        #fair_scores = x.value
        #np.savetxt('Fair_scores_topk.txt', fair_scores)
        #fair_scores = res.x


        ######################################Write the Regression Code here#######################################
        #y = b
        X1 = emb[:10000,:]
        X2 = emb[10000:,:]
        reg = LinearRegression().fit(X1, fair_scores)
        f = reg.predict(X2)
        fair_scores = np.concatenate([fair_scores,f], axis=0)
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
        n = len(data)
        max_size=0
        for i in range(m):
          count=0
          for j in range(n_old):
              if data1_old[i][j]==1:
                  count=count+1
          if count>max_size:
              max_size=count
          sizes[i]=count

        b = fair_scores
        scores = [[] for _ in range(m)]
        actual = [[] for _ in range(m)]


        brier = np.zeros(m)
        for i in range(n):
          x = dict[int(data.iloc[i,0])]
          y = dict[int(data.iloc[i,1])]
          x1 = int(data_old.iloc[i,0])
          y1 = int(data_old.iloc[i,1])
          for j in range(2):
            for k in range(2):
              if int(X.iloc[x,1]) == j and int(X.iloc[y,1]) == k:
                brier[2*j+k] = brier[2*j+k] + (b[i]-adj_mtx[x1,y1])**2
                scores[2*j+k].append(b[i])
                actual[2*j+k].append(adj_mtx[x1,y1])

          for j in range(6):
            for k in range(6):
              if int(X.iloc[x,2])==j and int(X.iloc[y,2])==k :
                brier[6*j+k+4] = brier[6*j+k+4] + (b[i]-adj_mtx[x1,y1])**2
                scores[6*j+k+4].append(b[i])
                actual[6*j+k+4].append(adj_mtx[x1,y1])

        for i in range(m):
          if sizes[i] != 0:
            brier[i] = brier[i]/sizes[i]

        brier_gap = brier.max()-np.min(brier[brier>0])
        print("The Brier Score of the fair scores are", brier)
        print("The Brier Score Gap of fairscores is", brier_gap)
        BR_list.append(brier_gap)
        print("old sizes are equal to", old_size)
        print("metrics computation is done on sizes equal to", sizes)

        Thr = [0.5]
        for thr in Thr:
          print("This is for Thr =", thr)
          ECE = np.zeros(m)
          bin_size = 0.1

          countB_m = np.zeros((m,10), dtype=int)
          accB_m = np.zeros((m,10), dtype=int)
          confB_m = np.zeros((m,10), dtype=int)

          for i in range(n):
            x = dict[int(data.iloc[i,0])]
            y = dict[int(data.iloc[i,1])]
            x1 = int(data_old.iloc[i,0])
            y1 = int(data_old.iloc[i,1])
            for j in range(2):
              for k in range(2):
                if int(X.iloc[x,1]) == j and int(X.iloc[y,1]) == k:
                  for l in range(10):
                    if b[i] > l*0.1 and b[i] <= (l+1)*0.1:
                      countB_m[2*j+k][l] = countB_m[2*j+k][l] + 1
                      confB_m[2*j+k][l] = confB_m[2*j+k][l] + b[i]
                      if (b[i] >= thr and adj_mtx[x1,y1]>0) or (b[i] < thr and adj_mtx[x1,y1]==0):
                        accB_m[2*j+k][l] = accB_m[2*j+k][l] + 1


            for j in range(6):
             for k in range(6):
                if int(X.iloc[x,2])==j and int(X.iloc[y,2])==k :
                  for l in range(10):
                    if b[i] > l*0.1 and b[i] <= (l+1)*0.1:
                      countB_m[6*j+k+4][l] = countB_m[6*j+k+4][l] + 1
                      confB_m[6*j+k+4][l] = confB_m[6*j+k+4][l] + b[i]
                      if (b[i] >= thr and adj_mtx[x1,y1]>0) or (b[i] < thr and adj_mtx[x1,y1]==0):
                        accB_m[6*j+k+4][l] = accB_m[6*j+k+4][l] + 1

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
          print("Max ECE for the fair scores is", ECE.max())
          ECE_list.append(ECE.max())
          smcalib_error = np.zeros(m)
          ece = np.zeros(m)

          for i in range(m):
            if sizes[i] != 0:
              smcalib_error[i] = rp.smECE(np.array(scores[i]),np.array(actual[i]))
              #ece[i] = expected_calibration_error(np.array(actual[i]), np.array(scores[i]))
    
          print("Smooth Calibration error for the fair scores is", smcalib_error)
          print("Max smooth calibration error for the fair scores is", smcalib_error.max())
          SMECE_list.append(smcalib_error.max())
          print("Metric computation is done on sizes", sizes)

        c = c_old
        b = b_old
        c1 = c1_old

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
        #print("Micro F1 score for unfair score  is", f1_score(c,b, average='micro'))
        #print("Micro F1 score for unfair score  is", f1_score(c,fair_scores, average='micro'))


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
        #print(data['score'].to_numpy())
      
        gr = [ [] for _ in range(m) ]
        grf= [ [] for _ in range(m) ]
        grf_newdef = [ [] for _ in range(m) ]
        hit_scores = [ [] for _ in range(m) ]
        non_hits =   [ [] for _ in range(m) ]
        hit_list = [ [] for _ in range(m) ]
        non_hit_list =  [ [] for _ in range(m) ]
        gr_dp = [ [] for _ in range(m) ]
        grf_dp = [ [] for _ in range(m) ]

        #count = np.zeros(r, dtype=int)

        DP_list = []
        DPf_list = []


        hits = np.zeros(m, dtype=int)
        hits_not = np.zeros(m, dtype=int)
        #count = np.zeros(r, dtype=int)
        EO_list_newdef = []
        EO_list = []
        EOf_list = []
        sizes_eo = []
        sizes_eo_edges = []
        actual_numerator = []
        thresholded_numerator = []
        hitscore = {}
        hit3 = np.zeros(m)
        hit4 = np.zeros(m)
        hit_sum1 = np.zeros(m)
        hit_sum2 = np.zeros(m)
        hit_sum3 = np.zeros(m)
        sizes_edges = np.zeros(m, dtype=int)
        sizes_edges_actual = np.zeros(m, dtype=int)
        data2 = np.zeros((m,n), dtype=int)
        for i in range(len(data)):
            x = dict[int(data.iloc[i,0])]
            y = dict[int(data.iloc[i,1])]
            #b.append(sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,1])])))
            for j in range(2):
              for k in range(2):
                if X.iloc[x,1]==j and X.iloc[y,1]==k :
                  data2[2*j+k][i] = 1
                #  a_adar_gr[2*j+k] =  a_adar_gr[2*j+k] + adamic_adar[i]
                if X.iloc[x,1]==j and X.iloc[y,1]==k and sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,1])])) > t:
                  sizes_edges[2*j+k] = sizes_edges[2*j+k]+1
              #scores[2*j+k].append(sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,1])])))
                if X.iloc[x,1]==j and X.iloc[y,1]==k and adj_mtx[int(data.iloc[i,0]), int(data.iloc[i,1])] > 0:
                  sizes_edges_actual[2*j+k] = sizes_edges_actual[2*j+k]+1


            for j in range(6):
              for k in range(6):
                if X.iloc[x,2]==j and X.iloc[y,2]==k :
                  data2[6*j+k+4][i] = 1   #begins at 4
                #  a_adar_gr[6*j+k+4] =  a_adar_gr[6*j+k+4] + adamic_adar[i]
                if X.iloc[x,2]==j and X.iloc[y,2]==k and sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,1])])) > t:
                  sizes_edges[6*j+k+4] = sizes_edges[6*j+k+4]+1
              #scores[6*j+k+4].append(sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,1])])))
                if X.iloc[x,2]==j and X.iloc[y,2]==k and adj_mtx[int(data.iloc[i,0]), int(data.iloc[i,1])] > 0:
                  sizes_edges_actual[6*j+k+4] = sizes_edges_actual[6*j+k+4]+1
        print(sizes_edges_actual)
        max_size = 0
        for i in range(m):
          count=0
          for j in range(n):
              if data2[i][j]==1:
                  count=count+1
          if count>max_size:
              max_size=count
          sizes[i]=count
        print(sizes)
        # need to compare the sums of the thresholded case and the actual case
        # and then divide by the number and compare the deltas
        for i in range(n):
              x = dict[int(data.iloc[i,0])]
              y = dict[int(data.iloc[i,1])]
              for j in range(2):
                for k in range(2):
                  if X.iloc[x,1]==j and X.iloc[y,1]==k:
                    gr[2*j+k].append(sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,1])]))*c[i])
                    grf[2*j+k].append(fair_scores[i]*c[i])
                    grf_newdef[2*j+k].append(fair_scores[i]*c1[i])
                    gr_dp[2*j+k].append(sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,1])])))
                    grf_dp[2*j+k].append(fair_scores[i])

                  if(int(X.iloc[x,1])==j and int(X.iloc[y,1])==k and c[i]==c1[i] and c[i]==1):
                    hits[2*j+k] = hits[2*j+k]+1
                    hit_scores[2*j+k].append(fair_scores[i])
                    hit_list[2*j+k].append((int(data.iloc[i,0]),int(data.iloc[i,1])))
                    hit_sum1[2*j+k] = hit_sum1[2*j+k] + fair_scores[i]
                  if(int(X.iloc[x,1])==j and int(X.iloc[y,1])==k and c[i]==0 and c1[i]==1):
                    non_hits[2*j+k].append(fair_scores[i])
                    non_hit_list[2*j+k].append((int(data.iloc[i,0]),int(data.iloc[i,1])))
                    hits_not[2*j+k] = hits_not[2*j+k]+1
                    hit_sum2[2*j+k] = hit_sum2[2*j+k] + fair_scores[i]
                  if(int(X.iloc[x,1])==j and int(X.iloc[y,1])==k and c[i]==1 and c1[i]==0):
                    hit3[2*j+k] = hit3[2*j+k]+1
                    hit_sum3[2*j+k] = hit_sum3[2*j+k] + fair_scores[i]
                  if(int(X.iloc[x,1])==j and int(X.iloc[y,1])==k and c[i]==0 and c1[i]==0):
                    hit4[2*j+k] = hit4[2*j+k]+1

              for j in range(6):
                for k in range(6):
                  if X.iloc[x,2]==j and X.iloc[y,2]==k:
                    gr[6*j+k+4].append(sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,1])]))*c[i])
                    grf[6*j+k+4].append(fair_scores[i]*c[i])
                    grf_newdef[6*j+k+4].append(fair_scores[i]*c1[i])
                    gr_dp[6*j+k+4].append(sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,1])])))   #begins at 4
                    grf_dp[6*j+k+4].append(fair_scores[i])

                  if(int(X.iloc[x,2])==j and int(X.iloc[y,2])==k and c[i]==c1[i] and c[i]==1):
                    hits[6*j+k+4] = hits[6*j+k+4]+1
                    hit_scores[6*j+k+4].append(fair_scores[i])
                    hit_list[6*j+k+4].append((int(data.iloc[i,0]),int(data.iloc[i,1])))
                    hit_sum1[6*j+k+4] = hit_sum1[6*j+k+4] + fair_scores[i]
                  if(int(X.iloc[x,2])==j and int(X.iloc[y,2])==k and c[i]==0 and c1[i]==1):
                    non_hits[6*j+k+4].append(fair_scores[i])
                    non_hit_list[6*j+k+4].append((int(data.iloc[i,0]),int(data.iloc[i,1])))
                    hits_not[6*j+k+4] = hits_not[6*j+k+4]+1
                    hit_sum2[6*j+k+4] = hit_sum2[6*j+k+4] + fair_scores[i]
                  if(int(X.iloc[x,2])==j and int(X.iloc[y,2])==k and c[i]==1 and c1[i]==0):
                    hit3[6*j+k+4] = hit3[6*j+k+4]+1
                    hit_sum3[6*j+k+4] = hit_sum3[6*j+k+4] + fair_scores[i]
                  if(int(X.iloc[x,2])==j and int(X.iloc[y,2])==k and c[i]==0 and c1[i]==0):
                    hit4[6*j+k+4] = hit4[6*j+k+4]+1

        for i in range(m):
          if sizes_edges[i] != 0:
            EO_list.append(np.sum(np.asarray(gr[i]))/sizes_edges_actual[i])
            EOf_list.append(np.sum(np.asarray(grf[i]))/sizes_edges_actual[i])
            EO_list_newdef.append(np.sum(np.asarray(grf_newdef[i]))/sizes_edges[i])
            sizes_eo.append(sizes[i])
            sizes_eo_edges.append(sizes_edges_actual[i])
            DP_list.append(np.sum(np.asarray(gr_dp[i]))/sizes[i])
            DPf_list.append(np.sum(np.asarray(grf_dp[i]))/sizes[i])

        print(max(DP_list), min(DP_list))
        print(max(DPf_list), min(DPf_list))
        DP = max(DP_list)-min(DP_list)
        DPf = max(DPf_list)-min(DPf_list)
        print("Demogrphic Parity is without/with fair", DP, DPf)
        DDP_list.append(DPf)

        print(max(EO_list), min(EO_list))
        print(max(EOf_list), min(EOf_list))
        EO = max(EO_list)-min(EO_list)
        EOf = max(EOf_list)-min(EOf_list)
        EOnewdef = max(EO_list_newdef)-min(EO_list_newdef)
        print("Equal Opportunity is without/with/newdefintiion fair", EO, EOf, EOnewdef)
        DEO_list.append(EOf)

        print(EO_list)
        print(EOf_list)
        print(EO_list_newdef)
        print("HITS are", hits)
        print("NON Hits are", hits_not)
        print("Number of elements in each group", sizes_eo)
        print("Number of edges per group", sizes_eo_edges)
        print("Sizes as per threshhold", sizes_edges.astype(int))
        print("Sum for both edges", hit_sum1)
        print("Sum for actual no edge, threhold edge", hit_sum2)
        print("Sum for actual edge, no threshold edge", hit_sum3)
        print("Actual edge threshold no edge", hit3)
        print("Both no edge", hit4)
        #print("Edge density is", np.array(sizes_eo)/np.array(sizes_eo_edges))
    
        #print("====================SCORE-DISTRIBUTION-FOR-EO-ONLY================================")
        #print("scores of group 0", scores[0])
        #print("scores of group 1", scores[1])
        '''
        print("sorted hit scores")
        for i in range(m):
          print(hit_scores[i])
        print("sorted non hits scores")

        print("HIT LIST", hit_list)
        print("NOT HIT",  non_hit_list)

        for i in range(m):
          print(non_hits[i])
        '''


        '''
        M = 67796
        k = 10
        accum_ndcg = 0
        node_cnt = 0
        accum_ndcg_u = 0
        node_cnt_u = 0    
        adj_mtx_fair = np.zeros((M,M))
        adj_mtx_unfair = np.zeros((M,M))
        selected_pairs = np.zeros((M,M))

        for i in range(n):
            adj_mtx_unfair[int(data.iloc[i,0])][int(data.iloc[i,1])] = sigmoid(np.inner(embedding_np[int(data.iloc[i,0])],embedding_np[int(data.iloc[i,1])]))
            selected_pairs[int(data.iloc[i,0])][int(data.iloc[i,1])] = 1
            adj_mtx_fair[int(data.iloc[i,0])][int(data.iloc[i,1])] = fair_scores[i]


        #print(adj_mtx_fair)
        #print(np.count_nonzero(adj_mtx_fair))
        print('Utility evaluation (link prediction)')
        s = random.sample(range(M),M)
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
            all_eval_edges = np.zeros(1500)  # because each node has 20 neighbors in the set


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

        if node_cnt != 0:
          score = accum_ndcg/node_cnt
        else: score = 0
        if node_cnt_u != 0:
          score_u = accum_ndcg_u/node_cnt_u
        else: score_u = 0

        print('-- ndcg of link prediction for QP score:{}'.format(score))
        print('-- ndcg of link prediction for unfair score:{}'.format(score_u))
        '''
        print("SMECE list is", SMECE_list)
        print("ECE list is", ECE_list) 
        print("DEO list is", DEO_list)
        print("DDP list is", DDP_list)
        print("BR list is", BR_list)
        print("AUC list is", AUC_list)
        print("F1 list is", F1_list)
        print("Acc list is", Acc_list)
        print("Alpha list is", Alpha)


M = 67796
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
adj_mtx = np.zeros((M,M), dtype=int)
# load the adjacency matrix of size MxM of training or testing set

print('loading data ...')
with open('pokec-z_edge.csv', 'r') as fin:
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


data = pd.read_csv('allPokec_region_age.csv', sep=',')
b = sigmoid(data.iloc[:,6].to_numpy())
epsilon = [0.7, 0.5,0.6, 0.8, 0.9]
total_score = 0
total_eo = 0
total_dp = 0
for t in range(10):
  #print("------------------This is for threshold=",t,"------------------------")
  QPG_EO(data,adj_mtx,0.5,0.7)
  #total_score = total_score + score
  #total_eo = total_eo + eo
  #total_dp = total_dp + dp

print("F1 score, DEO, DDP after 10 fold experiments are", total_score/10, total_eo/10, total_dp/10)

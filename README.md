Datasets: All the data files are processed data files obtained after running a GAT model (to be used by MCQP), along with the embeddings of different baselines is available at https://drive.google.com/drive/folders/1jPuqA3_DR4dIPtwckNy7FIVzJdQ329uS?usp=share_link . All the preproccesed data files needed for model training for various datasets is available at https://drive.google.com/drive/folders/1HL8gYg2mlKdgGKwban8GAomyoa7wUqVc?usp=share_link 

Running the Code (for our proposed approach MCQP[MultiCalibration using Quadratic Programming]) for Different Datasets:

1. Pokec-z : (a) Run the [QPPG1_EO_newtest_300K_prop_regression_50000_random_newtestdata.py] in the [pokec-z_multicalibration/src_age_region_bin_multical] folder to run MCQP-2.
             (b) Run [pokec_MCQP-2.py] in the same folder to run MCQP-1.
2. Cora: Run Run (a) Run [QPPG1_multical.py] for MCQP-2 and (b) [QPPG1_cora_MCQP-2.py] for MCQP-1 in [cora_twitter_polblog_multicalibration] folder
3. Political Blog: (a) Run [QPPG1_polblog_multical.py] for MCQP-2 and (b)   [QPPG1_polblog_MCQP-2.py] for MCQP-1 in [cora_twitter_polblog_multicalibration] folder
4. Twitter: (a) Run [QPPG1_twitter_multical.py] for MCQP-2 and (b) [QPPG1_twitter_MCQP-2.py] for MCQP-1 in [cora_twitter_polblog_multicalibration] folder 
5. NBA: (a) Run [QPPG1_multical.py] for MCQP-2 and (b)  [QPPG1_MCQP-2.py] for MCQP-1 in [nba_multicalibration/src_country_age_multical] folder


Embeddings: All the processed embeddings are present in the /embeddings folder. You need to change the path of the .bin files above mentioned python files while reading the embeddings. 

SENSITIVE_Attributes = { pokec-z: [gender, region, AGE],
                                   pol-blog :[party],
                                   twitter:[opinion],
                                   nba:[age,country],
                                   cora:[topic]  }

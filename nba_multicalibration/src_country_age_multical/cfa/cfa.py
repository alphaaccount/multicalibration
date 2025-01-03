
import ot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


class ContinuousFairnessAlgorithm():
    """
    TODO: write doc
    TODO: rewrite algorithm with float scores only!!
    """

    def __init__(self, rawData, groups, group_names, qual_attr, score_stepsize, thetas, regForOT, path='.', plot=False):
        """
        Arguments:
            rawData {dataframe} -- contains data points as rows and features as columns
            groups {dataframe} -- all possible groups in @rawData. One row contains manifestations
                                  of protected attributes, hence represents a group
            qual_attr {[str]} -- name of column that contains the scores
            score_stepsize {[int]} -- stepsize between two scores
            thetas {[type]} -- vector of parameters that determine how close a distribution is to be moved to
                               the general barycenter. One theta per group.
                               theta of 1 means that a group distribution is totally moved into the general
                               barycenter
                               theta of 0 means that a group distribution stays exactly as it is
            regForOT {[type]} -- regularization parameter for optimal transport, see ot docs for details

        Keyword Arguments:
            path {str} -- [description] (default: {'.'})
            plot {bool} -- tells if plots shall be generated (default: {False})
        """

        self.__rawData = rawData
        self.__qualityAtribute = qual_attr
        self.__groups = groups
        self.__rawDataByGroup = self._getScoresByGroup(self.__rawData, self.__qualityAtribute)
        self.__groupColumnNames = self.__rawDataByGroup.columns.values.tolist()

        # calculate bin number for histograms and loss matrix size
        self.__bin_edges = np.arange(rawData[qual_attr].min() - score_stepsize,
                                     rawData[qual_attr].max() +
                                     score_stepsize,
                                     score_stepsize)
        self.__num_bins = int(len(self.__bin_edges) - 1)

        # calculate loss matrix
        self.__lossMatrix = ot.utils.dist0(self.__num_bins)
        self.__lossMatrix /= self.__lossMatrix.max()
        self.__thetas = thetas
        self.__regForOT = regForOT

        # have some convenience for plots
        self.__groupNamesForPlots = self.__rawDataByGroup.rename(group_names, axis='columns')
        self.__plotPath = path
        self.__plot = plot

    def _getScoresByGroup(self, dataset, scoreColName):
        """
        takes a dataset with one data point per row
        each data point has a qualifying as well as >= 1 sensitive attribute column
        takes all values from column qual_attr and resorts data such that result contains all scores from
        qual_attr in one column per group of sensitive attributes.

        Arguments:
            dataset {[dataframe]} -- raw data with one data point per row
            scoreColName {[string]} -- name of column that contains scores
0.0,0.0,0.0,0.0
1.0,0.0,0.0,0.0
0.0,1.0,0.0,0.0
1.0,1.0,0.0,0.0
0.0,0.0,1.0,1.0
1.0,0.0,1.0,1.0
0.0,1.0,1.0,1.0
1.0,1.0,1.0,1.0
0.0,0.0,0.0,1.0
0.0,1.0,0.0,1.0
1.0,0.0,0.0,1.0
1.0,1.0,0.0,1.0
0.0,0.0,1.0,0.0
0.0,1.0,1.0,0.0
1.0,0.0,1.0,0.0
1.0,1.0,1.0,0.0


        Returns:
            [dataframe] -- group labels as column names and scores as column values,
                           columns can contain NaNs if group sizes are not equal
        """
        #self.__groups = pd.DataFrame({"sex":[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1], "race":[2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]})
        #self.__groups = pd.DataFrame({"gender0":[0.0,1.0,0.0,1.0], "region0":[0.0,0.0,0.0,0.0], "gender1":[0.0,0.0,1.0,1.0], "region1":[0.0,0.0,0.0,0.0]})
        protectedAttributes = self.__groups.columns.values
        print("PAPA", protectedAttributes)
        print("Details about the groups are")  
        print(self.__groups.columns.values)
        result = pd.DataFrame(dtype=float)
        print(dataset)
        print("HOHA")
        new_data = dataset.copy()
        #print(new_data)
        del new_data['score']
        '''
        bins =  np.arange(0, 120, 20)
        ind1 = pd.DataFrame(np.digitize(new_data['age0'], bins), columns = ['age0'])
        ind2 = pd.DataFrame(np.digitize(new_data['age1'], bins), columns = ['age1'])
        new_data['age0'] = ind1['age0']
        new_data['age1'] = ind2['age1']    
        dataset['age0'] = ind1['age0']
        dataset['age1'] = ind2['age1']           
        '''
        #new_data.drop(columns=['score']) this has to be written to file 
        #new_data.to_csv('../data/pokec/pokec-z/allPokec_age_gender_Groups.csv', index=False)
        print(new_data)
        #print(new_data.drop_duplicates())
        #df = new_data.groupby(['gender0','age0','gender1','age1'], as_index=False).ngroups
        #print(df)
        #print(df[df['score']>5000])
        #print(np.bincount(new_data.to_numpy())) 
        d_np, counts= np.unique(new_data.to_numpy(),axis=0, return_counts=True)
        result1 = np.column_stack((d_np,counts))
        print("Problem is", d_np)
        result_pd = pd.DataFrame(result1)
        #print(result_pd)
        #print(result_pd[result_pd[4]>500])
        self.__groups =  pd.DataFrame(d_np, columns = ['country0', 'age0', 'country1', 'age1'])  ##13000 groups can be changed if needed
        final_groups =  pd.DataFrame(d_np, columns = ['country0', 'age0', 'country1', 'age1'])
        final_groups.to_csv('allNBA_country_age_Groups.csv', index=False)
        dataset.to_csv('allNBA_country_age.csv', index=False)
        #print(d_np)
        #print(d_np.shape)
        #print(new_data.groupby(['gender0','age0','gender1','age1'], as_index=False).nunique())
        protectedAttributes = ['country0', 'age0', 'country1', 'age1']
        # select all rows that belong to one group
        for _, group in self.__groups.iterrows():  # this worked out because groups were correct, but __rawData is still not correct
            colName = str(group.values)
            #print(colName)
            copy = dataset.copy()
            for prot_attr in protectedAttributes:
                #print("HO", prot_attr)
                #print("HI", group.get(prot_attr))
                copy = copy.loc[(copy[prot_attr] == group.get(prot_attr))]
            resultCol = pd.DataFrame(
                data=copy[scoreColName].values, columns=[colName])
            #print(resultCol)
            # needs concat to avoid data loss in case new resultCol is longer than already existing result
            # dataframe
            result = pd.concat([result, resultCol], axis=1)
    
        print("Intermideate results is")
        print(result)
        return result

    def _dataToHistograms(self, data, bin_edges):
        """
        creates histogram for each column in 'data'
        excludes nans

        Arguments:
            data {pd.DataFrame} -- reference to raw data
            bin_edges {ndarray} -- bins for histogram calculation

        Raises:
            ValueError -- histograms are to be of the same length each
                          hence result cannot contain NaNs

        Returns:
            {dataframe} -- columnwise histograms
        """
        histograms = pd.DataFrame(columns=self.__groupColumnNames)
        for colName in data.columns:
            colNoNans = pd.DataFrame(data[colName][~np.isnan(data[colName])])
            colAsHist = np.histogram(
                colNoNans[colName], bins=bin_edges, density=True)[0]
            histograms[colName] = colAsHist
        print("OYE")
        print(histograms.isnull().values)

        if histograms.isnull().values.any():
            raise ValueError("Histogram data contains nans")

        print("Histograms are", histograms)

        return histograms

    def _getTotalBarycenter(self, group_histograms_raw):
        """calculates barycenter of whole dataset (self.__rawDataByGroup)

        Returns:
            ndarray -- barycenter of whole dataset
        """
        print("group_histograms_raw are")
        print(group_histograms_raw)
        # calculate group sizes in total and percent
        groupSizes = self.__rawDataByGroup.count()
        groupSizesPercent = self.__rawDataByGroup.count().divide(groupSizes.sum())

        # compute general barycenter of all score distributions
        total_bary = ot.bregman.barycenter(group_histograms_raw.to_numpy(),
                                           self.__lossMatrix,
                                           self.__regForOT,
                                           weights=groupSizesPercent.values,
                                           verbose=True,
                                           log=True)[0]
        print("Sum of total barycenter: " + str(total_bary.sum()))
        if self.__plot:
            self.__plott(pd.DataFrame(total_bary),
                         'totalBarycenter.png',
                         xLabel="raw score")
        print("final barycenter is", total_bary)

        return total_bary

    def _get_group_barycenters(self, total_bary, group_histograms):
        """compute barycenters between general barycenter and each score distribution
        (i.e. each social group)

        Arguments:
            total_bary {ndarray} -- barycenter for whole dataset
            group_histograms {dataframe} -- histogram data of each group in columns

        Returns:
            DataFrame -- barycenter for each group in columns
        """

        group_barycenters = pd.DataFrame(columns=self.__groupColumnNames)
        for groupName in group_histograms:
            # build 2-column matrix from group data and general barycenter
            groupMatrix = pd.concat([group_histograms[groupName],
                                     pd.Series(total_bary)],
                                    axis=1)
            # get corresponding theta
            print(groupName)
            print("Problem is ", group_histograms.columns.get_loc(groupName))  
            theta = self.__thetas[group_histograms.columns.get_loc(groupName)]
            # calculate barycenters
            weights = np.array([1 - theta, theta])
            group_barycenters[groupName] = ot.bregman.barycenter(groupMatrix.to_numpy(),
                                                                 self.__lossMatrix,
                                                                 self.__regForOT,
                                                                 weights=weights,
                                                                 verbose=True,
                                                                 log=True)[0]

        if self.__plot:
            self.__plott(group_barycenters,
                         'groupBarycenters.png',
                         xLabel="raw score")
        return group_barycenters

    def _calculateFairReplacementStrategy(self, group_barycenters, group_histograms_raw):
        """calculate mapping from raw score to fair score using group barycenters

        Arguments:
            group_barycenters {DataFrame} -- group barycenters, one for each social group per column

        Returns:
            DataFrame -- fair scores that will replace raw scores in self.__rawDataByGroup,
                         resulting frame is to be understood as follows: fair score at index 1 replaces
                         raw score at index 1
                         TODO: rephrase that for better understanding
        """
        # a few questions: why is a histogram for a group 20 dimensional, it is some embedding kind of
        # a thing which column has the raw score LSAT score I will put the probabilities
        # there itself, also these algorithms need to be run for pokec and movielens datasets
        # so that also has to be taken care of. I need to adapt these functions for the graph case pokec and
        groupFairScores = pd.DataFrame(columns=self.__groupColumnNames)
        for groupName in self.__groupColumnNames:
            # check that vectors are of same length
            if group_histograms_raw[groupName].shape != group_barycenters[groupName].shape:
                raise ValueError(
                    "length of raw scores of group and group barycenters should be equal")

            ot_matrix = ot.emd(group_histograms_raw[groupName].to_numpy(),
                               group_barycenters[groupName].to_numpy(),
                               self.__lossMatrix)
#             plt.imshow(ot_matrix)
#             plt.show()
            # normalize OT matrix such that each row sums up to 1
            norm_vec = np.matmul(ot_matrix, np.ones(ot_matrix.shape[0]))
            inverse_norm_vec = np.reciprocal(norm_vec)
            inverse_norm_vec = np.nan_to_num(inverse_norm_vec, copy=False)
            norm_matrix = np.diag(inverse_norm_vec)
            normalized_ot_matrix = np.matmul(norm_matrix, ot_matrix)

            # this contains a vector per group with len(score_values) entries (e.g. a score range from 1 to 100)
            # results into a group fair score vector of length 100

            groupFairScores[groupName] = np.matmul(normalized_ot_matrix,
                                                   self.__bin_edges[1:].T)
       
        print("Group Fainess Scores are", groupFairScores)     
  
        if self.__plot:
            self.__plott(groupFairScores,
                         'fairScoreReplacementStrategy.png',
                         xLabel="raw score",
                         yLabel="fair replacement")

        return groupFairScores

    def _replaceRawByFairScores(self, groupFairScores):
        '''
        replaces raw scores of individuals by their fair representation given in @groupFairScores
        fair scores are given column-wise, with one column for each group and matchings are identified
        by their indexes

        example: for a column, the original score at index 0 in @self.__bin_edges will be replaced
                 with the fair score at index 0 in @groupFairScores
        '''

        def buildGroupNameFromValues(dataRow):
            name = "[ "
            firstIter = True
            secondIter = False
            thirdIter = False
            for prot_attr in self.__groups.columns.values:
                if firstIter:
                    name += str(int(dataRow.iloc[0][prot_attr]))
                    firstIter = False
                    secondIter = True
                elif secondIter:
                    name += " " + str(int(dataRow.iloc[0][prot_attr]))
                    secondIter = False
                    thirdIter = True
                elif thirdIter:
                    name += "  " + str(int(dataRow.iloc[0][prot_attr]))
                    thirdIter = False
                else:
                    name += " " + str(int(dataRow.iloc[0][prot_attr]))
            name += "]"
            return name

        def replace(rawData, oldValue, newValue):
            #print(rawData)
            rawScores = rawData[oldValue]
            #print("rawdata is", rawData)
            groupName = buildGroupNameFromValues(rawData.head(1))
            #print("group name is", groupName)
            #print("bab",groupFairScores['[0 25 1 26]'])
            #del groupFairScores['[0 25 1 26]'] 
            fairScores = groupFairScores[groupName]
            

            for index, fairScore in fairScores.iteritems():
                range_left = self.__bin_edges[index]
                range_right = self.__bin_edges[index + 1]
                replaceAtIndex = (rawScores > range_left) & (
                    rawScores <= range_right)
                rawData.at[replaceAtIndex, newValue] = fairScore
            #print(rawData)
            return rawData
        
        print("Column values are", list(self.__groups.columns.values))
        self.__rawData = self.__rawData.groupby(list(self.__groups.columns.values),
                                                  as_index=False,
                                                  sort=False).apply(replace,
                                                                    oldValue=self.__qualityAtribute,
                                                                    newValue="fairScore")

        #if self.__plot:
        #    mpl.rcParams.update({'font.size': 24, 'lines.linewidth': 3,
        #                         'lines.markersize': 15, 'font.family': 'Times New Roman'})
            # avoid type 3 (i.e. bitmap) fonts in figures
        #    mpl.rcParams['ps.useafm'] = True
        #    mpl.rcParams['pdf.use14corefonts'] = True
        #    mpl.rcParams['text.usetex'] = True

        #    fairDataPerGroup = self._getScoresByGroup(self.__rawData, 'fairScore')
        #    ax = fairDataPerGroup.plot.kde()
        #    ax.legend(bbox_to_anchor=(1.05, 1), loc=2,
        #              borderaxespad=0., labels=self.__groupNamesForPlots)
        #    ax.set_xlabel("fair score")
            #plt.savefig(self.__plotPath + 'fairScoreDistributionPerGroup.png',dpi=100, bbox_inches='tight')

        #    bin_edges = np.linspace(groupFairScores.min().min(),
        #                            groupFairScores.max().max(),
        #                            int(self.__num_bins))
        #    group_histograms_fair = self._dataToHistograms(fairDataPerGroup,
        #                                                   bin_edges)
        #    self.__plott(group_histograms_fair,
        #                 'fairScoresAsHistograms.png',
        #                 xLabel="fair score",
        #                 yLabel="Density")
        print("final answer is")
        print(self.__rawData)
        return self.__rawData

    def __plott(self, dataframe, filename, xLabel="", yLabel=""):
        mpl.rcParams.update({'font.size': 24, 'lines.linewidth': 3,
                             'lines.markersize': 15, 'font.family': 'Times New Roman'})
        # avoid type 3 (i.e. bitmap) fonts in figures
        mpl.rcParams['ps.useafm'] = True
        mpl.rcParams['pdf.use14corefonts'] = True
        mpl.rcParams['text.usetex'] = True

        ax = dataframe.plot(kind='line', use_index=False)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                  labels=self.__groupNamesForPlots)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        #plt.savefig(self.__plotPath + filename, dpi=100, bbox_inches='tight')

    def run(self):
        """
        what is the implication of finite (integal scores)   
        TODO: write that algorithm assumes finite float scores, otherwise the algorithm doesn't work
        """

        group_histograms_raw = self._dataToHistograms(self.__rawDataByGroup,
                                                      self.__bin_edges)
        if self.__plot:
            self.__plott(group_histograms_raw,
                         'rawScoresAsHistograms.png',
                         xLabel="raw score",
                         yLabel="Density")

        print("group histograms raw are")
        print(group_histograms_raw)
        total_bary = self._getTotalBarycenter(group_histograms_raw)
        group_barycenters = self._get_group_barycenters(total_bary,
                                                        group_histograms_raw)

        fairScoreReplacementStrategy = self._calculateFairReplacementStrategy(
            group_barycenters, group_histograms_raw)
   
        print("fair-replacement-score strategy is")
        print(fairScoreReplacementStrategy)
        return self._replaceRawByFairScores(fairScoreReplacementStrategy)

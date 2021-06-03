#!/usr/bin/env python3
# Name: Beth DeVogelaere
# Date: 2021-06-03
# Github acct: edevog


import pandas as pd
import numpy as np

from scipy.stats import spearmanr
import math

import matplotlib.pyplot as plt
import seaborn as sns

"""
This program calculates the Spearman Correlation of a data frame, the Confidence Interval 
and p-value for each correlation, and creates a bar graph for each dependent variable of its 
correlation to the independent variables with the Confidence Interval as the error bars and 
the p-value as stars.
"""


class SpearmanCorrelation:
    """
    This class contains the methods to calculate the rho, ci, and p-value for a spearman correlation 
    and output a bar graph visualization
    
    Attributes:
        df = a pandas dataframe with all independent and depdent metrics as columns
        rho = a pandas dataframe of the Spearman correlation coefficients for all metrics
        ci = a pandas dataframe of the confidence interval for all Spearman correlations
        pval = a pandas dataframe of the p-values for all Spearman correlations
    """

    def __init__(self, df):
        self.df = df
        self.rho, self.pval, self.ci = self.corr_ci_p_analysis()

    def spearman_ci(self, r, n):
        """Calculates the Confidence Interval for the Spearman corrlation using the corrlation value (r) and the sample size (n)"""
        if r == 1:
            return None
        stderr = 1.0 / math.sqrt(n - 3)
        delta = 1.96 * stderr
        lower = math.tanh(math.atanh(r) - delta)
        upper = math.tanh(math.atanh(r) + delta)

        return np.array([round(r - lower, 2), round(upper - r, 2)])


    def stars(self, p):
        """Returns the number of stars based on the p-value"""
        if p <= 0.0001:
            return "****"
        elif (p <= 0.001):
            return "***"
        elif (p <= 0.005):
            return "**"
        elif (p <= 0.01):
            return "*"
        else:
            return ""


    def corr_ci_p_analysis(self):
        """
        calculates the spearman correlation of all columns in the data frame, calculates the confidence interval and p-value for each correlation, then combines all calculated values and correlation values into a single data frame.
        """

        rho = self.df.corr(method='spearman').round(4)
        ci = rho.applymap(lambda x: self.spearman_ci(x, len(self.df.index.unique()))).round(4)
        pval = self.df.corr(method=lambda x, y: spearmanr(x, y)[1]) - np.eye(*rho.shape).round(4)

        return rho, pval, ci

    def plot_correlation(self, dependent_variables, independent_variables, file_path='./', file_name_prefix='correlation'):
        """
        creates a barplot for each dependent variable of its correlation to all independent variables with the confidence interval as error bars and the p-value represented by stars
        """
        plt.style.use('seaborn-white')
        # iterate over all dependent variables
        for d in dependent_variables:
            # create a new plot for each dependent variable
            fig, ax = plt.subplots(figsize=(3,4/11*len(independent_variables)))
            # plot a horizontal bar plot with the Confidence Interval as error bars
            ax.barh(width=self.rho.loc[independent_variables, d], y=independent_variables,
                    xerr=np.stack(self.ci.loc[independent_variables, d], axis=0).T,
                    color='gray'
                    )
            # iterate over all independent variables
            for i in independent_variables:
                # store the p-value for the independent and dependent variable combination in an object
                p = self.stars(self.pval.loc[i,d])
#               # include the p-value translated into stars for each correlation in the bar plot
                if self.rho[d][i] > 0:
                    ax.text(s=p, x=self.rho.loc[i,d]+self.ci.loc[i,d][1]+0.01, y=independent_variables.index(i)-0.44, fontsize=15)
                else:
                    ax.text(s=p, x=self.rho.loc[i,d]-self.ci.loc[i,d][0]-0.01, y=independent_variables.index(i)-0.44, fontsize=15)
            # set the axis range
            plt.xlim(-1,1)
            # set the plot title
            plt.title(d)
            # save the figure to the specified directory and name
            plt.savefig(file_path+file_name_prefix+d+'.png', dpi=300, bbox_inches='tight')

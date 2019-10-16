# By Dingyi-Zhuang (dingyi.zhuang@outlook.com)
import numpy as np
import pandas as pd
import random
from itertools import combinations
from sklearn.externals import joblib
from tqdm import tqdm_notebook
from matplotlib import pyplot as plt
import numba as nb

'''
What is VCG Auction? From Wiki https://en.wikipedia.org/wiki/Vickrey%E2%80%93Clarke%E2%80%93Groves_auction:

Consider an auction where a set of identical products are being sold. Bidders can take part in the auction by announcing the maximum price they are willing to pay to receive N products. Each buyer is allowed to declare more than one bid, since its willingness-to-pay per unit might be different depending on the total number of units it receives. Bidders cannot see other people's bids at any moment since they are sealed (only visible to the auction system). Once all the bids are made, the auction is closed.

All the possible combinations of bids are then considered by the auction system, and the one maximizing the total sum of bids is kept, with the condition that it does not exceed the total amount of products available and that at most one bid from each bidder can be used. Bidders who have made a successful bid then receive the product quantity specified in their bid. The price they pay in exchange, however, is not the amount they had bid initially but only the marginal harm their bid has caused to other bidders (which is at most as high as their original bid).

This marginal harm caused to other participants (i.e. the final price paid by each individual with a successful bid) can be calculated as: (sum of bids of the auction from the best combination of bids excluding the participant under consideration) - (what other winning bidders have bid in the current (best) combination of bids). If the sum of bids of the second best combination of bids is the same as that of the best combination, then the price paid by the buyers will be the same as their initial bid. In all other cases, the price paid by the buyers will be lower.

At the end of the auction, the total utility has been maximized since all the goods have been attributed to the people with the highest combined willingness-to-pay. If agents are fully rational and in the absence of collusion, we can assume that the willingness to pay have been reported truthfully since only the marginal harm to other bidders will be charged to each participant, making truthful reporting a weakly-dominant strategy. This type of auction, however, will not maximize the seller's revenue unless the sum of bids of the second best combination of bids is equal to the sum of bids of the best combination of bids.
'''

class VCG_Auction_Process(object):
    def __init__(self,value_mat_origin):
        self.value_mat_origin=value_mat_origin
        self.value_mat=self.value_mat_origin
        self.allocation=self.value_mat.columns # For single-item condition
        self.best_price=0 #Initialization
    @nb.jit
    def who_win(self):
        '''
        Pick up the winners for each item, if bids are the same, then a winner will be randomly picked
        '''
        self.winner_list=[] #Bidder index for each winner of the item
        self.second_price_list=[]
        winner_temp=[]
        for item_set in self.allocation:
            item_represent=item_set[0] #Because all other items in the set share the same value
            winner_temp.append(np.where(self.value_mat[item_represent]==np.max(self.value_mat[item_represent]))[0])
        for item_set_index,winner in enumerate(winner_temp):
            item_set=self.allocation[item_set_index] # The item set like [0,1]
            item_represent=item_set[0]# Representation
            if len(winner)>1:
                #print("Item set {:} has multiple winner: {:}".format(item_set,winner))
                random.seed(10)
                self.winner_list.append(random.choice(winner))
                second_price=sorted(self.value_mat[item_represent])[-2]
                # Add the payment of each winner, pay your value if the "same" occur
                self.second_price_list.append(second_price) 
            else:
                self.winner_list.append(winner[0])
                second_price=sorted(self.value_mat[item_represent])[-2]
                self.second_price_list.append(second_price)
        #print(self.value_mat)
        #print(self.winner_list)
        return self.winner_list,self.second_price_list
    
    def winner_price(self):
        '''
        Calculate the price that the winner does to other agents and the mechanism will charge the price for each winner
        
        Make sure function who_win is run in advance
        '''
        self.welfare_list=[] # List for the welfare contributed by the winner. the sum of this list is the social welfare
        self.price_list=[] # List for the price that the winner charged, for losers they don't pay
        value_winner_list=[]
        value_without_winner_list=[]
        # Get the value list of winner
        # With the winners
        for item_set_index,winner in enumerate(self.winner_list):
            item_set=self.allocation[item_set_index] # The item set like [0,1]
            item_represent=item_set[0]# Representation
            value_winner_list.append(self.value_mat[item_represent].iloc[winner])
        self.welfare_list=value_winner_list
        # Without the winners
        value_without_winner_list=self.second_price_list
        # Contribution of the winner
        
        ctrib_list=list(map(lambda x: x[0]-x[1],zip(value_winner_list,value_without_winner_list)))
        # Price of the winner charged by the mechanism
        self.price_list=list(map(lambda x: x[0]-x[1],zip(value_winner_list,ctrib_list)))
        return self.price_list,self.welfare_list
    def allocate_items(self,allocation):
        '''
        Allocate different sets of items
        '''
        self.allocation=allocation# Consider a finxed allocation case
        self.update_value_mat()
    @nb.jit
    def update_value_mat(self):
        '''
        Update the value of bidders for item sets as the value of item sets are the maximum value inside
        '''
        self.value_mat=self.value_mat_origin.copy()
        for set_index,item_set in enumerate(self.allocation):
            for bidder,value in self.value_mat.iterrows():
                self.value_mat.iloc[bidder][item_set]=np.max(self.value_mat.iloc[bidder][item_set])
                
    def find_best_allocation_price(self,current_price):
        if current_price>=self.best_price:
            self.best_allocation=self.allocation # Initialization
            self.best_price=current_price
        
    def begin(self,possible_allocations):
        f=open('allocation_price.txt','w+')
        for allocation in tqdm_notebook(possible_allocations): #Define your possible_allocations here:
            self.allocate_items(allocation)
            self.who_win()
            self.winner_price()
            self.find_best_allocation_price(np.sum(np.sum(self.price_list)))
            f.write('For Allocation: {:} Pirce charged by mechanism {:} and total {:}\n'.format(self.allocation,self.price_list,np.sum(self.price_list)))
        f.close()
        print('Best allocation: {:} and the mechanism charges {:}'.format(self.best_allocation,self.best_price))

if __name__=='__main__':
    value_mat_origin=pd.read_table('Assignment#2-Q8-values.txt',delimiter=' ',header=None,sep='\\t')
    value_mat_origin=value_mat_origin.drop([8],axis=1)
    print('Input value:', value_mat_origin)
    possible_allocations=joblib.load('possible_allocations_7.asv')
    print('Some possible allocations:',possible_allocations[:5])
    vcg=VCG_Auction_Process(value_mat_origin)
    vcg.begin(possible_allocations)
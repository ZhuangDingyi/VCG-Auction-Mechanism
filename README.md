# VCG-Auction-Mechanism
Implementation of Vickrey–Clarke–Groves auction (VCG auction)

This is my assignment code from *COMP 553 Algorithmic Game theory (Adrian Vetta)* in McGill University implemented with Python

The whole VCG auction is implemented by class **VCG_Auction_Process** in *VCG.py*, you only need to determine the input allocations (which is heavy computation to enumerate). 

Initialze the class with your original value matrix for bidders (remember that bidders always bid truthfully in VCG mechanism). If you Prof. Adrian provides a txt for you in *Assignment#2-Q8-values.txt* , also, I generate some of the allocations in *.asv* file, just joblib.load them and run *begin(your_allocations)*

Enjoy game theory!

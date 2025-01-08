# FA-01 
It's similar to linear optimisation & mostly its related to the matrix things.

Main to pics 
1. Fraud 
2. Spectral
3.  GNN
4. trust, Index.

# FA-02
 - use case → to check with set of vectors that this book is manipulated or genuine. {Book-s can be anything's }
	 - but when observed that the probability of  selecting page from 1000 book pages is very different& not equally divided.
	 - =={ newcomb Belford }==
	 ![benfold formula](Fraud%20analysis-1.png) 
 When Benford’s Law is followed:
1. Financial Data: Tax records, stock prices, or accounting data.
2. Scientific Data: Physical constants, populations, or data from experiments.
3. Natural Phenomena: Earthquake magnitudes, river lengths, or heights of mountains.
4. Business Metrics: Sales figures, production quantities, or inventory levels.

![](Fraud%20analysis.png)

When Benford’s Law does NOT apply:
1. Data with fixed ranges: Heights of humans, test scores, or data constrained to specific intervals.
2. Random or uniform distributions: Lottery numbers, random number generators, or sequential data.
3. Arbitrary assignments: Phone numbers or ZIP codes.
The law works best for datasets that combine multiple scales or orders of magnitude. 

# Fa-02
## scale invariant 
Basically they are saying as as benfold law says that number frequently appear mostly in order of formula where 
Number 1 will appear more time then the other number & that will remain true even is I've convert that number for example conversion of dollars to rs.

![](Fraud%20analysis%20distribuition.png)

- So basically they are saying that even if we check multiply starting digit of scientific notation it will follow the same distribution

![](Fraud%20analysis%20distribuition-1.png)
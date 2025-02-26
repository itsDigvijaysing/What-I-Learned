# FA-01 
It's similar to linear optimisation & mostly its related to the matrix things.

Main to pics 
1. Fraud 
2. Spectral
3.  GNN
4. trust, Index.

# FA - 02
 - use case → to check with set of vectors that this book is manipulated or genuine. {Book-s can be anything's }
	 - but when observed that the probability of  selecting page from 1000 book pages is very different& not equally divided.
	 - =={ newcomb Belford }==
	 ![benfold formula](Fraud%20analysis-1.png) 
 When Benford’s Law is followed:
1. Financial Data: Tax records, stock prices, or accounting data.
2. Scientific Data: Physical constants, populations, or data from experiments.
3. Natural Phenomena: Earthquake magnitudes, river lengths, or heights of mountains.
4. Business Metrics: Sales figures, production quantities, or inventory levels.

![FA Digits spread as per research](Fraud%20analysis.png)

When Benford’s Law does NOT apply:
1. Data with fixed ranges: Heights of humans, test scores, or data constrained to specific intervals.
2. Random or uniform distributions: Lottery numbers, random number generators, or sequential data.
3. Arbitrary assignments: Phone numbers or ZIP codes.
The law works best for datasets that combine multiple scales or orders of magnitude. 

# FA - 03
## scale invariant 
Basically they are saying as as benfold law says that number frequently appear mostly in order of formula where 
Number 1 will appear more time then the other number & that will remain true even is I've convert that number for example conversion of dollars to rs.
![FA Digits scale](Fraud%20analysis%20distribuition.png)

- So basically they are saying that even if we check multiply starting digit of scientific notation it will follow the same distribution
![Some Algo](Fraud%20analysis%20distribuition-1.png)
So basically log is continuous that's why we use it.

# FA - 04
- equally likey digits are not scale invarients.
 - benfold law hinges on the simple observation that, it a hat is a covered evenly in black & white strips then half of black stripes will be same as White when curve is smooth & spread several order magnitude.
![FA_U Shape](Fraud%20analysis%20black%20and%20white.png)

# FA - 05
- Ranking Web
- Page Ranking
- Mathematical Way of Solving Page Ranking with actual example when google wants to create page ranking (Matrix solving way)

![FA 05](Fraud%20analysis%204.png)
# FA - 06

- They tought about this concepts in more depth

# FA - 07

- Ranking the page using the Logarithmic approach
- Then they had Modified Page Rank-
	- Z=0.85*M + 0.15*Y
- Algorithm Convergence
	- ZU = Z((alpha)V + X)
	- Z(aplha)V+ZX
	- ==(alpha)V + X== (Either alpha Increases of X increase)
	- ==(alpha)V + (PU + X)== 0<P<=1

# FA - 08

![](Fraud%20analysis%208.png)

# FA - 10

![](Fraud%20analysis%2010.png)
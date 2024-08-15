Links: [IIT Hyderabad](IIT%20Hyderabad.md)

# FoML - 01

## What is Machine Learning:
- It's Model which makes prediction & decisions based/from data.
- A computer Program which is said to learn experience E from past & perform task F using past experiences & we use this E and F Tasks Experience to do new Tasks.

> Traditional Program
> Data+Program => Compute => Output

> Machine Learning Program
> Data+Output => Compute => Program

## Where ML is not required:
- Where no data or very low data is present
- when there is specific input/output already present (Discrete Algo)

## Supervised Learning:

- Predict 'Y' when given an input 'X' (X: Labelled data):
	- For Categorical Y: Classification
		- Discrete Values, e.g. Male or Female, we give input X with Tag (Label) 'M/F'
	- For Real-Valued Y: Regression
		- Continuous Values, e.g. Age, we give input X with value (0 to 100)

## Unsupervised Learning:

- Predict 'Y' when given raw data input 'X' (Non Labelled data), It Create an internal representation of the input (e.g. Clustering, dimensionality)
	- This is imp in machine learning as getting labelled data is expensive & difficult.
	- For Categorical Y: Clustering
		- We give raw data & based on it's own analysis and information differentiation it will predict the output. e.g. M/F
	- For Continuous Y: Dimensionality Reduction
		- It automatically analyze data & do the process. e.g. Image reduction.
## Reinforcement Learning:
- It works on Reward & Punishment Sequence of output & learns which solution is correct/High Accuracy.

> Chat GPT: is Self Supervised Learning (New Way)
> Basically 'Unsupervised Learning' but can also work as 'Supervised Learning'

> We in ML we try to Approximate Functions:
> Supervised -> f(x) -> y
> Unsupervised -> f^ (x) -> x^
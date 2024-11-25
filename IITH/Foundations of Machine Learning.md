Links: [IIT Hyderabad](IIT%20Hyderabad.md), [University of Waterloo Lectures](https://www.youtube.com/watch?v=GouhgbE5gPk&list=PLdAoL1zKcqTW-uzoSVBNEecKHsnug_M0k), [Stanford Lectures Andrew NG](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU), [Rishabh Iyer - GitHub](https://github.com/rishabhk108), 

# FoML - 01

## Machine Learning Overview

**Definition:**
- **Machine Learning (ML)**: A model that makes predictions or decisions based on data. It involves a computer program that learns from past experiences (E) to perform tasks (F) and applies this knowledge to new tasks.

**Traditional vs. ML Program:**

- **Traditional Program**: 
  - Data + Program → Compute → Output
- **Machine Learning Program**: 
  - Data + Output → Compute → Program

- **Artificial Intelligence (AI)**: The broad field encompassing various sub fields like Machine Learning (ML), Natural Language Processing (NLP), Robotics, etc.
    - **Machine Learning (ML)**: A prominent sub field of AI, focused on developing algorithms that allow computers to learn from data.
        - **Supervised Learning**
        - **Unsupervised Learning**
        - **Reinforcement Learning**

## When ML is Not Required:
- **Insufficient Data**: When there is no or very low data available.
- **Specific Inputs/Outputs**: When discrete algorithms can handle well-defined tasks.

## Types of Learning:

1. **Supervised Learning:**
   - **Purpose**: Predict 'Y' given an input 'X' (Labelled Data).
   - **Types**:
     - **Classification**: For categorical Y (Discrete values). Example: Gender (Male/Female).
     - **Regression**: For real-valued Y (Continuous values). Example: Age (0 to 100).

2. **Unsupervised Learning:**
   - **Purpose**: Predict 'Y' given raw input 'X' (Non-labelled Data). It creates an internal representation of the input.
   - **Types**:
     - **Clustering**: For categorical Y. Example: Grouping data into categories like M/F based on its features.
     - **Dimensionality Reduction**: For continuous Y. Example: Reducing the size of image data while preserving important features.

3. **Reinforcement Learning:**
   - **Purpose**: Learns through a reward and punishment mechanism to determine the most accurate solution or behavior.

> ChatGPT and Learning Types:
> - **ChatGPT**: Utilizes self-supervised learning. It is a hybrid of supervised and unsupervised learning, leveraging both approaches.

> Function Approximation in ML:
> - **Supervised Learning**: Approximate function \( f(x) \to y \).
> - **Unsupervised Learning**: Approximate function \( f^ (x) \to x^ \).

## Linear Algebra Fundamentals for Machine Learning

**1. Vectors**

- **Definition**: An ordered list of numbers that can represent data points or features.
- **Example**: **v** = [4, 2] (a vector in 2D space).

**2. Operations with Vectors**

- **Addition**: Combine vectors component-wise.
  - **v** = [1, 2] + **w** = [3, 4] = [4, 6]
  
- **Scalar Multiplication**: Multiply each component of a vector by a scalar.
  - 2 * **v** = 2 * [1, 2] = [2, 4]
  
- **Dot Product**: Multiply corresponding components and sum them.
  - **v** · **w** = [1, 2] · [3, 4] = (1 * 3) + (2 * 4) = 11
  
- **Norm**: The length or magnitude of a vector.
  - ||**v**|| = √(1² + 2²) = √5
  
- **Unit Vector**: A vector with a norm of 1.
  - **u** = **v** / ||**v**|| = [1/√5, 2/√5]

**3. Matrices**

- **Definition**: A rectangular array of numbers arranged in rows and columns.
  
- **Matrix Operations**:
  - **Addition**: Combine matrices component-wise.
  - **Scalar Multiplication**: Multiply each element by a scalar.
  - **Matrix Multiplication**: Multiply rows by columns.
  - **Transpose**: Flip rows and columns.

**4. Linear Transformations**

- **Definition**: Operations that map vectors from one space to another, often represented by matrices.
- **Example**: Scaling, rotation, and translation operations in 2D or 3D space.

**5. Vector Spaces**

- **Definition**: A collection of vectors that can be added together and scaled.
- **Basis and Dimension**: A basis is a set of linearly independent vectors that span the vector space. The dimension is the number of vectors in the basis.

**6. Eigenvalues and Eigenvectors**

- **Eigenvalues**: Scalars that provide insights into the properties of transformations.
- **Eigenvectors**: Vectors that remain in the same direction after a transformation.
- **Example**: For a matrix \( A \), if \( A \mathbf{v} = \lambda \mathbf{v} \), then \( \lambda \) is an eigenvalue, and \( \mathbf{v} \) is the corresponding eigenvector.

**7. Applications in Machine Learning**

- **Feature Representation**: Vectors represent features of data points.
- **Model Parameters**: Weights and biases in models.
- **Dimensionality Reduction**: Techniques like PCA use eigenvalues and eigenvectors to reduce the number of features.
- **Transformations**: Used for operations like scaling, rotation, and projection.

# FoML - 02

## Training is the process of making the system able to learn.
### Resources: [FoML PDF 2](https://classroom.google.com/u/3/c/NzAyODg3MDIxNTQw/m/NzAzNDUxMjI5NDEx/details)
#### 1. **No Free Lunch Rule**
- **Concept**: There is no single model or algorithm that works best for every problem. The effectiveness of a model depends on the specific problem and data at hand.
- **Implication**: It's essential to experiment with different models and approaches, as no one model will universally outperform others across all tasks.

#### 2. **Training Set & Testing Set Distribution**
- **Concept**: The training set (used to train the model) and the testing set (used to evaluate the model) may not always come from the same data distribution.
- **Challenge**: If the distributions differ, the model's performance on the testing set may not accurately reflect its performance in real-world scenarios.
- **Solution**: We may need to make assumptions about the data or employ techniques to handle distribution shifts (e.g., domain adaptation).

---

## Types of Models

#### 1. **Inductive vs. Transductive Learning**
- **Inductive Learning**:
  - The model learns a general rule from the training data and applies it to unseen data.
  - **Example**: Most traditional machine learning models.
- **Transductive Learning**:
  - The model focuses on making predictions only for the specific examples in the testing set without deriving a general rule.
  - **Example**: Semi-supervised learning where the testing set is partially known during training.

#### 2. **Online vs. Offline Learning**
- **Online Learning**:
  - The model is updated continuously as new data arrives.
  - Suitable for environments where data comes in streams.
  - **Example**: Spam filtering in email.
- **Offline Learning**:
  - The model is trained on a fixed dataset before being deployed.
  - **Example**: Traditional batch learning algorithms.

#### 3. **Generative vs. Discriminative Models**
- **Generative Models**:
  - Learn the joint probability distribution \( P(X, Y) \) and can generate new data points.
  - **Example**: Naive Bayes, Gaussian Mixture Models.
- **Discriminative Models**:
  - Focus on learning the decision boundary between classes by modeling \( P(Y|X) \).
  - **Example**: Logistic Regression, Support Vector Machines.

#### 4. **Parametric vs. Non-Parametric Models**
- **Parametric Models**:
  - Characterized by a finite number of parameters, often with assumptions about the data distribution.
  - **Example**: Linear Regression, Logistic Regression.
- **Non-Parametric Models**:
  - Do not assume a fixed number of parameters, and the model complexity can grow with the data.
  - **Example**: k-Nearest Neighbors, Decision Trees.

---

## Classifier Evaluation Concepts

![Model Selection](../Archive/Attachment/Estimating%20ML.png)

#### 1. **Training Error**
- **Definition**: The error rate of a model on the training set.
- **Goal**: Minimizing training error is necessary but not sufficient for good generalization.
- ![Training Error](../Archive/Attachment/Training%20error.png)

#### 2. **Generalization Error**
- **Definition**: The error rate of a model on unseen data (testing set).
- **Goal**: A low generalization error indicates that the model is performing well on new, unseen data.
- $$E_{gen}\underbrace{=\int}_{\text{over all possible x,y}}\underbrace{error(f_D(\mathbf{x}),y)}_{\text{error as before}}\underbrace{p(y,\mathbf{x})}_{\text{how often we expect to see such x and y}}d\mathbf{x}$$

#### 3. **Vector Space**
- **Concept**: In many models, particularly in text classification and natural language processing, data is represented in a high-dimensional vector space.
- **Importance**: The representation of data in vector space affects model performance and interpretation.

#### 4. **Underfitting and Overfitting**
- **Underfitting**:
  - Occurs when a model is too simple and fails to capture the underlying structure of the data.
  - Leads to high training and generalization errors.

![Under fitting & Over fitting](../Archive/Attachment/Overfitting%20and%20Overfitting.png)

- **Overfitting**:
  - Occurs when a model is too complex and captures noise in the training data.
  - Leads to low training error but high generalization error.

#### 5. **Training Set, Validation Set, Testing Set**
- **Training Set**: The data used to train the model.
- **Validation Set**: The data used to tune model parameters (e.g., hyper parameters) and prevent overfitting.
- **Testing Set**: The data used to evaluate the final model's performance.

#### 6. **Stratified Sampling**
- **Concept**: Ensuring that each class is represented proportionally in the training, validation, and testing sets.
![Stratified Sampling](../Archive/Attachment/Stratified%20Sampling.png)
- **Importance**: This technique is particularly useful when dealing with imbalanced datasets.

#### 7. **K-Fold Cross Validation**
- **Concept**: A technique to assess model performance by dividing the dataset into \( k \) equal parts (folds). The model is trained on \( k-1 \) folds and tested on the remaining fold. This process is repeated \( k \) times, each time with a different fold as the testing set.
- **Benefit**: Provides a more robust evaluation of model performance by reducing variance and bias in the error estimate.

# FoML - 03

## Cross-Validation
### Leave-One-Out (LOO) / N-Fold Cross Validation
- **N-Fold Cross Validation**:
  - The dataset is divided into **N folds** N =  Data points of set.
  - We train the model on **N-1 folds** and test on the remaining fold.
  - This process is repeated **N times**, each time with a different fold as the test set.
  - The final **cross-validation accuracy** is computed by averaging the accuracies from all test sets.

### Resubstitution
- **Resubstitution** refers to using the **same dataset** for both training and testing.
- This can lead to **overestimation of model performance** because the model has already seen the data it's being tested on.

## Evaluation Metrics
### For Classification
- The objective is to evaluate how well a model classifies instances into categories, such as positive or negative.

|                     | **Predicted Positive** | **Predicted Negative** |
| ------------------- | ---------------------- | ---------------------- |
| **Actual Positive** | True Positive (TP)     | False Negative (FN)    |
| **Actual Negative** | False Positive (FP)    | True Negative (TN)     |


### Accuracy
- Formula:  
$$  [
  \text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN}
  ]$$
  where:
  - **TP**: True Positive
  - **FP**: False Positive
  - **TN**: True Negative
  - **FN**: False Negative
- **Accuracy = 1 - error**  
  - Useful when class distribution is balanced.
  - May not be useful in cases where there is a large **class skew** or **imbalanced datasets**.
    - For example, in cases where 98% of data belongs to one class, accuracy might be misleading.

### Precision
- Formula:  
$$  [
  \text{Precision} = \frac{TP}{TP + FP}
  ]$$
  - Measures the proportion of **true positives** out of the predicted positives.
  - Useful when the cost of **false positives** is high.

### Recall (True Positive Rate)
- Formula:  
$$  [
  \text{Recall} = \frac{TP}{TP + FN}
  ]$$
  - Measures the proportion of **true positives** out of the actual positives.
  - Also known as **sensitivity** or **hit rate**.

### False Negative Rate (Miss Rate)
- Measures how often we incorrectly classify a positive instance as negative.
$$  [
  \text{Miss Rate} = \frac{FN}{TP + FN}
  ]$$
  
### False Positive Rate (Fall-out)
- Measures how often we incorrectly classify a negative instance as positive.
$$  [
  \text{False Positive} = \frac{FP}{TN + FP}
  ]$$
  
### Multi-Class Problems
- In real-world applications, the cost associated with different types of errors can vary.
  - **Detection cost**:  
    $$[
    \text{Cost} = C_{fp} \cdot FP + C_{fn} \cdot FN
    ]$$
    - Where:
      - \(C_{fp}\): Cost of false positive
      - \(C_{fn}\): Cost of false negative
    - In some cases, **false negatives** are costlier than **false positives** (e.g., detecting a serious illness).

## ROC Curve (Receiver Operating Characteristic)
- The **ROC curve** is a graphical plot that illustrates the diagnostic ability of a binary classifier as its discrimination threshold is varied.
- **X-axis**: False Positive Rate (Fall-out)  
- **Y-axis**: True Positive Rate (Recall)
- Many algorithms output a **confidence score** \( F(x) \), so we can adjust the **threshold** \( t \) to classify:
  - If \( F(x) > t \), classify as **positive**.
  - If \( F(x) \leq t \), classify as **negative**.
![ROC Curve](Foundations%20of%20Machine%20Learning.png)

- **Purpose of ROC**:
  - Visualizes the trade-off between **sensitivity (recall)** and **specificity** for different thresholds.
  - A **perfect classifier** would have an ROC curve that passes through the upper left corner (100% sensitivity, 0% false positives).

## Distance Metrics
### Lp Distance
- A general distance metric family, where \( p \) represents the order.
  - **Euclidean Distance** is a special case with \( p = 2 \).

### Euclidean Distance (Mean Square Distance)
- Formula:
$$  [
  d(p,q) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}
  ]$$
- Measures the straight-line distance between two points in Euclidean space.

# FoML - 04

## Empirical Error
- **Empirical Error** refers to the error rate of a model when evaluated on a **training dataset**.
- It is the average loss over the training examples and is often minimized to create models.
- Formula (in case of classification):
$$  [
  \text{Empirical Error} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{I} \left( f(x_i) \neq y_i \right)
  ]$$
  where:
  - ( f(x_i) \) is the predicted output for the input \( x_i \)
  - ( y_i \) is the actual label of the input
  - ( \mathbb{I} ) is the indicator function, which is 1 if \( f(x_i) \neq y_i \) and 0 otherwise.

## Empirical Risk Minimization (ERM)
- **ERM** is a strategy to minimize the empirical error.
- The goal is to find a model that minimizes the empirical risk (or error) on the training data.
- While ERM helps to fit the training data well, it does not always generalize to unseen data. Overfitting can occur if the model becomes too complex.

## K-Nearest Neighbors (KNN) (Transductive)
- **KNN** is a **lazy learning algorithm**, meaning it doesn't learn an explicit model. Instead, it stores all training data and classifies new data points based on proximity to existing points.
- For each new point, the algorithm finds the **K nearest neighbors** (based on a distance metric like Euclidean distance) and classifies the new point by majority voting.

### Choosing the value of K
- **Small K values**:
  - Sensitive to **noise** and outliers.
  - The decision boundary can be more **complex** and vary widely between points.
- **Large K values**:
  - The decision boundary is more **smooth**, but may include points from other classes, leading to **misclassification**.
  - The model becomes less sensitive to noise but risks making incorrect generalizations.

### Occam's Razor Principle
- When multiple models produce similar performance, **choose the simplest model**.
- This principle encourages avoiding overcomplicated models that may overfit the training data.

## Inductive vs. Transductive Learning
### Inductive Learning
- In **inductive learning**, the model learns an **explicit target function** from the entire training set.
- The trained model is then applied to any unseen test data to predict outcomes.
- **Examples**: Decision trees, SVMs, etc.

### Transductive Learning
- In **transductive learning**, the model directly makes predictions for the **specific test data** provided during training.
- It focuses on mapping the training data to the given test data rather than generalizing to all possible test data.
- **Examples**: KNN, some semi-supervised learning methods.

## Lazy Learner (KNN)
- **KNN** is a **lazy learner** because it **doesn't construct an explicit model**.
- Instead, it waits until it receives a query and then computes the prediction based on the training data.
- Once the prediction is made, the algorithm returns to its idle state without any further updates or learning.

## Bayes Optimal Error
- The **Bayes Optimal Error** is the **lowest possible error** that can be achieved by any classifier for a particular problem, assuming the true underlying probability distribution is known.
- It represents the **theoretical limit** of a classifier's performance.

## Voronoi Diagrams (KNN)
- **Voronoi Diagrams** can be used to represent regions around each data point in **KNN**.
- Each region contains all points closest to a particular training point.
- The diagram divides the space into cells, each containing all the locations closer to one training data point than to any other.

## Normalization of Data
- **Normalization** is important for distance-based methods like **KNN** because features may have different scales.
- Without normalization, features with larger scales can dominate the distance calculation, leading to incorrect classifications.
- **Methods**:
  - **Min-Max Scaling**: Rescale features to a range [0,1].
  - **Z-Score Normalization**: Rescale features to have a mean of 0 and a standard deviation of 1.

## Decision Trees
- **Decision Trees** are a **non-parametric** method used for classification and regression tasks.
- It models decisions as a series of nodes that split data based on feature values.
- **Divide and Conquer Strategy**: It recursively divides the dataset into smaller subsets based on specific features and constructs a tree-like structure for classification.
- Advantages:
  - Easy to interpret.
  - Handles both categorical and continuous data.
- Drawbacks:
  - Can easily overfit if not pruned.
  - Pruning helps to simplify the tree and prevent overfitting.

# FoML - 05

## Handling Categorical Variables
- For **categorical variables**, assigning **numerical values** (like 1, 2, 3, etc.) can be misleading.
  - The model might treat the numerical values as continuous and apply operations like **Euclidean distance**, which would lead to incorrect classifications.
  
### One-Hot Encoding
- A common method to handle categorical variables is **one-hot encoding**.
  - Each category is represented as a **binary vector**, where only one element is "1" and the rest are "0".
  - This ensures the model **doesn't compare categories incorrectly** by treating them as numbers.

## K-Nearest Neighbors (KNN) and Hyperparameters
- In **KNN**, we define **hyperparameters**, such as the value of **K** (the number of neighbors).
  - **Hyperparameters** are set manually before training and can be tuned using techniques like **cross-validation**.
- Normal parameters (e.g., distances between points) are determined by the **algorithm** based on the model.

## Decision Trees

### Divide and Conquer Strategy
- **Decision Trees** use a **divide and conquer** approach to recursively split the data into smaller subsets.
- The **split** is based on feature values and the goal is to maximize the **purity** of the resulting subsets.

### Decision Nodes
- **Univariate Decision Nodes**: Splits the data based on a single feature.
- **Multivariate Decision Nodes**: Splits the data based on a combination of features.

### Leaves
- The **leaf nodes** in a decision tree represent the final output of the tree:
  - **Classification Trees**: The leaf node represents a class label.
  - **Regression Trees**: The leaf node represents a continuous value.

### Greedy Learning
- **Decision Trees** use a **greedy learning** approach to find the **best split** at each node.
  - At each step, the algorithm selects the split that results in the greatest **reduction in impurity**.

## Measure of Impurity
- **Entropy** is a common measure of impurity in decision trees.
  - Formula:
    $$[
    I_m = - \sum_{i} p_{mi} \log_2(p_{mi})
    ]$$
  - Where:
    - \( p_{mi} \) is the proportion of instances belonging to class \( i \) at node \( m \).
  - Node \( m \) is **pure** if \( p_{mi} = 0 \) or \( p_{mi} = 1 \), meaning all instances at the node belong to one class.

### Information Gain
- **Information Gain** measures the **reduction in impurity** after a split.
  - The goal is to **maximize information gain** at each split.
  - If the impurity after splitting is lower than before, the split is deemed useful.

### Example of Impurity and Information Gain
- **Entropy** is often used as the impurity measure in decision trees.
- **Information gain** is the difference in impurity before and after the split.

## Overfitting and Generalization in Decision Trees
- **Overfitting** occurs when a model becomes too complex and captures noise in the training data, leading to poor performance on unseen data.
- **Generalization** is the model's ability to perform well on unseen data.

### Pruning
- To avoid overfitting, we can apply **pruning** to decision trees:
  - **Pre-Pruning** (Early Stopping):
    - Stop growing the tree before it becomes overly complex, usually by limiting the depth of the tree or the minimum number of samples required to split a node.
  - **Post-Pruning**:
    - First, grow the complete tree, then **chop off** the last connected nodes (subtrees) to simplify the tree and improve generalization.

# FoML - 06

## Bayes Theorem
- **Bayes Theorem** provides a way to calculate the **posterior probability** of a hypothesis based on prior knowledge and evidence.
- Formula:
$$  [
  P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}
  ]$$
  Where:
  - \( P(H|E) \): Posterior probability of hypothesis \( H \) given the evidence \( E \).
  - \( P(E|H) \): Likelihood of the evidence given the hypothesis.
  - \( P(H) \): Prior probability of the hypothesis.
  - \( P(E) \): Marginal probability of the evidence (normalization factor).

## Naive Bayes Assumption
- The **Naive Bayes Algorithm** assumes **independence** between the features (attributes) of the data.
  - This assumption simplifies calculations but is often unrealistic in real-life scenarios where attributes can be dependent.
- Despite this "naive" assumption, the algorithm works surprisingly well in many applications, as it reduces computational complexity.

### Why Assume Independence?
- Without the independence assumption, calculating probabilities considering all possible dependencies between features becomes computationally expensive.
  - The **computational cost** of modeling dependencies grows exponentially with the number of features.
  - By assuming independence, we can reduce this to a manageable level, allowing the algorithm to run efficiently on standard hardware (like normal CPUs).

## Posterior, Prior, and Likelihood
- **Posterior Probability**: The probability of a hypothesis (class) given observed evidence.
- **Prior Probability**: The initial belief or probability of the hypothesis before observing the evidence.
- **Likelihood**: The probability of the evidence given that the hypothesis is true.

### Maximum Likelihood Hypothesis
- If we assume that all hypotheses (or classes) are **equally probable** before observing any evidence (a **uniform prior**), the computation of the posterior simplifies.
- The class with the highest likelihood becomes the **maximum likelihood hypothesis**, which is the class that Naive Bayes assigns to the new data.

## Naive Bayes Algorithm Steps
1. **Calculate Prior Probability** for each class.
2. **Calculate Likelihood** of the evidence for each class (using the Naive assumption of feature independence).
3. **Compute Posterior Probability** for each class using Bayes Theorem.
4. **Classify** the instance by choosing the class with the highest posterior probability.

## Pros of Naive Bayes Algorithm
1. **Simple and Fast**:
   - Due to the independence assumption, the algorithm has low computational complexity.
   - It works well even with a large number of features.
2. **Performs well in many real-world applications**:
   - Surprisingly effective for problems like text classification and spam detection.
3. **Handles both binary and multiclass classification**.
4. **Requires less training data** compared to other algorithms like logistic regression or SVMs.

## Cons of Naive Bayes Algorithm
1. **Strong independence assumption**:
   - Naive Bayes assumes that features are independent of each other, which is rarely true in real-world datasets.
   - When the features are highly dependent, Naive Bayes may not perform well.
2. **Zero frequency problem**:
   - If a category (value) in a feature was not seen in the training data for a given class, it will lead to **zero probabilities**.
   - This can be handled by techniques like **Laplace smoothing**.

## Underflow Prevention
- When calculating the probabilities for a large number of features, multiplying many small probabilities together can lead to **numerical underflow**.
  - To prevent this, we often compute the **logarithms** of probabilities instead of directly multiplying them.
  - The logarithmic form of Bayes Theorem is:
 $$   [
    \log P(H|E) = \log P(H) + \sum_{i=1}^{n} \log P(E_i|H)
    ]$$
    where \( E_i \) is the \(i\)-th feature.

## Overcoming the Independence Assumption
- While the independence assumption is a limitation, several techniques can help improve Naive Bayes' performance when the features are dependent:
  1. **Feature selection**: Remove or merge highly correlated features.
  2. **Hierarchical Bayesian models**: Introduce dependencies between features, though this increases complexity.
  3. **Bayesian networks**: A more complex model that explicitly represents dependencies between features.

## Example Applications
- **Text Classification** (spam detection, sentiment analysis).
- **Document Categorization**.
- **Medical Diagnosis** (e.g., predicting disease based on symptoms).


# FoML - 07

## Overview of AI, Machine Learning, and Deep Learning
- **Computer Science (CS)**: A vast field that includes various subfields, one of which is **Artificial Intelligence (AI)**.
- **AI**: In AI, both **Machine Learning (ML)** and **Deep Learning (DL)** are key concepts.
  - **Machine Learning (ML)**: Focuses on creating systems that learn from data.
  - **Deep Learning (DL)**: A subset of ML that uses neural networks to model complex patterns in data.

## Neural Networks
- **Inspiration**: Neural networks are inspired by how the human brain works. They attempt to mimic the brain's structure, where neurons (brain cells) process and pass information.
- **Purpose**: Neural networks are used for tasks like classification, regression, and pattern recognition. Their ability to learn from data and adjust themselves makes them powerful tools in AI and ML.

## Rosenblatt’s Algorithm (Perceptron)
- **Rosenblatt**: Proposed the first algorithm for a **single-layer neural network**, known as the **Perceptron**.
- **Perceptron**: It is a linear classifier, which classifies data points by learning a hyperplane that separates two classes.
- **Complexity**: While a single neuron can perform simple linear classification, by using multiple neurons (forming a network), we can solve more complex, non-linear problems.

## Working of Neural Networks
1. **Initialization**: The network is initialized with **random weights**.
2. **Training**: The network is trained over millions of iterations:
   - Data is passed through the network (forward pass).
   - The **weights are adjusted** gradually to increase accuracy.
3. **Output**: After training the model for many iterations, the network begins producing outputs similar to what we expect (desired outcomes).
4. **Improvement**: The accuracy of the neural network improves gradually as the model continues learning from the data.

## Backpropagation and Gradient Descent
- **Backpropagation**: This is the method used to calculate the error and propagate it backward through the network to update the weights.
  - The error is calculated at the output layer.
  - The network adjusts the weights by sending the error signal back through the layers.
- **Gradient Descent**: The optimization technique used to minimize the error (loss function).
  - It adjusts weights to reach the optimal solution, typically by following the gradient of the loss function.
  - **Stochastic Gradient Descent (SGD)** is often used for faster and more efficient optimization.

## Multilayer Perceptron (MLP) and Feedforward Neural Networks (FFNN)
- **MLP**: A **Multilayer Perceptron** is a type of neural network with multiple layers of neurons.
  - It is a **feedforward neural network (FFNN)**, meaning the data flows in one direction: from input to output.
  - MLPs can model complex non-linear relationships by stacking multiple layers (hidden layers).

## Gradient Descent Variants
- **Steepest Descent**: Sometimes referred to as the "steepest descent," this refers to the basic idea of moving in the direction of the negative gradient to minimize the loss function.

# FoML - 08

## Neural Network Training
- The goal of training a neural network is to minimize the error between the predicted output \( y_i \) and the true output \( t_i \).
- **Error function**: 
 $$ [
  E = \frac{1}{2} \sum (t_i - y_i)^2
  ]$$
  This represents the sum of squared errors between the target and predicted values.

## Backpropagation and Gradient Descent
- **Backpropagation**: A method used to compute the gradient of the loss function with respect to each weight by propagating the error backward through the network.
  - The weights are updated by calculating the gradient error and adjusting them accordingly.
  - The number of hidden layers in the network is determined empirically:
    - Too few hidden layers: The network might not learn the complex relationships in the data.
    - Too many hidden layers: The network risks overfitting the data.

### Gradient Descent Optimization
- **Goal**: Minimize the error (loss function) as fast as possible using gradient descent.
- To update the weights in the direction that decreases error, we need to calculate the **derivative** of the loss function with respect to the weights.
- The activation function in neural networks must meet certain criteria:
  - **Continuous**, **differentiable**, **non-decreasing**, and **easy to compute**.

### Variants of Gradient Descent:
1. **Stochastic Gradient Descent (SGD)**: Updates weights after each data point (faster, noisier).
2. **Batch Gradient Descent**: Updates weights after calculating the gradient on the entire dataset (slower but more stable).
3. **Mini-Batch Gradient Descent**: A compromise between the two, where weights are updated after a small batch of data points.

---

## Activation Functions
Activation functions introduce non-linearity to the network, which allows it to learn complex patterns.

1. **Sigmoid**: Maps input to a value between 0 and 1.
$$   [
   \sigma(x) = \frac{1}{1 + e^{-x}}
   ]$$
2. **Tanh**: Maps input to a value between -1 and 1.
$$   [
   \tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
   ]$$
3. **ReLU (Rectified Linear Unit)**: Activation is 0 if the input is negative, otherwise it is the input.
$$   [
   f(x) = \max(0, x)
   ]$$
4. **Leaky ReLU**: Similar to ReLU, but allows a small slope for negative inputs (helps avoid dead neurons).
$$   [
   f(x) = \begin{cases} 
   x & \text{if } x > 0 \\
   \alpha x & \text{if } x \leq 0 
   \end{cases}
   ]$$

---

## Loss Functions
The loss function measures the difference between the true output and the predicted output.

1. **Euclidean Loss**: Also known as squared loss, used in regression problems.
  $$ [
   L = \frac{1}{2} \sum (t_i - y_i)^2
   ]$$
2. **Softmax Loss**: Also called **multinomial logistic regression loss**, commonly used in classification tasks. 
   - It transforms the output into a probability distribution and calculates the cross-entropy between the predicted probabilities and the true labels.

---

## Support Vector Machine (SVM)
- **SVM** is derived from **statistical learning theory** and is a powerful model for both classification and regression tasks.
- **History**: In the mid-1990s, SVMs outperformed neural networks in various benchmarks.
- **Linear Classifier**: SVM constructs a hyperplane that separates data points of different classes. The objective is to maximize the margin between the classes.
  - **Pros**: Effective for high-dimensional spaces, especially with clear margins.
  - **Cons**: Can struggle with noisy data or overlapping classes.

# FoML - 09

## Linear Classifier and Margin
- **Support Vector Machine (SVM)** is a type of **linear classifier** that separates classes by finding the optimal hyperplane with the **maximum margin**.
- **Margin**: The distance between the separating hyperplane and the nearest data points from each class (called **support vectors**).
  - The goal of SVM is to maximize this margin to ensure good generalization.

## Estimating Margin
- The margin is calculated based on the distance between the hyperplane and the support vectors.
  - For a given hyperplane \( w^T x + b = 0 \), the margin is estimated as:
    $$[
    \text{Margin} = \frac{2}{\|w\|}
    ]$$
  - This ensures that the classifier is as far away as possible from both classes, reducing the risk of misclassification.

## Lagrange Multipliers
- **Lagrange Multipliers** are used to convert the constrained optimization problem (maximizing the margin) into an unconstrained problem.
  - The original optimization problem involves constraints that all data points should be classified correctly.
  - The **Lagrangian** is introduced to handle these constraints by incorporating **Lagrange multipliers**.
  - The optimization problem becomes:
$$[
    L(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum \alpha_i [y_i(w^T x_i + b) - 1]
    ]$$
  - Where \( \alpha_i \) are the Lagrange multipliers.

## Convexity and Duality
- **Convexity**: The optimization problem in SVM is **convex**, meaning that any local minimum is a global minimum. This ensures that the optimization problem can be solved efficiently.
- **Duality**: SVM leverages **duality theory** to solve the problem in the **dual form**. Instead of solving the primal problem directly, we solve its dual, which often has fewer variables and constraints.
  - The dual form of the optimization problem is:
    $$[
    \max_{\alpha} \sum \alpha_i - \frac{1}{2} \sum \alpha_i \alpha_j y_i y_j (x_i^T x_j)
    ]$$
    Subject to: 
    $$[
    \sum \alpha_i y_i = 0 \quad \text{and} \quad \alpha_i \geq 0
    ]$$
    - Solving this dual form gives the optimal values of \( \alpha \), from which we can compute \( w \) and \( b \).

## SVM Standard Primal Form
- The **primal form** of the SVM optimization problem is:
$$  [
  \min_{w, b} \frac{1}{2} \|w\|^2
  ]$$
  Subject to:
$$  [
  y_i(w^T x_i + b) \geq 1 \quad \forall i
  ]$$
  This ensures that each point is correctly classified with a margin of at least 1.

## Hinge Loss
- SVM uses a **hinge loss function** for classification tasks.
  - The hinge loss is defined as:
  $$  [
    L(y_i, f(x_i)) = \max(0, 1 - y_i f(x_i))
    ]$$
    Where \( f(x_i) = w^T x_i + b \).
  - If the point is correctly classified and far from the margin, the loss is 0. If it is within the margin or misclassified, the loss increases.

## Regularization in SVM
- To prevent overfitting, **regularization** is added to the SVM objective function. This balances the trade-off between maximizing the margin and minimizing classification error.
  - The regularized primal form of SVM becomes:
    $$[
    \min_{w, b} \frac{1}{2} \|w\|^2 + C \sum \xi_i
    ]$$
    Where \( \xi_i \) are slack variables representing the amount by which a data point is on the wrong side of the margin, and \( C \) is the regularization parameter controlling the trade-off.

## SVM Dual with KKT (Karush-Kuhn-Tucker) Conditions
- The **dual formulation** of SVM leads to the application of the **KKT conditions**, which are necessary conditions for a solution to be optimal.
  - The KKT conditions ensure that:
    1. **Primal feasibility**: $$( y_i(w^T x_i + b) \geq 1 - \xi_i )$$
    2. **Dual feasibility**: $$( \alpha_i \geq 0 )$$
    3. **Complementary slackness**: $$( \alpha_i [y_i(w^T x_i + b) - 1 + \xi_i] = 0 )$$
  - These conditions help in solving the dual problem efficiently, ensuring that the support vectors are identified and the hyperplane is correctly placed.


# FoML - 10

## Solving SVM Duality with KKT Conditions

### KKT Conditions and SVM Duality
- The **Karush-Kuhn-Tucker (KKT) conditions** are necessary for a solution to be optimal in constrained optimization problems, such as in the dual formulation of **Support Vector Machines (SVM)**.
- The dual problem transforms the primal optimization into a convex optimization problem that is easier to solve, especially when using **Lagrange multipliers**.

#### KKT Conditions for SVM
The KKT conditions for SVM can be summarized as:
1. **Primal feasibility**: The data points should satisfy the margin constraints.
$$   [
   y_i(w^T x_i + b) \geq 1 - \xi_i
   ]$$
  $$ (\text{For separable data:} ( y_i(w^T x_i + b) \geq 1 ))$$
   
2. **Dual feasibility**: The Lagrange multipliers \( \alpha_i \) should be non-negative.
$$   [
   \alpha_i \geq 0
   ]$$
   
3. **Complementary slackness**: The product of the Lagrange multipliers and the margin violations must be zero.
$$   [
   \alpha_i [y_i(w^T x_i + b) - 1 + \xi_i] = 0
   ]$$

4. **Stationarity**: The gradient of the Lagrangian with respect to \( w \) and \( b \) must vanish, ensuring optimality.
$$   [
   \frac{\partial L(w, b, \alpha)}{\partial w} = 0, \quad \frac{\partial L(w, b, \alpha)}{\partial b} = 0
   ]$$

By satisfying these conditions, we can solve the dual problem and identify the support vectors that lie on the margin.

## Convex Optimization Problem in SVM

- **Convex optimization** problems are easier to solve because they guarantee a **global minimum**.
- The SVM optimization problem is convex, meaning the cost function (hinge loss with regularization) is convex.
- The goal in SVM is to minimize the objective function while maintaining the constraints on margin and classification.

#### Solving Convex Optimization with Duality
- The **dual form** of the SVM optimization problem:
$$  [
  \max_{\alpha} \sum \alpha_i - \frac{1}{2} \sum \alpha_i \alpha_j y_i y_j (x_i^T x_j)
  ]$$
  subject to:
$$  [
  \sum \alpha_i y_i = 0 \quad \text{and} \quad \alpha_i \geq 0
  ]$$
  allows us to solve the convex problem efficiently using numerical methods, such as **Quadratic Programming (QP)**.

## Non-Separable Data and Soft Margin SVM

### Non-Separable Data
- In real-world data, classes are often **non-separable**, meaning there is no clear hyperplane that can perfectly separate them.
- To handle non-separable data, **Soft Margin SVM** is used. This introduces a **slack variable** \( \xi_i \), allowing some points to be misclassified or lie within the margin.

### Soft Margin SVM (C-SVM)
- **Soft Margin SVM** allows for flexibility in the decision boundary by introducing slack variables \( \xi_i \) for each point that does not satisfy the margin constraint.
- The objective function for Soft Margin SVM is:
$$  [
  \min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_{i} \xi_i
  ]$$
  where \( C \) is a regularization parameter that controls the trade-off between maximizing the margin and minimizing classification errors.

#### Role of Slack Variables
- The slack variable \( \xi_i \) allows points to be inside the margin or misclassified, but at a cost. The **penalty** for violating the margin is controlled by \( C \).
- **Larger \( C \)**: Penalizes misclassifications heavily, leading to a smaller margin.
- **Smaller \( C \)**: Allows more misclassifications, leading to a wider margin and better generalization.

## SVM for Noisy Data: C-SVM and Soft Margin
- **Noisy data** often contains mislabeled or outlier data points that do not follow the general pattern of the dataset.
- **C-SVM (C-Support Vector Classification)** is a soft-margin SVM that handles noisy data by allowing some flexibility in the decision boundary.
- **Soft margin** allows for better generalization in the presence of noise by not forcing the classifier to perfectly fit the data.

## Harder 2-Dimensional Data

### Challenges in 2-Dimensional Data
- In certain datasets, especially in **2-dimensional spaces**, the data may not be linearly separable, or the margin may be hard to estimate because of noise or outliers.
- In such cases, SVM can still be effective by:
  - **Soft margins** to allow for misclassification.
  - **Kernel tricks** to map the data into a higher-dimensional space where it is separable.

#### Handling Harder Data with Kernel SVM
- For data that is difficult to separate in 2 dimensions, we can use the **Kernel Trick** to project the data into a higher-dimensional space where a linear separation is possible.
  - Common kernels include **polynomial kernels** and **Radial Basis Function (RBF)** kernels.

# FoML - 11

## Regularization in SVM to Avoid Overfitting
- **Overfitting** occurs when a model learns noise and patterns specific to the training data, which hurts generalization.
- In **SVM**, regularization is introduced to prevent overfitting by controlling the trade-off between maximizing the margin and minimizing classification errors.
- The regularized **primal objective** in SVM is:
  $$
  \min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_{i} \xi_i
  $$
  Where:
  - \( C \) is the **regularization parameter**. A smaller \( C \) allows a wider margin but with potential misclassifications, reducing overfitting.

## Complementary Slackness
- **Complementary slackness** is a condition derived from the **Karush-Kuhn-Tucker (KKT) conditions**, which are used to find optimal solutions for the dual form of SVM.
- In the context of SVM, the **complementary slackness condition** ensures that either:
  - The **Lagrange multiplier** \( \alpha_i \) is zero, meaning the data point is correctly classified with a sufficient margin.
  - Or, the point lies on the margin (or within the margin for soft-margin SVM), and the multiplier is positive.
- **Equation**:
  $$
  \alpha_i \cdot [y_i(w^T x_i + b) - 1 + \xi_i] = 0
  $$

---

## Kernel Tricks
- **Kernel tricks** allow SVM to handle **non-linearly separable data** by transforming the input features into higher-dimensional spaces where linear separation is possible.
- **Key Kernels**:
  1. **Higher-Order Polynomial Kernel**:
     - Projects data into a higher-dimensional space using polynomials.
     - Kernel function:
     $$
     K(x_i, x_j) = (x_i^T x_j + 1)^d
     $$
     Where \( d \) is the degree of the polynomial.
     
  2. **Radial Basis Function (RBF) Kernel**:
     - Maps data into an infinite-dimensional space, often used when the decision boundary is highly complex.
     - Kernel function:
     $$
     K(x_i, x_j) = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)
     $$
     Where \( \sigma \) is a parameter controlling the width of the Gaussian.

  3. **Linear Kernel**:
     - Simply computes the dot product in the original feature space, used when the data is already linearly separable.
     - Kernel function:
     $$
     K(x_i, x_j) = x_i^T x_j
     $$

---

## Gaussian Kernel Function
- A common choice for **non-linear SVM** is the **Gaussian Kernel** (a type of RBF kernel).
- The **Gaussian Kernel function** computes similarity based on the distance between two data points in feature space:
  $$
  K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)
  $$
  Where \( \gamma \) is a **hyperparameter** controlling the influence of single training examples. Higher \( \gamma \) means the influence is more localized around specific points.

---

## Decision Tree Pruning
- **Pruning** is a regularization technique used in **Decision Trees** to reduce overfitting by removing sections of the tree that provide little value for classifying instances.
- Two common pruning strategies:
  1. **Pre-Pruning**: Stop the tree's growth early, based on conditions such as maximum depth or minimum node size.
  2. **Post-Pruning**: Grow the full tree and then prune nodes based on performance on a validation set.

### Criteria for Pruning:
- **Error Reduction**: Nodes that do not improve classification accuracy significantly are pruned.
- **Cost Complexity Pruning**: This strategy prunes the tree based on a combination of the tree's size and its performance on the training data.

---

## Neural Network Regularization Techniques
To improve the generalization and avoid overfitting in **Neural Networks**, several regularization techniques are applied:

### Stochastic Gradient Descent (SGD)
- **SGD** is an optimization technique where the weights of the neural network are updated using a small batch of training data at a time.
- Helps in faster convergence and introduces a bit of noise in weight updates, which can act as a regularizer.

### Dropout
- **Dropout** is a technique where, during training, random neurons are dropped from the network at each iteration.
- This forces the network to not rely on specific neurons, helping it generalize better.
- **Dropout rate** is the proportion of neurons dropped at each iteration.

### Early Stopping
- **Early stopping** is a technique where training is stopped as soon as the performance on a validation set stops improving.
- This prevents overfitting by not allowing the model to train for too long and start memorizing the noise.

### Quantization
- **Quantization** involves reducing the precision of weights, which can prevent overfitting and also reduce the size of the model.
- Weights are approximated to fewer levels, introducing a form of regularization.

---

## Neural Network Loss Functions:
- **Loss functions** in neural networks measure the difference between the predicted output and the actual label.
- Commonly used loss functions include:
  - **Euclidean Loss** for regression tasks:
  $$
  E = \frac{1}{2} \sum_{i=1}^{n} (y_i - \hat{y_i})^2
  $$
  - **Softmax Loss** (Multinomial Logistic Loss) for classification tasks:
  $$
  L = - \sum_{i=1}^{n} y_i \log(\hat{y_i})
  $$

# FoML - 12

**Absent in today's lecture due to health issue,**
**But Quiz happened this Day & I lost 2% free marks. (easiest Quiz)**
**😢**

## Ensemble Methods Overview
- **Ensemble methods** are techniques that combine multiple learning models to improve overall performance.
- The key idea is that combining several weak models (weak learners) can result in a strong model.
- Ensemble methods generally reduce **variance** and **bias**, leading to better generalization on unseen data.

## Training Set and Bootstrap Sampling
- **Bootstrap Sampling** is a technique used to generate multiple training datasets by randomly sampling from the original dataset with replacement.
- Each sample (called a **bootstrap sample**) is used to train a different model in the ensemble.
  
### Bootstrap Process:
  1. From the training set of size \( N \), generate multiple bootstrap samples, each of size \( N \).
  2. These samples are created by sampling with replacement, meaning some data points may appear more than once in a single bootstrap sample.
  3. Models trained on each bootstrap sample may capture different aspects of the data, leading to better generalization when combined.

## Bagging (Bootstrap Aggregating)
- **Bagging** is an ensemble technique that combines multiple models trained on different bootstrap samples of the data.
- Each model is trained independently, and the final prediction is obtained by averaging the predictions (for regression) or by majority voting (for classification).

### Bagging Process:
  1. Create multiple bootstrap samples from the training set.
  2. Train a separate model (usually a decision tree) on each sample.
  3. Combine the models' outputs by **averaging** (regression) or **majority vote** (classification).

### Advantages of Bagging:
- Reduces **variance** by averaging out the noise in different models.
- Helps avoid **overfitting**, particularly in high-variance models like decision trees.

## Boosting
- **Boosting** is another ensemble technique that builds models sequentially, where each subsequent model tries to correct the errors made by the previous ones.
- In boosting, models are trained in a sequence, and the misclassified data points are given higher weights so that the next model focuses more on those errors.

### Boosting Process:
  1. Initialize the first model with equal weights for all data points.
  2. After each iteration, the weights of incorrectly classified points are increased so that the next model focuses more on those difficult cases.
  3. Final predictions are made by combining the predictions of all models, usually by weighted majority voting or a weighted sum for regression.

### Key Points in Boosting:
- Boosting reduces **bias** by correcting errors made in previous models.
- **Each model is dependent** on the previous one, unlike in bagging, where models are independent.

## Adaboost (Adaptive Boosting)
- **Adaboost** is one of the most popular boosting algorithms. It adjusts the weights of incorrectly classified points, forcing the next weak learner to focus more on them.
  
### Adaboost Algorithm:
1. **Initialize weights** for all data points: Initially, all points are given equal weights.
   $$
   w_i = \frac{1}{N}, \text{where } N \text{ is the number of data points.}
   $$
2. **Train weak learner**: Train a weak model (e.g., a decision stump or shallow tree) on the weighted dataset.
3. **Calculate error**: Calculate the error of the model and adjust the weights of misclassified points. The error \( \epsilon \) is calculated as:
   $$
   \epsilon = \frac{\sum w_i \cdot \mathbb{I}(y_i \neq h(x_i))}{\sum w_i}
   $$
   where \( h(x_i) \) is the predicted label and \( y_i \) is the actual label.
4. **Update weights**: Increase the weights of misclassified points:
   $$
   w_i^{\text{new}} = w_i^{\text{old}} \cdot \exp(\alpha \cdot \mathbb{I}(y_i \neq h(x_i)))
   $$
   where \( \alpha \) is the weight given to the weak learner's contribution and is calculated as:
   $$
   \alpha = \frac{1}{2} \ln\left(\frac{1 - \epsilon}{\epsilon}\right)
   $$
5. **Repeat**: The process is repeated for several iterations, and each model focuses more on the misclassified points.
6. **Final Prediction**: The final prediction is a weighted sum of all weak learners:
   $$
   H(x) = \text{sign}\left(\sum \alpha_i h_i(x)\right)
   $$

### Example of Adaboost:
- Imagine a dataset where some points are hard to classify correctly.
- The first weak learner may classify 70% correctly, but the 30% misclassified points will have their weights increased.
- The next learner will focus more on these harder points, and the process continues until the majority of points are correctly classified.

# FoML - 13

## AdaBoost (Adaptive Boosting) - In Depth
- **AdaBoost** is a boosting algorithm where each subsequent model in the ensemble focuses more on the mistakes made by the previous ones.
- It combines weak learners (e.g., decision stumps) to form a strong classifier.
  
### Key Characteristics:
1. **Weight Updates**:
   - Initially, all data points are given equal weights.
   - After each iteration, weights of misclassified points are increased, forcing the next weak learner to focus on these difficult examples.
   - The updated weights influence how much the next weak learner focuses on each point.
   
2. **Error Calculation**:
   - Each learner’s error \( \epsilon \) is calculated based on the weights of misclassified points:
     $$
     \epsilon = \frac{\sum w_i \cdot \mathbb{I}(y_i \neq h(x_i))}{\sum w_i}
     $$
     
3. **Classifier Contribution**:
   - A classifier's influence in the final prediction is determined by:
     $$
     \alpha = \frac{1}{2} \ln\left(\frac{1 - \epsilon}{\epsilon}\right)
     $$
   - The more accurate a learner, the higher its influence on the final prediction.
   
4. **Final Prediction**:
   - The final model is a weighted combination of all weak learners:
     $$
     H(x) = \text{sign}\left(\sum \alpha_i h_i(x)\right)
     $$

## Gradient Boosting
- **Gradient Boosting** is a more general boosting framework where models are added sequentially to minimize a **loss function** (e.g., mean squared error, log loss).
- Each new model corrects the residuals (errors) of the previous model by learning the gradient of the loss function with respect to the prediction.

### Gradient Boosting Process:
1. **Initialize** the model with a constant prediction (e.g., the mean for regression).
2. **Calculate Residuals**: At each iteration, calculate the residuals (errors) of the current model:
   $$
   r_i = y_i - \hat{y_i}
   $$
   where \( y_i \) is the true label and \( \hat{y_i} \) is the predicted label.
   
3. **Fit a New Model**: Train a new model to predict these residuals.
4. **Update the Prediction**: Update the predictions by adding the new model’s output to the current prediction, weighted by a step size \( \eta \):
   $$
   \hat{y_i}^{\text{new}} = \hat{y_i}^{\text{old}} + \eta \cdot f(x_i)
   $$
   
5. **Repeat** the process for a fixed number of iterations or until the residuals are minimized.

### Key Points in Gradient Boosting:
- **Step Size** \( \eta \): Controls the contribution of each new model.
- **Regularization**: Gradient boosting can be regularized by controlling the number of models or the complexity of each model (e.g., decision tree depth).

## XGBoost (Extreme Gradient Boosting)
- **XGBoost** is an efficient and scalable implementation of gradient boosting that includes several optimizations for better performance.
- It is widely used in machine learning competitions and applications due to its speed and accuracy.

### Key Innovations in XGBoost:
1. **Regularization**:
   - XGBoost introduces both **L1** (Lasso) and **L2** (Ridge) regularization terms in the objective function to prevent overfitting:
     $$
     \text{Objective} = \sum \text{loss}(y_i, \hat{y_i}) + \lambda \|w\|^2 + \alpha \|w\|
     $$
     where \( \lambda \) controls L2 regularization and \( \alpha \) controls L1 regularization.

2. **Second-Order Approximation**:
   - Instead of just using the gradient (first-order derivative), XGBoost also uses the **Hessian** (second-order derivative) to make more precise updates.

3. **Sparsity Awareness**:
   - XGBoost is optimized to handle **sparse data** (e.g., missing values) efficiently, skipping unnecessary computations.

4. **Parallelization**:
   - The training process in XGBoost is parallelized, making it much faster than traditional gradient boosting implementations.

5. **Tree Pruning**:
   - XGBoost prunes trees by performing **depth-first search** to optimize leaf nodes, removing nodes that do not contribute significantly to the model’s performance.

6. **Handling Imbalanced Data**:
   - XGBoost includes parameters like `scale_pos_weight` to adjust for class imbalance.

### XGBoost Process:
1. **Initialize**: Start with a base model (often a constant value).
2. **Compute Gradient and Hessian**: For each data point, compute the gradient and Hessian (second derivative) of the loss function with respect to the prediction.
3. **Fit Trees**: Build decision trees that fit the residuals, using the gradients and Hessians to guide the splits.
4. **Update Predictions**: Update the model’s predictions using the trees' output, similarly to standard gradient boosting, but with additional regularization and optimization.

# FoML - 14

## Overview of Ensemble Methods

Ensemble methods combine multiple models to improve predictive performance. The main idea is that by aggregating the predictions from several models, we can achieve better accuracy than any single model.

### Types of Ensemble Methods

1. **Bagging (Bootstrap Aggregating)**
   - Combines predictions from multiple models trained on different subsets of the data.
   - Reduces variance and helps prevent overfitting.
   - Example: Random Forest, where multiple decision trees are trained on bootstrapped samples.

2. **Boosting**
   - Sequentially trains models, where each new model focuses on correcting the errors made by the previous ones.
   - Reduces bias and can lead to higher accuracy.
   - Example: Gradient Boosting, AdaBoost.

## Gradient Boosting

### Overview

Gradient Boosting is a powerful ensemble technique that builds models sequentially. It optimizes the loss function by adding new models that predict the residuals (errors) of the existing model.

### Key Concepts

- **Base Learner**: Typically, a weak learner (e.g., decision tree) that performs slightly better than random guessing.
- **Loss Function**: A measure of how well the model is performing. Commonly used loss functions include Mean Squared Error (MSE) for regression and log loss for classification.
- **Learning Rate**: A hyperparameter that controls the contribution of each new model to the ensemble. Lower values lead to more robust models but require more iterations.

### Algorithm Steps

1. **Initialize the model** with a constant prediction (e.g., mean of the target variable).
2. **For each iteration**:
   - Calculate the residuals (errors) from the current model.
   - Fit a new model to these residuals.
   - Update the predictions by adding the predictions of the new model, scaled by the learning rate.
3. **Repeat** until a specified number of models is reached or improvements plateau.

### Advantages of Gradient Boosting

- **Flexibility**: Can optimize various loss functions and work with different types of data.
- **Accuracy**: Often produces state-of-the-art predictive performance on structured/tabular data.
- **Feature Importance**: Provides insights into feature contributions through variable importance scores.

### Disadvantages of Gradient Boosting

- **Overfitting**: Risk of overfitting if not properly regularized.
- **Computationally Intensive**: Training can be slow and require significant computational resources.
- **Sensitivity to Noisy Data**: Performance can degrade if the dataset contains a lot of noise or irrelevant features.

### Popular Implementations

- **XGBoost**: An efficient and scalable implementation of gradient boosting, known for its speed and performance.
- **LightGBM**: Optimized for large datasets, it uses histogram-based learning to reduce memory usage and increase training speed.
- **CatBoost**: Designed to handle categorical features automatically, making it user-friendly and effective for various tasks.

# FoML - 15

## Data Handling Issues

Data handling issues can significantly affect the performance of machine learning models. Common issues include:

### 1. Missing Data
- **Description**: Occurs when some data points are not recorded.
- **Handling Strategies**:
  - **Imputation**: Fill missing values using statistical methods (mean, median, mode).
  - **Removal**: Discard rows or columns with missing data, but this may lead to loss of information.
  - **Prediction Models**: Use models to predict missing values based on available data.

### 2. Outliers
- **Description**: Data points that differ significantly from other observations.
- **Handling Strategies**:
  - **Removal**: Eliminate outliers if justified.
  - **Transformation**: Apply transformations (e.g., log transformation) to reduce the impact of outliers.
  - **Robust Models**: Use models that are less sensitive to outliers (e.g., decision trees).

### 3. Data Imbalance
- **Description**: Unequal representation of classes in the dataset.
- **Impact on Classifiers**:
  - Classifiers may become biased towards the majority class, leading to poor performance on the minority class.
  - Metrics like accuracy can be misleading in imbalanced datasets; precision, recall, and F1-score are better indicators.

- **Handling Strategies**:
  - **Resampling**: Techniques like oversampling the minority class or undersampling the majority class.
  - **SMOTE (Synthetic Minority Over-sampling Technique)**:
    - Generates synthetic samples for the minority class by interpolating between existing minority instances.
    - Helps balance the dataset without simply duplicating existing examples.

### 4. Feature Scaling
- **Description**: Variability in feature ranges can affect model performance.
- **Handling Strategies**:
  - **Normalization**: Scale features to a range (e.g., 0 to 1).
  - **Standardization**: Center features around zero and scale to unit variance (Z-score normalization).

---

## Bias-Variance Tradeoff

The bias-variance tradeoff is a fundamental concept in machine learning that affects model performance.

![Bias Variance](Bias_Variance.png)
### 1. Generalization
- **Definition**: The ability of a model to perform well on unseen data, beyond the training set.
- **Generalization Error**: The difference between the expected prediction of the model on the entire population and the actual prediction error on unseen data.

### 2. Bias
- **Definition**: The error due to overly simplistic assumptions in the learning algorithm.
- **Characteristics**:
  - High bias can lead to **underfitting**, where the model cannot capture the underlying patterns in the data.
  - Example: A linear model applied to a nonlinear dataset.

### 3. Variance
- **Definition**: The error due to excessive complexity in the learning algorithm.
- **Characteristics**:
  - High variance can lead to **overfitting**, where the model learns noise and random fluctuations in the training data.
  - Example: A complex model capturing all data points in the training set.

### 4. Tradeoff
- Striking the right balance between bias and variance is crucial for optimal model performance.
- **Goal**: Minimize total error (Bias² + Variance + Irreducible Error) while maintaining a good fit to the training data and generalizing well to unseen data.

# FoML - 16
## Regularization

Regularizers are techniques used to prevent overfitting in machine learning models by adding a penalty term to the loss function. This encourages simpler models that generalize better to unseen data.

## 1. Overview of Regularization

- **Purpose**: Regularization discourages the learning of overly complex models by imposing a constraint on the model parameters.
- **Effect**: Helps to improve generalization by reducing the risk of overfitting to the training data.

## 2. Common Types of Regularization

### A. L1 Regularization (Lasso)

- **Description**: Adds the absolute value of the coefficients as a penalty term to the loss function.
- **Mathematical Representation**:

  Given a loss function L(y, ŷ), where y is the true value and ŷ is the predicted value, L1 regularization modifies the loss function as follows:
  
  Loss_Lasso = L(y, ŷ) + λ ∑ |wi|

  where:
  - λ is the regularization parameter that controls the strength of the penalty.
  - wi are the model weights.

- **Effects**:
  - Encourages sparsity in the model parameters; some weights may be driven to zero, effectively selecting a simpler model.
  - Useful for feature selection as it can eliminate irrelevant features.

### B. L2 Regularization (Ridge)

- **Description**: Adds the squared value of the coefficients as a penalty term to the loss function.
- **Mathematical Representation**:

  The loss function is modified as follows:
  
  Loss_Ridge = L(y, ŷ) + λ ∑ wi^2

- **Effects**:
  - Shrinks the coefficients towards zero but does not eliminate them completely, leading to models that utilize all features.
  - Helps prevent multicollinearity (when predictor variables are highly correlated) by stabilizing estimates.

### C. Elastic Net Regularization

- **Description**: Combines L1 and L2 regularization.
- **Mathematical Representation**:

  The modified loss function is given by:
  
  Loss_ElasticNet = L(y, ŷ) + λ1 ∑ |wi| + λ2 ∑ wi^2

- **Effects**:
  - Provides a balance between L1 and L2, promoting both sparsity and small coefficients.
  - Particularly useful when dealing with datasets with highly correlated features.

## 3. Lp Norm Regularizers

### A. Overview of Lp Norms

The Lp norm is a generalization of the L1 and L2 norms and is defined for a vector w as:

L^p(w) = (∑ |wi|^p)^(1/p)

where:
- p is a positive integer (e.g., 1 for L1, 2 for L2).
- wi are the individual weights or coefficients.

### B. Common Lp Norms

1. **L1 Norm (p=1)**:
   - Defined as: L^1(w) = ∑ |wi|
   - Encourages sparsity by pushing some weights to zero.

2. **L2 Norm (p=2)**:
   - Defined as: L^2(w) = (∑ wi^2)^(1/2)
   - Shrinks weights evenly but retains all features.

3. **Lp Norm (General Case)**:
   - For 0 < p < 1, the Lp norm encourages sparsity and can be used in situations similar to L1 regularization but is not commonly used due to difficulties in optimization.
   - Higher values of p (like p=3) can create a balance between L1 and L2 properties.

### C. How Lp Norms Work

- **Influence on Model Complexity**:
  - The choice of p affects how the penalty is applied to the weights:
    - Lower p values (e.g., L1) can lead to more sparsity, while higher p values (e.g., L2) lead to smoother weight distributions.
  - As p increases, the regularization effect becomes stronger, and the model is less likely to overfit.

- **Gradient Descent and Lp Regularization**:
  - The gradients for different Lp norms affect weight updates during optimization:
    - For L1, the gradient is a subgradient, which is not defined at zero.
    - For L2, the gradient is smooth and continuous.
  
  The gradient of the loss function with Lp regularization changes based on the chosen norm, affecting how weights are updated during training.

## 4. How Regularizers Work Mathematically

### A. The Concept of Loss Function

The loss function quantifies the difference between the predicted values and the actual values. By adding a regularization term, the loss function is transformed into a form that accounts for model complexity.

- **General Form**:
  
  Total Loss = Prediction Error + Regularization Penalty

### B. Impact of the Regularization Parameter λ

- **Role of λ**:
  - A small value of λ results in a model that may fit the training data very closely (potential overfitting).
  - A large value of λ increases the penalty on complexity, potentially leading to underfitting.

### C. Model Complexity Control

- Regularization effectively controls the tradeoff between bias and variance:
  - **Higher Bias**: Strong regularization can lead to increased bias but reduced variance.
  - **Lower Bias**: Weak regularization may fit the training data well (low bias) but can lead to high variance.

## 5. Regularization in Different Models

#### K-Nearest Neighbors (K-NN)
- **Strategy**: Choose a higher \( k \) to reduce the model's sensitivity to noise, which acts as a form of regularization.

#### Decision Trees
- **Strategy**: Use **pruning** to remove nodes that provide little power to the model, preventing overfitting.

#### Naïve Bayes
- **Description**: As a parametric model, it automatically acts as a regularizer by assuming independence between features, which limits complexity.

#### Support Vector Machines (SVMs)
- **Strategy**: Incorporates a regularization parameter \( C \) that balances the tradeoff between maximizing the margin and minimizing classification errors.

#### Neural Networks
- **Strategies**:
  - Use dropout layers to randomly deactivate a fraction of neurons during training, which helps prevent overfitting.
  - Implement weight regularization (L1 or L2) to constrain weight values.

# FoML - 17

## Bayes Error

Bayes error refers to the **minimum error rate** that any classifier can achieve for a given learning problem. It represents the error of the **Bayes optimal classifier**, which is the best possible classifier under the assumption that the true data distribution is known. 

The key point is that **Bayes error** requires knowledge of the **true underlying distribution** of the data. If we have access to this distribution, we can compute the optimal decision boundary and minimize classification errors. In practice, this is challenging because we often do not know the true distribution. However, if the distribution were known, Bayes' Theorem would allow us to calculate the most probable class for a given input, minimizing the error.

### Formalizing Learning

To formalize the concept of learning, we often use the **PAC (Probably Approximately Correct)** framework, which defines a learning problem in terms of the relationship between a hypothesis and the true distribution.

### PAC Learning

**PAC Learning** is a framework in machine learning that aims to formalize the process of learning from examples. It guarantees that, with high probability, a learned hypothesis will be approximately correct, given a sufficiently large sample of data. 

- **Probability Approximately Correct**: A learning algorithm is said to be **PAC-learnable** if, for any distribution of data, the algorithm can learn a hypothesis that is close to the optimal hypothesis with high probability. In other words, for a given error rate [ε] and confidence level [1 - δ], the algorithm will output a hypothesis that deviates from the optimal hypothesis by no more than [ε] with probability at least [1 - δ].

#### PAC Definition:
- A hypothesis class [H] is PAC-learnable if there exists an algorithm that, given a sample of size [m], produces a hypothesis such that the probability of error is less than or equal to [ε], with high probability.

### Statistical Learning and Generalization of Error

**Statistical learning theory** is the theoretical foundation of machine learning, which seeks to understand the performance of algorithms in terms of generalization from a sample to the entire population.

- **Generalization** refers to the ability of a model to perform well on unseen data (i.e., not just the data it was trained on).
- **Error Generalization** concerns how the error on the training set compares to the error on the test set. A model that generalizes well is one where the error on the test set is close to the error on the training set.

### Model Class [F] and Hypothesis Space

In machine learning, the **model class [F]** represents the set of all possible hypotheses that a learning algorithm can choose from. This is also known as the **hypothesis space**.

- If we have a large hypothesis space, the learning algorithm may have more options to choose from, but it also increases the complexity of finding the best model.
- **PAC Learnability of [F]**: A hypothesis space [F] is said to be **PAC-learnable** if there exists an algorithm that can produce a hypothesis with a high probability that is approximately correct, given enough training data.

### Empirical Risk Minimization

Empirical Risk Minimization (ERM) is a strategy for learning where we minimize the **empirical risk** (or error) based on the training data. The **empirical risk** is the average loss over the sample.

- **Empirical Risk**: For a hypothesis [h], the empirical risk is given by:

$$  [ R_emp(h) = (1/m) * Σ L(h(x_i), y_i) ]$$

  where [m] is the number of training samples, [L] is the loss function, and [(x_i, y_i)] are the data points and their corresponding labels.

The idea behind **ERM** is to choose the hypothesis that minimizes the average loss over the training data, which we hope will also minimize the generalization error.

### Proof Sketch and VC Theory Concepts

**VC (Vapnik-Chervonenkis) Theory** provides a framework for understanding the capacity of a model class and its ability to generalize. It is closely related to the concept of **shattering**.

- A **VC dimension** is the size of the largest set of points that can be shattered by a hypothesis class. A set of points is said to be **shattered** by a hypothesis class if, for every possible labeling of the points, there exists a hypothesis in the class that perfectly classifies the points.

The **VC dimension** provides insight into the complexity of the hypothesis class and is a key factor in understanding the trade-off between model complexity and generalization.

### No Free Lunch Theorem

The **No Free Lunch (NFL) theorem** states that **no single learning algorithm works best for all problems**. The theorem suggests that, for any algorithm that performs well on one class of problems, there is an equal and opposite algorithm that performs poorly on another class. 

- **NFL Theorem for Learning**: For any algorithm [A], any sample size [m], and any error [ε > 0], there exists a distribution [D] such that, when the algorithm is applied to data drawn from [D], the performance of the algorithm is no better than random guessing. This underscores the importance of selecting appropriate algorithms and datasets based on the task at hand.

### Conclusion

In summary, the PAC learning framework provides a formal understanding of the ability of an algorithm to learn from data. The concepts of **empirical risk minimization**, **VC dimension**, and **the No Free Lunch theorem** offer theoretical insights into how to design and evaluate learning algorithms. These theories are crucial for understanding how models generalize to new, unseen data, which is at the core of machine learning.

# FoML - 18

## Regression Formulation

Regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. The goal of regression is to predict the value of the dependent variable based on the input features (independent variables).

In regression problems, we aim to minimize the error between the predicted values and the true values (ground truth). This is typically done using a **loss function** that measures the difference between the predicted and actual values.

### Parametric vs Non-Parametric Regression

- **Parametric Regression**:
  In parametric regression, the form of the model is predefined, and the parameters of the model are learned from the data. A common example of parametric regression is **Linear Regression**. 
  The model assumes that there is a linear relationship between the independent and dependent variables. Once the model is defined, it estimates the parameters (e.g., slope and intercept) from the data.

  The key advantage of parametric regression is that it is computationally efficient, as it only requires the estimation of a fixed number of parameters.

- **Non-Parametric Regression**:
  In non-parametric regression, there is no assumption about the functional form of the model. Instead, the model adapts to the data by using a flexible structure, and the complexity of the model increases as more data points are added. Non-parametric regression methods do not rely on predefined parameters and can capture more complex relationships in the data.

  A common non-parametric regression method is **Kernel Regression**, which estimates the value of the dependent variable at a point based on the values of nearby points.

### Linear Regression

**Linear regression** is a simple and widely used parametric method for regression problems. The model assumes a linear relationship between the dependent variable [y] and independent variables [x]:

- **Model**: 
$$  [ y = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n + ε ]$$

  Where:
  - [y] is the dependent variable (target).
  - [x_1, x_2, ..., x_n] are the independent variables (features).
  - [w_0, w_1, ..., w_n] are the parameters (weights) to be learned.
  - [ε] is the error term (random noise).

In linear regression, we aim to find the parameters [w_0, w_1, ..., w_n] that minimize the error between the predicted values and the actual values. This is usually done by minimizing the **mean squared error (MSE)** between the predictions and the true values.

---

# FoML - 19

## Regularized Least Squares Regression

In **regularized regression**, we modify the cost function to include a penalty term that discourages large model coefficients. Regularization helps prevent overfitting, which can occur when the model becomes too complex and fits the noise in the training data.

The two most common types of regularization are **L1 regularization** (Lasso) and **L2 regularization** (Ridge).

### L2 Regularization (Ridge Regression)

Ridge regression is a regularized form of linear regression that adds a penalty to the cost function based on the squared values of the coefficients. This encourages the model to find smaller coefficients, which helps reduce overfitting.

- **Cost function** for Ridge regression:

$$  [ J(w) = (1/2m) * Σ (y_i - (w_0 + w_1x_1i + w_2x_2i + ... + w_nx_ni))^2 + λ Σ w_j^2 ]$$

  Where:
  - [m] is the number of data points.
  - [y_i] is the actual value of the target for the i-th data point.
  - [w_j] is the coefficient (weight) of the j-th feature.
  - [λ] is the regularization parameter that controls the strength of the penalty.

In Ridge regression, we add the squared sum of the coefficients [Σ w_j^2] to the cost function. The regularization parameter [λ] controls the trade-off between fitting the data and minimizing the size of the coefficients. As [λ] increases, the coefficients are driven towards zero, and the model becomes simpler.

### L1 Regularization (Lasso Regression)

Lasso (Least Absolute Shrinkage and Selection Operator) regression is another form of regularized regression, but it uses **L1 regularization**. The Lasso method adds the absolute values of the coefficients to the cost function, encouraging sparsity in the model.

- **Cost function** for Lasso regression:

$$  [ J(w) = (1/2m) * Σ (y_i - (w_0 + w_1x_1i + w_2x_2i + ... + w_nx_ni))^2 + λ Σ |w_j| ]$$

  Where:
  - [λ] is the regularization parameter.
  - The sum of the absolute values of the coefficients encourages some coefficients to be exactly zero, leading to feature selection.

Lasso regression is particularly useful when we believe that only a subset of features are relevant, as it can effectively eliminate irrelevant features by setting their coefficients to zero.

### Regularization and Overfitting

The goal of regularized regression is to prevent overfitting by penalizing large coefficients, which helps the model generalize better to new, unseen data. Regularization strikes a balance between bias and variance:
- **High regularization** (large [λ]) leads to higher bias but lower variance (underfitting).
- **Low regularization** (small [λ]) leads to lower bias but higher variance (overfitting).

By tuning the regularization parameter [λ], we can find the optimal balance and improve the performance of the regression model.

# FoML - 20

## Support Vector Machines (SVM) and Support Vector Regression (SVR)

Support Vector Machines (SVM) are a powerful class of algorithms used for classification and regression tasks. SVMs aim to find a hyperplane that best separates the data points in the feature space. For regression tasks, **Support Vector Regression (SVR)** is employed, which adapts the SVM approach for continuous output predictions rather than discrete classifications.

### Support Vector Regression (SVR)

SVR works by fitting a regression line (or hyperplane in higher dimensions) while allowing some margin of error. It attempts to minimize the error by focusing on the data points that are closest to the regression line, known as **support vectors**.

- **Complexity**:
  The objective of SVR is to minimize the following equation:

$$  [ 1/2 * ||w||^2 + C * Σ ε_i ]$$

  Where:
  - [||w||^2] is the complexity of the model (the weight vector).
  - [ε_i] is the deviation from the margin for each support vector.
  - [C] is the regularization parameter controlling the trade-off between model complexity and fitting errors.

In SVR, we focus on minimizing [1/2 * ||w||^2], where [w] is the weight vector of the model. The goal is to maximize the margin, i.e., the distance between the regression line (or hyperplane) and the support vectors, while keeping the model simple and avoiding overfitting.

### Tubes in SVR

The concept of a **tube** is used in SVR, where a regression line is constructed with a certain width. The width of the tube (also known as the margin) plays a crucial role in controlling the balance between underfitting and overfitting:

- **Wider Tube**: 
  - A wider tube reduces errors but allows the regression line to have more flexibility, meaning it can vary more. This reduces the risk of overfitting but can lead to **underfitting** as the model may fail to capture the nuances of the data. In this case, the slope of the regression line may be less defined due to fewer constraints from support vectors.

- **Narrower Tube**: 
  - A narrower tube leads to a regression line that fits the data points closely, which increases the risk of **overfitting**. The model becomes more sensitive to small fluctuations in the data, resulting in a more defined slope but potentially fitting noise rather than the underlying trend. This can make the model less generalizable.

### Non-Linear (Kernel) SVR

In cases where the data is not linearly separable, we use **kernel tricks** to map the data into a higher-dimensional space where a hyperplane can be used to separate the data. The kernel function allows us to work with non-linear relationships by applying a transformation that makes the problem linearly separable in a higher-dimensional space.

For example, in a **3D space**, the formula for a support vector regression hyperplane is:

- **Formula in 3D**:

$$  [ f(x) = w_1 * x_1 + w_2 * x_2 + w_3 * x_3 + b ]$$

Where:
- [w_1, w_2, w_3] are the weights for the three dimensions.
- [x_1, x_2, x_3] are the input features.
- [ b ] is the bias term.

### Logistic Regression

Logistic Regression is a classification algorithm used for binary classification problems. Unlike linear regression, which predicts a continuous output, logistic regression outputs probabilities that a given input belongs to one of two classes.

The key feature of logistic regression is the **sigmoid function**:

- **Sigmoid Function**:

$$  [ σ(x) = 1 / (1 + exp(-x)) ]$$

The output of the sigmoid function is always between 0 and 1, which is interpreted as the probability of the input belonging to the positive class.

### Logistic Regression Formula

The logistic regression model predicts the probability [p] of the positive class given the input features [x]:

- **Formula**:

$$  [ p = σ(w_0 + w_1 * x_1 + w_2 * x_2 + ... + w_n * x_n) ]$$

Where:
- [w_0, w_1, ..., w_n] are the weights to be learned by the model.
- [x_1, x_2, ..., x_n] are the input features.
- [ σ ] is the sigmoid function.

The goal is to find the weights [w_0, w_1, ..., w_n] that minimize the **log-loss** or **cross-entropy loss** between the predicted probabilities and the true labels.

### Regularization in Logistic Regression

To prevent overfitting, **regularization** is applied to logistic regression. This typically involves adding a penalty term to the loss function to constrain the magnitude of the weights. The two common types of regularization are **L1 regularization** (Lasso) and **L2 regularization** (Ridge), similar to regression.

The regularized cost function for logistic regression with L2 regularization is:

- **Cost Function with L2 Regularization**:

$$  [ J(w) = - (1/m) * Σ [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)] + λ Σ w_j^2 ]$$

Where:
- [p_i] is the predicted probability for the i-th data point.
- [y_i] is the true label (0 or 1).
- [ λ ] is the regularization parameter.
- [w_j] is the weight of the j-th feature.

### Gradient Descent for Logistic Regression

Gradient descent is commonly used to optimize the weights in logistic regression. During training, we iteratively adjust the weights to minimize the cost function.

- **Gradient Descent Update Rule**:

  [ w_j := w_j - α * (∂J(w)/∂w_j) ]

Where:
- [ α ] is the learning rate.
- [∂J(w)/∂w_j] is the partial derivative of the cost function with respect to the weight [w_j].

By applying gradient descent, the algorithm converges to the optimal set of weights that minimize the loss and regularization term.

# FoML - 21

## Bias-Variance Tradeoff

In machine learning, the **bias-variance tradeoff** describes the relationship between the bias and variance of a model. It is challenging to achieve both low bias and low variance simultaneously because they typically trade off against each other:

- **Bias** refers to the error introduced by approximating a real-world problem with a simplified model. High bias means that the model is too simple, leading to systematic errors or underfitting.
- **Variance** refers to the model's sensitivity to fluctuations in the training data. High variance means the model is too complex and overfits the training data, resulting in large fluctuations in predictions for different datasets.

### Tradeoff Explanation:
- **Reducing Bias**: To reduce bias, we often use more complex models, which can lead to higher variance.
- **Reducing Variance**: To reduce variance, we typically use simpler models, which can increase bias.
  
The goal is to strike a balance between bias and variance to achieve an optimal model performance.

## Clustering

Clustering is an unsupervised learning technique used to group similar data points together. This technique is widely used in applications such as:

- **Google Photos**: To group similar images and identify users uniquely.
- **Customer Segmentation**: To identify customers based on shared interests or behaviors.

### K-Means Clustering

**K-means clustering** is a popular method where the number of clusters (k) must be predefined. It assigns each data point to the cluster with the nearest centroid and iteratively refines these assignments. 

**Objective of K-means**: The objective is to minimize the **within-cluster variance** (i.e., the sum of squared distances from each point to the centroid of its cluster):

- **Monothetic vs. Polythetic Patterns**: 
  - **Monothetic patterns** are those where all features must satisfy the same condition across all members of the cluster.
  - **Polythetic patterns** allow for different features to define clusters, not necessarily requiring every feature to hold true.

### Types of Clusters
- **Hard Clustering**: Each point belongs strictly to one cluster.
- **Soft Clustering**: Each point can belong to multiple clusters with different probabilities.

### K-Means Methodology

1. **Initialization**: K-means begins by selecting initial centroids, which can have a significant impact on the final results. If the initial centroids are poorly chosen, the algorithm may converge to suboptimal clusters.
   
2. **Centroid Selection Probability**: The probability of selecting a good initial centroid is determined by:

   [ P(good centroid) = (Number of ways to select points as centroids from each cluster) / (Total number of possible outcomes) ]

3. **Cluster Assignment**: Each point is assigned to the cluster whose centroid is nearest, based on the chosen distance metric.

4. **Convergence**: The algorithm repeats the assignment and centroid update steps until the centroids do not change significantly.

### Sensitivity to Outliers

K-means is sensitive to outliers, as outliers can disproportionately influence the placement of the centroids. In real-world applications, such as in **termite removal** from data, it’s often useful to **remove outliers** before running the algorithm to improve clustering results.

### Distance Metrics in K-Means

Different distance metrics (e.g., Euclidean, Manhattan, Cosine) can significantly affect the results of K-means clustering. Choosing the right metric is crucial for the effectiveness of clustering.

## Hierarchical Clustering

**Hierarchical clustering** is an alternative to K-means where the number of clusters does not need to be predefined. Instead, it generates a **dendrogram**, which is a tree-like structure that shows the merging of clusters at each level.

- **Agglomerative (bottom-up)**: Start with each data point as its own cluster, and iteratively merge the closest clusters.
- **Divisive (top-down)**: Start with all points in a single cluster and iteratively split the clusters into smaller ones.

Hierarchical clustering has the advantage of not requiring the number of clusters (K) to be specified upfront.

### K-Median vs. K-Means

In **K-median** clustering, the centroid is replaced with the **median** rather than the mean. This is useful in cases where the data is not normally distributed and outliers are present, as the median is less sensitive to outliers than the mean.

## DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

Unlike K-means, **DBSCAN** does not require the user to specify the number of clusters beforehand. It works by identifying regions of high density and separating them from regions of low density, making it well-suited for datasets with irregular shapes and varying cluster sizes. DBSCAN also handles noise (outliers) by labeling them as points that do not belong to any cluster.

# FoML - 22

## Dimensionality Reduction

**Dimensionality reduction** is a technique used to reduce the number of features (dimensions) in a dataset while preserving as much of the relevant information as possible. The goal is to simplify the data, improve performance, and avoid overfitting. 

### Why Use Dimensionality Reduction?

- **Reduce Overhead**: Reducing the number of features can decrease computational cost and increase processing speed.
- **Improve Performance**: By focusing on the most important features, dimensionality reduction can improve the performance of machine learning models.
- **Avoid Overfitting**: High-dimensional data often leads to overfitting, where the model learns noise rather than the underlying pattern. Reducing dimensions can help mitigate this problem.
- **Easy Visualization**: Lower-dimensional data can be visualized more easily, making it simpler to interpret.
- **Noise Reduction**: Dimensionality reduction can help reduce noise by removing less relevant features that may introduce unnecessary variance.

### Curse of Dimensionality

As the number of features in a dataset increases, the data becomes more sparse, and the distances between data points become larger. This can make it difficult to discern meaningful patterns and leads to the **curse of dimensionality**. In high-dimensional spaces, data points are distributed across many dimensions, and the sparsity of the data can hinder analysis and model performance.

#### Hypercube vs. Hypersphere in Dimensionality Reduction

- **Hypercube**: A hypercube in n-dimensions is a geometric figure where each edge has equal length. In high-dimensional space, the volume of the hypercube increases exponentially with the number of dimensions, causing data points to become sparse. This sparsity makes it difficult to detect patterns, and the curse of dimensionality becomes more pronounced.

- **Hypersphere**: A hypersphere is a generalization of a circle (in 2D) and a sphere (in 3D) to n dimensions. As the number of dimensions increases, the surface area of the hypersphere becomes increasingly concentrated around the surface, and the volume inside becomes less relevant. The focus shifts to the surface area, which is where the majority of data points are located in high-dimensional spaces.

### Principal Component Analysis (PCA)

**PCA** is one of the most widely used techniques for dimensionality reduction. It works by finding new axes (principal components) in the data space, such that the variance along these new axes is maximized. The key steps in PCA are:

1. **Covariance Matrix**: Compute the covariance matrix of the data to understand how the features are correlated with each other.
2. **Eigenvalues and Eigenvectors**: Calculate the eigenvalues and eigenvectors of the covariance matrix. The eigenvectors represent the directions of maximum variance, and the eigenvalues represent the magnitude of variance along those directions.
3. **Select Principal Components**: Choose the top k eigenvectors with the largest eigenvalues to form a new feature space. The number of components k is typically chosen based on how much of the total variance is retained.
4. **Transformation**: Project the original data onto the new axes defined by the selected principal components, resulting in reduced dimensions.

PCA helps in representing the data in fewer dimensions while retaining as much variance (information) as possible, making it an effective technique for dimensionality reduction.

# FoML - 23

## Principal Component Analysis (PCA)

PCA is a technique for dimensionality reduction that aims to transform data into a lower-dimensional space while preserving the most significant variance in the data. The core idea behind PCA is to identify the directions (principal components) along which the data varies the most, and project the data along these new directions. 

The optimization problem for PCA can be formalized as:

### Maximizing Variance

Given a dataset, PCA seeks to maximize the variance along new axes (principal components), subject to the constraint that the vectors defining these new axes have unit length. Mathematically, we aim to find the **weight vector** `w` that maximizes the variance of the data along the direction defined by `w`, subject to the constraint that the magnitude of `w` is 1 (i.e., `|w| = 1`).

This optimization problem can be expressed as:

- Maximize [Var(Xw)] subject to [|w| = 1]

Where:
- `X` is the dataset (in matrix form, each row represents an observation, and each column represents a feature).
- `w` is the direction along which we are maximizing variance.

The solution to this problem is the **eigenvector** of the covariance matrix of `X` corresponding to the largest eigenvalue. This eigenvector defines the first principal component of the data. By iteratively finding the eigenvectors corresponding to the next largest eigenvalues, we can identify the axes that capture the most variance in the data, which are the principal components.

### Generalized Eigenvalue Problem

The generalized eigenvalue problem is used to find the principal components. Given the covariance matrix `Σ` of the data, we solve the following:

- `Σw = λw`

Where:
- `w` is the eigenvector (the direction of maximum variance).
- `λ` is the eigenvalue, which corresponds to the amount of variance along the direction defined by `w`.

PCA involves solving this eigenvalue problem to identify the most significant eigenvectors and eigenvalues, which are then used to transform the data into a new basis (principal component space).

---

## Gaussian Mixture Model (GMM)

**GMM** is a probabilistic model used for clustering and density estimation in unsupervised learning. It assumes that the data is generated from a mixture of several Gaussian distributions, each with its own mean and variance. The key idea is that the data is modeled as coming from a combination of different clusters, each represented by a Gaussian distribution.

### Key Components of GMM:
- **Gaussian Distribution**: Each cluster in the GMM is modeled by a Gaussian distribution (also called a normal distribution). Each Gaussian distribution has two parameters:
  - Mean (`μ`): The center of the distribution.
  - Covariance (`Σ`): The spread or shape of the distribution.

- **Mixture Model**: The overall data distribution is a weighted sum of these Gaussian distributions. Each Gaussian is associated with a weight, which reflects the proportion of data points that belong to that cluster.

Mathematically, the GMM is defined as:

- `P(x) = Σ (π_k * N(x | μ_k, Σ_k))`

Where:
- `x` is the data point.
- `π_k` is the weight of the k-th Gaussian.
- `N(x | μ_k, Σ_k)` is the Gaussian distribution with mean `μ_k` and covariance `Σ_k` for the k-th cluster.

### Expectation-Maximization (EM) Algorithm:
GMM is typically trained using the **Expectation-Maximization (EM)** algorithm:
1. **Expectation Step**: Given the current parameters of the model (means, covariances, and weights), calculate the probability (responsibility) that each data point belongs to each Gaussian component.
2. **Maximization Step**: Update the parameters (means, covariances, and weights) of the Gaussian components based on the responsibilities calculated in the E-step.

The EM algorithm iterates between these two steps to converge to the optimal parameters for the Gaussian mixture.

### Applications:
- **Unsupervised Learning**: GMM is widely used for clustering, where it groups data points into clusters based on the Gaussian distributions.
- **Density Estimation**: GMM can also be used to estimate the probability density of a given dataset.

---

## Graph-based Spectral Clustering

**Spectral clustering** is a powerful technique that uses the eigenvalues and eigenvectors of a similarity or affinity matrix to reduce the dimensionality of the data and cluster it effectively.

### Key Steps in Spectral Clustering:

1. **Represent the Dataset as a Weighted Graph**:
   - The dataset is represented as a graph where each data point is a node, and the edges between the nodes represent the similarity between the data points. 
   - The weights on the edges represent the similarity between the nodes (data points), often based on a distance metric (e.g., Gaussian similarity function).
   
2. **Construct the Laplacian Matrix**:
   - The Laplacian matrix `L` of the graph is used to describe the structure of the graph. It is computed as:
   
   - `L = D - W`
   
   Where:
   - `D` is the degree matrix, which is a diagonal matrix where each entry `D_i` is the sum of the weights of the edges connected to vertex `i`.
   - `W` is the affinity (similarity) matrix where each entry `W_ij` is the weight (similarity) between vertices `i` and `j`.

3. **Eigenvalue Decomposition**:
   - Compute the eigenvalues and eigenvectors of the Laplacian matrix. The eigenvectors corresponding to the smallest eigenvalues capture the structure of the data and can be used for clustering.
   
4. **Cluster the Data**:
   - After obtaining the eigenvectors, the data points are embedded into a lower-dimensional space. Then, a clustering algorithm (such as k-means) is applied to the eigenvectors to partition the data into clusters.

### Min Cut and Normalized Cut:

- **Min Cut**: The goal of the min cut problem is to partition a graph into two sets such that the sum of the edge weights between the sets is minimized. In spectral clustering, we aim to minimize the cut between clusters.
  
- **Normalized Cut**: Normalized cut is a variation of the min cut that normalizes the cut cost by the total weight of the edges connected to each of the two sets. The objective is to minimize the normalized cut to obtain well-balanced clusters.

   - **Normalized Cut Formula**:
   - `NCut(A, B) = (Cut(A, B) / Vol(A)) + (Cut(A, B) / Vol(B))`
   
   Where:
   - `Cut(A, B)` is the sum of the weights of edges between sets `A` and `B`.
   - `Vol(A)` and `Vol(B)` are the total edge weights connected to sets `A` and `B`.

### Spectral Clustering with Kernel Methods:
Spectral clustering can also use **kernel functions** to compute similarity matrices. This allows spectral clustering to handle more complex structures, such as spiral data, by transforming the data into a higher-dimensional space before applying clustering.

### Applications:
- **Image Segmentation**: Spectral clustering is often used in computer vision tasks like image segmentation, where the goal is to partition an image into different regions.
- **Community Detection**: In network analysis, spectral clustering is used to identify communities within a network.

# FoML - 24

## **Clustering and Spectral Clustering**

### **Partitioning into \( k \)-Clusters**
- Clustering divides data into \( k \) groups such that data points within a cluster are more similar to each other than to those in other clusters.
- The process often involves **minimizing intra-cluster distance** and **maximizing inter-cluster distance**.
- Eigenvectors can be used to **unroll data** from high-dimensional or non-linear spaces into a representation suitable for clustering, aiding the training process.

---

### **Spectral Clustering**
- **Spectral clustering** is a technique that uses the **eigenvalues** of a similarity matrix to reduce dimensions and cluster data effectively.
  
#### **Key Steps**:
1. **Create a Similarity Graph**:
   - Represent the dataset as a graph \( G = (V, E) \), where:
     - \( V \) are data points.
     - \( E \) are edges weighted by a similarity function (e.g., Gaussian kernel).

2. **Construct the Laplacian Matrix**:
   - Compute the graph Laplacian \( L \), which can be:
     - Unnormalized: \( L = D - A \) (where \( D \) is the degree matrix, and \( A \) is the adjacency matrix).
     - Normalized: \( L_{sym} = I - D^{-1/2}AD^{-1/2} \).

3. **Compute Eigenvectors**:
   - Find the eigenvalues and eigenvectors of \( L \).
   - Select the top \( k \) eigenvectors corresponding to the smallest eigenvalues.

4. **Cluster the Eigenvectors**:
   - Treat the eigenvectors as features and use a traditional clustering algorithm (e.g., \( k \)-means) to group data points.

---

## **Cluster Validity**

### **Definition**
- **Cluster validity** evaluates the quality of clustering results based on how well the data has been partitioned.

### **Types of Indices**
1. **External Index**:
   - Measures the agreement between a clustering result and a ground truth partition.
   - Examples: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI).

2. **Internal Index**:
   - Assesses clustering quality without external ground truth by analyzing properties like cohesion and separation.
   - Example: **Silhouette Coefficient**:
     - Combines ideas of **cohesion** (how close points in a cluster are) and **separation** (how distinct clusters are).
     - Silhouette value for a point:
$$       [
       S(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
       ]$$
       where:
       - \( a(i) \): Mean intra-cluster distance for point \( i \).
       - \( b(i) \): Mean nearest-cluster distance for point \( i \).

3. **Relative Index**:
   - Compares the quality of different clustering algorithms on the same dataset.

---

## **Dimensionality Reduction**

### **Principal Component Analysis (PCA)**
- **PCA** is a linear dimensionality reduction technique used to transform data into a lower-dimensional space while minimizing information loss.

#### **Key Concepts**:
1. **Projection**:
   - PCA finds a new set of axes (principal components) such that the first component captures the most variance, the second captures the next highest variance, and so on.

2. **Optimization Objective**:
   - Minimize the reconstruction error:
    $$ [
     \min ||X - X_{reconstructed}||^2
     ]$$
     Where \( X_{reconstructed} \) is the data projected onto the principal components.

3. **Steps**:
   - Center the data by subtracting the mean.
   - Compute the covariance matrix.
   - Find the eigenvalues and eigenvectors of the covariance matrix.
   - Select the top \( k \) eigenvectors as the principal components.

4. **Applications**:
   - Reducing dimensions for visualization.
   - Preprocessing for machine learning models to eliminate noise.

---

### **Linear Discriminant Analysis (LDA)**
- Unlike PCA, **LDA** is supervised and aims to maximize the separation between classes.
- Projects data onto a linear space that maximizes the ratio of **between-class variance** to **within-class variance**.

---

### **T-SNE (t-Distributed Stochastic Neighbor Embedding)**
- A non-linear dimensionality reduction technique for high-dimensional data visualization.

#### **Key Concepts**:
1. **Perplexity**:
   - A user-defined parameter controlling the balance between local and global structure.
   - Higher perplexity captures broader global relationships; lower perplexity focuses on local clustering.

2. **Steps**:
   - Compute pairwise similarities between data points in high-dimensional space.
   - Compute pairwise similarities in low-dimensional space.
   - Minimize the Kullback-Leibler divergence between the two distributions.

#### **Applications**:
- Visualizing high-dimensional data (e.g., embeddings from neural networks).

---

### **Kernel PCA**
- An extension of PCA using kernel functions to handle non-linear data structures.
- Transforms data into a higher-dimensional feature space where linear separation becomes possible.
- Steps:
  1. Compute the kernel matrix.
  2. Find eigenvalues and eigenvectors of the kernel matrix.
  3. Use the top \( k \) eigenvectors for projection.

---

### **Multi-Dimensional Scaling (MDS)**
- A technique that represents data in a lower-dimensional space while preserving pairwise distances.
- Often used for visualization when the exact structure of the data is less important.

---

### **Manifold Learning**
- Manifold learning methods aim to uncover the low-dimensional structure (manifold) underlying high-dimensional data.

#### **ISOmap**:
- Constructs a graph of nearest neighbors and computes geodesic distances on the graph.
- Applies classical MDS to embed the data into a low-dimensional space.

#### **T-SNE**:
- Focuses on preserving neighborhood structures, as detailed above.

---

## **Comparisons of Techniques**
| Technique      | Linear/Non-linear | Use Case                                   |
|----------------|-------------------|--------------------------------------------|
| PCA            | Linear            | Dimensionality reduction, noise removal    |
| LDA            | Linear            | Classification, supervised dimensionality reduction |
| Kernel PCA     | Non-linear        | Non-linear dimensionality reduction        |
| T-SNE          | Non-linear        | High-dimensional data visualization        |
| ISOmap         | Non-linear        | Manifold learning, preserving geodesic distances |
| MDS            | Non-linear        | General dimensionality reduction for distances |

---

## **Applications**
1. **Data Preprocessing**:
   - Reduce noise and eliminate irrelevant features.
2. **Visualization**:
   - Visualizing complex, high-dimensional datasets in 2D or 3D.
3. **Feature Engineering**:
   - Creating meaningful features for machine learning models.
4. **Cluster Analysis**:
   - Identifying natural groupings in the data.


**Done**
---
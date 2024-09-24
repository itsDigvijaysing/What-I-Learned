Links: [IIT Hyderabad](IIT%20Hyderabad.md), 

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

Quiz Happened & I was absent 




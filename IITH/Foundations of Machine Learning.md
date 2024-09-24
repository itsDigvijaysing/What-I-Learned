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

# FoML - 4

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

## K-Nearest Neighbors (KNN)
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

# FoML - 5

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
    $$\[
    I_m = - \sum_{i} p_{mi} \log_2(p_{mi})
    \]$$
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

# FoML - 6

## Bayes Theorem
- **Bayes Theorem** provides a way to calculate the **posterior probability** of a hypothesis based on prior knowledge and evidence.
- Formula:
$$  \[
  P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}
  \]$$
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
 $$   \[
    \log P(H|E) = \log P(H) + \sum_{i=1}^{n} \log P(E_i|H)
    \]$$
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


# FoML - 06

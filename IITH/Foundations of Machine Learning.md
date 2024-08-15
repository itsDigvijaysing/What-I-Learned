Links: [IIT Hyderabad](IIT%20Hyderabad.md)

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
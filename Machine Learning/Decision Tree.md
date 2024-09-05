## Implementing Decision Tree

![Decision Tree](../Archive/Attachment/DecisionTree.png)
- **If it classifies into category -> Classification tree**
- **If it predicts numeric values -> Regression tree**

### Process

- When we get dataset we divide that in training data & testing data.
- Calculate information gain with each split & until stopping criteria is reached
- If leaf node is pure then 
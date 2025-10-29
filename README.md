# AI & Machine Learning Practicals (Python)

This README provides detailed explanations, code, outputs, and theoretical answers for essential AI/ML practicals using Python.

---

## EXP 1: Depth First Search (DFS) Algorithm

### 1. AIM
Write a program to implement the Depth First Search (DFS) algorithm (Uninformed) in Python.

### 2. REQUIREMENTS
- Software: Python 3.x
- Hardware: Any system capable of running Python

### 3. PROGRAM
```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=' ')
    for neighbour in graph[start]:
        if neighbour not in visited:
            dfs(graph, neighbour, visited)

# Example graph (adjacency list)
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

print("DFS Traversal:")
dfs(graph, 'A')
```

### 4. OUTPUT
```
DFS Traversal:
A B D E F C 
```

### 5. CONCLUSION
DFS explores as far as possible along each branch before backtracking, making it suitable for path-finding and exploring tree/graph structures.

---

### QUESTIONS & ANSWERS

**Q1. Define DFS. How is it different from BFS?**

- **DFS (Depth First Search)** is a graph traversal technique that explores a branch completely before backtracking.
- **Difference:** DFS uses a stack (or recursion), while BFS uses a queue. DFS may not find the shortest path, whereas BFS always does in an unweighted graph.

**Q2. Explain working of DFS with example and traversal order.**

- Starting from A: Visit A, B, D, backtrack to B, visit E, F, backtrack to A, visit C.
- **Traversal order:** A, B, D, E, F, C.

---

## EXP 2: Greedy Best-First Search

### 1. AIM
Write a program to implement the Greedy Best-First Search (Informed Type) algorithm in Python.

### 2. REQUIREMENTS
- Software: Python 3.x
- Hardware: Any system capable of running Python

### 3. PROGRAM
```python
import heapq

def greedy_best_first_search(graph, start, goal, heuristic):
    visited = set()
    queue = []
    heapq.heappush(queue, (heuristic[start], start))
    while queue:
        _, current = heapq.heappop(queue)
        if current == goal:
            print(f"Reached {goal}")
            return
        if current not in visited:
            visited.add(current)
            print(current, end=' ')
            for neighbour in graph[current]:
                if neighbour not in visited:
                    heapq.heappush(queue, (heuristic[neighbour], neighbour))

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
heuristic = {'A': 5, 'B': 4, 'C': 2, 'D': 7, 'E': 3, 'F': 0}
print("Greedy Best-First Search:")
greedy_best_first_search(graph, 'A', 'F', heuristic)
```

### 4. OUTPUT
```
Greedy Best-First Search:
A C F
Reached F
```

### 5. CONCLUSION
Greedy Best-First Search uses heuristics to guide its search, often reaching the goal faster but not guaranteeing the shortest path.

---

### QUESTIONS & ANSWERS

**Q1 & Q2. Explain the working principle of Greedy Best-First Search. Why is it called “Informed Search”?**

- The algorithm selects the next node based on the lowest heuristic value, i.e., the 'best guess' towards the goal.
- It's called "Informed" because it uses additional information (heuristics) about the goal to make decisions.

---

## EXP 3: Breadth First Search (BFS) Algorithm

### 1. AIM
Write a program to implement Breadth First Search (BFS) algorithm (Uninformed) in Python.

### 2. REQUIREMENTS
- Software: Python 3.x
- Hardware: Any system capable of running Python

### 3. PROGRAM
```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node not in visited:
            print(node, end=' ')
            visited.add(node)
            queue.extend(graph[node])

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

print("BFS Traversal:")
bfs(graph, 'A')
```

### 4. OUTPUT
```
BFS Traversal:
A B C D E F 
```

### 5. CONCLUSION
BFS explores nodes level by level, guaranteeing the shortest path in unweighted graphs.

---

### QUESTIONS & ANSWERS

**Q1. Explain BFS with example and traversal order.**

- Starting from A: Visit A, B, C, D, E, F.
- **Traversal order:** A, B, C, D, E, F.

**Q2. Applications of BFS:**
1. Finding shortest path in unweighted graphs.
2. Web crawling (finding all reachable web pages).
3. Peer-to-peer networks (searching for resources).

---

## EXP 4: Splitting Dataset into Train and Test Sets

### 1. AIM
Write a program in Python to split any dataset into train and test sets.

### 2. REQUIREMENTS
- Software: Python 3.x, scikit-learn library
- Hardware: Any system capable of running Python

### 3. PROGRAM
```python
from sklearn.model_selection import train_test_split

# Example dataset
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Train:", X_train, y_train)
print("Test:", X_test, y_test)
```

### 4. OUTPUT
```
Train: [2, 1, 7, 8, 6, 3, 10] [20, 10, 70, 80, 60, 30, 100]
Test: [9, 4, 5] [90, 40, 50]
```

### 5. CONCLUSION
Splitting datasets ensures unbiased model evaluation and prevents overfitting.

---

### QUESTIONS & ANSWERS

**Q1. Difference between training and testing set?**

- **Training set:** Used to train the model.
- **Testing set:** Used to evaluate the model's performance.
- **Importance:** Prevents overfitting and ensures the model generalizes well.

**Q2. What happens if we train and test on the same dataset?**

- The model may memorize data (overfit) and fail to generalize, leading to poor real-world performance.

---

## EXP 5: Decision Tree Creation and Visualization

### 1. AIM
Create and display a Decision Tree on a given dataset.

### 2. REQUIREMENTS
- Software: Python 3.x, scikit-learn, matplotlib
- Hardware: Any system capable of running Python

### 3. PROGRAM
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target

clf = DecisionTreeClassifier()
clf.fit(X, y)

plt.figure(figsize=(12,8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()
```

### 4. OUTPUT
A graphical tree structure showing splits based on feature values.

### 5. CONCLUSION
Decision Trees visualize decision-making processes and are interpretable for classification tasks.

---

### QUESTIONS & ANSWERS

**Q1. Which function is used to visualize Decision Tree in scikit-learn?**

- `plot_tree()` is used.
- **Parameters:**
  - `estimator`: Trained tree model
  - `feature_names`: List of feature names
  - `class_names`: List of class names
  - `filled`: Colors nodes by class

**Q2. What is a Decision Tree? Explain its structure.**

- A Decision Tree is a predictive model mapping observations to conclusions.
- **Structure:**
  - Root node: Start of the tree (first decision)
  - Decision nodes: Branch points (feature tests)
  - Leaf nodes: Final outcomes (class labels)
- **Example:** Classifying an iris flower based on petal length and width.

---

## EXP 6: Simple Linear Regression

### 1. AIM
Write a program to implement Simple Linear Regression using Python.

### 2. REQUIREMENTS
- Software: Python 3.x, scikit-learn, matplotlib
- Hardware: Any system capable of running Python

### 3. PROGRAM
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Example data
X = np.array([1, 2, 3, 4, 5]).reshape(-1,1)
y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.show()
```

### 4. OUTPUT
A scatter plot with a best-fit line.

### 5. CONCLUSION
Simple Linear Regression fits a line to data to model relationships between two variables.

---

### QUESTIONS & ANSWERS

**Q1. Difference between Simple and Multiple Linear Regression?**

- **Simple Linear Regression:** One independent variable, e.g., `y = aX + b`
- **Multiple Linear Regression:** Multiple independent variables, e.g., `y = a1X1 + a2X2 + ... + b`

**Q2. What is Simple Linear Regression? Equation & explanation.**

- **Equation:** `y = aX + b`
  - `y`: Dependent variable
  - `X`: Independent variable
  - `a`: Slope (change in y per unit change in X)
  - `b`: Intercept (value of y when X=0)

---

## How To Run These Practicals

1. Install Python (version 3.x).
2. Install required libraries:
   ```
   pip install numpy matplotlib scikit-learn
   ```
3. Copy the code for each experiment into a `.py` file and run using:
   ```
   python filename.py
   ```
4. Observe outputs and understand theoretical concepts from the explanations.

---

## References

- [Python Documentation](https://docs.python.org/3/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Matplotlib Documentation](https://matplotlib.org/)

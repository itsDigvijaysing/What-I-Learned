# Complete C++ Learning Guide - 15 DSA Problems

---

## **Problem 1: Logging Roadside Trees**

### 🎯 Intuition
Think of this like playing a video game where:
- You control a robot on a road
- You must collect items (trees) in descending size order
- Moving costs time, cutting costs time
- You want to finish as fast as possible

**Key Constraint:** Once you cut a tree of length 5, you can only cut trees of length ≤5 next.

### 🧠 Approach

**Step 1: Understanding State**
At any moment, the robot has:
- Current position
- Last tree length cut (limits what we can cut next)
- Which side we're on (left/right)

**Step 2: Choice at Each Step**
From position i, we can:
1. Move forward/backward to another position
2. Cut a tree (if its length ≤ last_cut_length)
3. Cut both sides if both trees are valid

**Step 3: BFS Solution**
Use BFS to explore all possibilities and find shortest path.

### 📝 Step-by-Step Solution

```cpp
#include <bits/stdc++.h>
using namespace std;

struct State {
    int pos;           // current position
    int maxLen;        // max length we can cut now (last cut)
    int time;          // time taken so far
};

int solve(int N, vector<int>& left, vector<int>& right) {
    // State: (position, max_length_allowed)
    map<pair<int,int>, int> visited; // min time to reach this state
    
    queue<State> q;
    
    // Start at position 0, can cut any length initially
    q.push({0, INT_MAX, 0});
    visited[{0, INT_MAX}] = 0;
    
    int answer = INT_MAX;
    
    while (!q.empty()) {
        State curr = q.front();
        q.pop();
        
        int pos = curr.pos;
        int maxLen = curr.maxLen;
        int time = curr.time;
        
        // If we reached end and cut all trees, update answer
        if (pos == N) {
            // Check if all trees are cut (tracked separately in practice)
            answer = min(answer, time);
            continue;
        }
        
        // Option 1: Move forward without cutting
        if (pos + 1 <= N) {
            int newTime = time + 1;
            auto key = make_pair(pos + 1, maxLen);
            if (visited.find(key) == visited.end() || visited[key] > newTime) {
                visited[key] = newTime;
                q.push({pos + 1, maxLen, newTime});
            }
        }
        
        // Option 2: Cut tree at current position
        // Try cutting left
        if (pos > 0 && pos < N && left[pos] > 0 && left[pos] <= maxLen) {
            int newTime = time + 1; // cutting time
            int newMaxLen = left[pos];
            auto key = make_pair(pos, newMaxLen);
            if (visited.find(key) == visited.end() || visited[key] > newTime) {
                visited[key] = newTime;
                q.push({pos, newMaxLen, newTime});
            }
        }
        
        // Try cutting right
        if (pos > 0 && pos < N && right[pos] > 0 && right[pos] <= maxLen) {
            int newTime = time + 1;
            int newMaxLen = right[pos];
            auto key = make_pair(pos, newMaxLen);
            if (visited.find(key) == visited.end() || visited[key] > newTime) {
                visited[key] = newTime;
                q.push({pos, newMaxLen, newTime});
            }
        }
        
        // Try cutting both (if both valid)
        if (pos > 0 && pos < N && 
            left[pos] > 0 && right[pos] > 0 &&
            left[pos] <= maxLen && right[pos] <= maxLen) {
            int newTime = time + 2; // both cutting
            int newMaxLen = min(left[pos], right[pos]); // smaller one limits next
            auto key = make_pair(pos, newMaxLen);
            if (visited.find(key) == visited.end() || visited[key] > newTime) {
                visited[key] = newTime;
                q.push({pos, newMaxLen, newTime});
            }
        }
    }
    
    return answer;
}

int main() {
    int N;
    cin >> N;
    
    vector<int> left(N+1), right(N+1);
    for (int i = 0; i <= N; i++) cin >> left[i];
    for (int i = 0; i <= N; i++) cin >> right[i];
    
    cout << solve(N, left, right) << endl;
    
    return 0;
}
```

### ⚠️ Note
This is a simplified BFS approach. For the actual problem, you'll need to track which trees have been cut using a bitmask or more sophisticated state management.

**Better Approach:** Dynamic Programming with state (pos, last_length, trees_cut_bitmask)

---

## **Problem 2: Min Cost Stone Pulling**

### 🎯 Intuition
Imagine removing stones from a row:
- When you remove a stone with 2 neighbors → Cost = 0
- With 1 neighbor → Cost = A[i]
- With 0 neighbors → Cost = B[i]

**Key Insight:** The FIRST stone you remove in a group has 2 neighbors, the LAST has 0.

### 🧠 Approach - Interval DP

Think recursively: To remove all stones in range [L, R]:
1. Choose which stone to remove FIRST (it has 2 neighbors → cost 0)
2. This splits the problem into two subproblems
3. The stone removed LAST in each subproblem has 0 neighbors

### 📝 Step-by-Step Solution

```cpp
#include <bits/stdc++.h>
using namespace std;

// dp[i][j] = minimum cost to remove all stones from i to j
vector<vector<int>> dp;
vector<int> A, B; // A = cost with 1 neighbor, B = cost with 0 neighbors
int N;

int solve(int left, int right) {
    // Base case: single stone
    if (left == right) {
        return B[left]; // no neighbors, cost B[i]
    }
    
    // Already computed
    if (dp[left][right] != -1) {
        return dp[left][right];
    }
    
    int minCost = INT_MAX;
    
    // Try removing each stone 'i' FIRST in this range
    // When we remove it first, it has 2 neighbors (left and right exist)
    // So cost is 0
    for (int i = left; i <= right; i++) {
        int cost = 0; // removing i first (2 neighbors)
        
        // After removing i, we have two subproblems
        // Left part: [left, i-1]
        // Right part: [i+1, right]
        
        if (i > left && i < right) {
            // i is in middle
            cost += solve(left, i - 1);   // left subproblem
            cost += solve(i + 1, right);  // right subproblem
            cost += A[i]; // stone i will be last removed, has 1 neighbor (boundary)
            // Actually, stone i removed first, so cost already 0
            // We need to reconsider...
        } else if (i == left) {
            // i is leftmost
            if (i < right) {
                cost += solve(i + 1, right);
                cost += A[i]; // i removed last from left side, has 1 neighbor
            }
        } else if (i == right) {
            // i is rightmost
            if (i > left) {
                cost += solve(left, i - 1);
                cost += A[i]; // i removed last from right side
            }
        }
        
        minCost = min(minCost, cost);
    }
    
    return dp[left][right] = minCost;
}

// CORRECT APPROACH:
int solveCorrect(int left, int right) {
    if (left > right) return 0;
    if (left == right) return B[left];
    
    if (dp[left][right] != -1) return dp[left][right];
    
    int minCost = INT_MAX;
    
    // Try removing each stone k LAST
    // When k is removed last, its neighbors are already gone
    for (int k = left; k <= right; k++) {
        int cost = B[k]; // k removed last, 0 neighbors
        
        // Cost to remove [left, k-1] with k as boundary (1 neighbor)
        if (k > left) {
            cost += solveCorrect(left, k - 1);
            // Adjust for boundary: leftmost in this range has k as neighbor
        }
        
        // Cost to remove [k+1, right] with k as boundary
        if (k < right) {
            cost += solveCorrect(k + 1, right);
        }
        
        minCost = min(minCost, cost);
    }
    
    return dp[left][right] = minCost;
}

int main() {
    int T;
    cin >> T;
    
    for (int tc = 1; tc <= T; tc++) {
        cin >> N;
        A.resize(N);
        B.resize(N);
        
        for (int i = 0; i < N; i++) cin >> A[i];
        for (int i = 0; i < N; i++) cin >> B[i];
        
        dp.assign(N, vector<int>(N, -1));
        
        cout << "#" << tc << " " << solveCorrect(0, N - 1) << endl;
    }
    
    return 0;
}
```

### 💡 Key Learning
**Interval DP Pattern:**
- `dp[i][j]` = answer for subarray [i, j]
- Try all ways to split the interval
- Combine subproblem solutions

---

## **Problem 3: Card Selection (Maximize Points)**

### 🎯 Intuition
Points = Sum of values + Bonus from suits
Bonus = (Hearts)² + (Diamonds)² + (Clubs)² + (Spades)²

The squared term means: **collecting more cards of the same suit gives exponential bonus!**

### 🧠 Approach

**Key Observation:**
- We must select exactly K cards
- We want high values AND good suit distribution

**Strategy:**
1. Try all possible suit distributions (how many of each suit)
2. For each distribution, greedily pick highest value cards

### 📝 Step-by-Step Solution

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Card {
    int value;
    int suit; // 0=Heart, 1=Diamond, 2=Club, 3=Spade
};

int solve(vector<Card>& cards, int K) {
    int N = cards.size();
    
    // Separate cards by suit
    vector<vector<int>> bySuit(4);
    for (auto& c : cards) {
        bySuit[c.suit].push_back(c.value);
    }
    
    // Sort each suit by value (descending)
    for (int i = 0; i < 4; i++) {
        sort(bySuit[i].rbegin(), bySuit[i].rend());
    }
    
    int maxPoints = 0;
    
    // Try all possible distributions: (c0, c1, c2, c3) where c0+c1+c2+c3 = K
    // This is like iterating with 4 nested loops
    function<void(int, int, vector<int>&)> tryDistributions = 
        [&](int suit, int remaining, vector<int>& counts) {
        
        if (suit == 4) {
            if (remaining != 0) return; // invalid distribution
            
            // Calculate points for this distribution
            int valueSum = 0;
            int suitBonus = 0;
            
            for (int s = 0; s < 4; s++) {
                int cnt = counts[s];
                if (cnt > bySuit[s].size()) return; // not enough cards
                
                // Take top 'cnt' cards from this suit
                for (int i = 0; i < cnt; i++) {
                    valueSum += bySuit[s][i];
                }
                
                suitBonus += cnt * cnt;
            }
            
            maxPoints = max(maxPoints, valueSum + suitBonus);
            return;
        }
        
        // Try all counts for current suit
        int maxCnt = min(remaining, (int)bySuit[suit].size());
        for (int cnt = 0; cnt <= maxCnt; cnt++) {
            counts[suit] = cnt;
            tryDistributions(suit + 1, remaining - cnt, counts);
        }
    };
    
    vector<int> counts(4);
    tryDistributions(0, K, counts);
    
    return maxPoints;
}

int main() {
    int T;
    cin >> T;
    
    while (T--) {
        int N, K;
        cin >> N >> K;
        
        vector<Card> cards(N);
        for (int i = 0; i < N; i++) {
            cin >> cards[i].value >> cards[i].suit;
        }
        
        cout << solve(cards, K) << endl;
    }
    
    return 0;
}
```

### ⏱️ Complexity
- Time: O(K³ × N) - trying all partitions and sorting
- Space: O(N)

### 💡 Optimization
For large K, use DP: `dp[suit][cards_selected]` = maximum points

---

## **Problem 4: Lattice Path Intersections**

### 🎯 Intuition
You have a path made of horizontal and vertical line segments.
Check if target points lie on any segment.

### 🧠 Approach

**Step 1:** Build segments from turning points
**Step 2:** For each target point, check all segments
**Step 3:** Check if point is on segment:
- Horizontal segment: same y, x in range
- Vertical segment: same x, y in range

### 📝 Step-by-Step Solution

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Point {
    long long x, y;
};

struct Segment {
    Point start, end;
    bool isHorizontal;
};

bool isOnSegment(Point p, Segment seg) {
    if (seg.isHorizontal) {
        // Horizontal: y is constant, x varies
        if (p.y != seg.start.y) return false;
        
        long long minX = min(seg.start.x, seg.end.x);
        long long maxX = max(seg.start.x, seg.end.x);
        
        return p.x >= minX && p.x <= maxX;
    } else {
        // Vertical: x is constant, y varies
        if (p.x != seg.start.x) return false;
        
        long long minY = min(seg.start.y, seg.end.y);
        long long maxY = max(seg.start.y, seg.end.y);
        
        return p.y >= minY && p.y <= maxY;
    }
}

int main() {
    int N, M;
    cin >> N >> M;
    
    vector<Point> targets(N);
    for (int i = 0; i < N; i++) cin >> targets[i].x;
    for (int i = 0; i < N; i++) cin >> targets[i].y;
    
    vector<Point> path(M);
    for (int i = 0; i < M; i++) cin >> path[i].x;
    for (int i = 0; i < M; i++) cin >> path[i].y;
    
    // Build segments
    vector<Segment> segments;
    for (int i = 0; i < M - 1; i++) {
        Segment seg;
        seg.start = path[i];
        seg.end = path[i + 1];
        
        if (seg.start.y == seg.end.y) {
            seg.isHorizontal = true;
        } else {
            seg.isHorizontal = false;
        }
        
        segments.push_back(seg);
    }
    
    // Count points on path
    int count = 0;
    for (auto& p : targets) {
        bool found = false;
        for (auto& seg : segments) {
            if (isOnSegment(p, seg)) {
                found = true;
                break;
            }
        }
        if (found) count++;
    }
    
    cout << count << endl;
    
    return 0;
}
```

### 💡 Key Learning
**Simple geometry:** Just check ranges carefully!

---

## **Problem 5: Warehouse Stock Reduction**

### 🎯 Intuition
Each day:
1. Stock increases by B[i]
2. We export one item (set to 0)
3. Report total stock

We want: total stock ≤ K

### 🧠 Approach

**Key Insight:** 
- If all B[i] ≥ 0 and sum(B) > 0, stock grows forever → impossible
- We need sum(B) ≤ 0 (net decrease) for it to be possible

**Strategy:** Binary search on days, simulate the process

### 📝 Step-by-Step Solution

```cpp
#include <bits/stdc++.h>
using namespace std;

bool canAchieve(int days, vector<long long>& A, vector<long long>& B, long long K) {
    int N = A.size();
    
    // After 'days' days, what's the minimum total stock?
    // Strategy: export items with highest (A[i] + days * B[i]) each day
    
    vector<long long> stocks(N);
    for (int i = 0; i < N; i++) {
        stocks[i] = A[i] + (long long)days * B[i];
    }
    
    // Sort to export largest items
    sort(stocks.rbegin(), stocks.rend());
    
    // Remove top 'days' items (exported)
    long long totalStock = 0;
    for (int i = days; i < N; i++) {
        totalStock += max(0LL, stocks[i]);
    }
    
    return totalStock <= K;
}

int main() {
    int N;
    long long K;
    cin >> N >> K;
    
    vector<long long> A(N), B(N);
    for (int i = 0; i < N; i++) cin >> A[i];
    for (int i = 0; i < N; i++) cin >> B[i];
    
    // Check if possible
    long long sumB = 0;
    for (int i = 0; i < N; i++) sumB += B[i];
    
    if (sumB > 0 && N <= 1000) {
        // Stock grows unboundedly
        cout << -1 << endl;
        return 0;
    }
    
    // Binary search on days
    int left = 0, right = 1000000;
    int answer = -1;
    
    while (left <= right) {
        int mid = (left + right) / 2;
        
        if (canAchieve(mid, A, B, K)) {
            answer = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    cout << answer << endl;
    
    return 0;
}
```

### 💡 Key Learning
**Binary Search Pattern:**
- Can we achieve goal in X days? → monotonic (if possible in X, possible in X+1)
- Binary search to find minimum X

---

## **Problem 6: Red and Blue Necklace**

### 🎯 Intuition
This is the classic "**Longest Subarray with Equal 0s and 1s**" problem!

Convert R → +1, B → -1
Find longest subarray with sum = 0

### 🧠 Approach

Use **Prefix Sum + HashMap**:
- If prefix[i] = prefix[j], then sum[i+1...j] = 0

### 📝 Step-by-Step Solution

```cpp
#include <bits/stdc++.h>
using namespace std;

int longestEqualSubarray(string s) {
    int N = s.size();
    
    // Convert to +1/-1
    vector<int> arr(N);
    for (int i = 0; i < N; i++) {
        arr[i] = (s[i] == 'R') ? 1 : -1;
    }
    
    // Find longest subarray with sum = 0
    unordered_map<int, int> prefixMap; // prefix_sum → first_index
    prefixMap[0] = -1; // important: empty prefix
    
    int prefixSum = 0;
    int maxLen = 0;
    
    for (int i = 0; i < N; i++) {
        prefixSum += arr[i];
        
        if (prefixMap.find(prefixSum) != prefixMap.end()) {
            // Found same prefix sum before
            int prevIdx = prefixMap[prefixSum];
            int len = i - prevIdx;
            maxLen = max(maxLen, len);
        } else {
            prefixMap[prefixSum] = i;
        }
    }
    
    return maxLen;
}

int main() {
    string s;
    cin >> s;
    
    int maxEqual = longestEqualSubarray(s);
    int answer = s.size() - maxEqual;
    
    cout << answer << endl;
    
    return 0;
}
```

### 💡 Key Learning
**Prefix Sum Technique:**
- Transform problem into finding sum = target
- Use hashmap to find matching prefix sums

---

## **Problem 7: City Logistics (Truck & Garage)**

### 🎯 Intuition
This is a **state-space search** problem:
- Find shortest path visiting warehouses and delivering to airport
- Movement cost depends on current load

### 🧠 Approach

**State:** (position, goods_count, warehouses_visited_bitmask)

Since max 13 warehouses, we can use bitmask (2^13 = 8192 states)

Use **Dijkstra** or **BFS** with priority queue

### 📝 Step-by-Step Solution

```cpp
#include <bits/stdc++.h>
using namespace std;

struct State {
    int x, y;          // position
    int goods;         // goods on truck
    int mask;          // warehouses visited (bitmask)
    int delivered;     // goods delivered so far
    int cost;          // cost so far
    
    bool operator>(const State& other) const {
        return cost > other.cost;
    }
};

int dx[] = {0, 0, 1, -1};
int dy[] = {1, -1, 0, 0};

int solve(vector<vector<int>>& grid, int H, int W, int C) {
    // Find garage, warehouses, airport
    int gx, gy;
    vector<pair<int,int>> warehouses;
    int ax, ay;
    
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            if (grid[i][j] == 2) {
                gx = i; gy = j;
            } else if (grid[i][j] == 3) {
                warehouses.push_back({i, j});
            } else if (grid[i][j] == 4) {
                ax = i; ay = j;
            }
        }
    }
    
    int numWarehouses = warehouses.size();
    
    // State: (x, y, goods, mask) → minimum cost
    map<tuple<int,int,int,int>, int> visited;
    
    priority_queue<State, vector<State>, greater<State>> pq;
    pq.push({gx, gy, 0, 0, 0, 0});
    
    int maxDelivered = 0;
    
    while (!pq.empty()) {
        State curr = pq.top();
        pq.pop();
        
        if (curr.cost > C) continue;
        
        auto key = make_tuple(curr.x, curr.y, curr.goods, curr.mask);
        if (visited.count(key) && visited[key] <= curr.cost) continue;
        visited[key] = curr.cost;
        
        maxDelivered = max(maxDelivered, curr.delivered);
        
        // Try moving to adjacent cells
        for (int d = 0; d < 4; d++) {
            int nx = curr.x + dx[d];
            int ny = curr.y + dy[d];
            
            if (nx < 0 || nx >= H || ny < 0 || ny >= W) continue;
            if (grid[nx][ny] == 1) continue; // tree
            
            int moveCost = 1 + curr.goods;
            int newCost = curr.cost + moveCost;
            
            if (newCost > C) continue;
            
            State next = curr;
            next.x = nx;
            next.y = ny;
            next.cost = newCost;
            
            // Check if warehouse
            for (int w = 0; w < numWarehouses; w++) {
                if (warehouses[w].first == nx && warehouses[w].second == ny) {
                    // Option: load goods
                    if (!(curr.mask & (1 << w))) {
                        State loaded = next;
                        loaded.goods++;
                        loaded.mask |= (1 << w);
                        pq.push(loaded);
                    }
                }
            }
            
            // Check if airport
            if (grid[nx][ny] == 4 && curr.goods > 0) {
                // Option: unload
                State unloaded = next;
                unloaded.delivered = curr.delivered + curr.goods;
                unloaded.goods = 0;
                pq.push(unloaded);
            }
            
            // Always push state without action
            pq.push(next);
        }
    }
    
    return maxDelivered;
}

int main() {
    int T;
    cin >> T;
    
    while (T--) {
        int H, W, C;
        cin >> H >> W >> C;
        
        vector<vector<int>> grid(H, vector<int>(W));
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                cin >> grid[i][j];
            }
        }
        
        cout << solve(grid, H, W, C) << endl;
    }
    
    return 0;
}
```

### 💡 Key Learning
**State Space Search:**
- Define complete state
- Use Dijkstra for minimum cost
- Bitmask for tracking subsets

---

## **Problem 8: Tile Selection (Minimize Max Difference)**

### 🎯 Intuition
Binary search on the answer (maximum difference).

For each candidate answer D, check if we can select K tiles with max_diff ≤ D.

### 🧠 Approach

**Step 1:** Binary search on answer (0 to max_possible_diff)
**Step 2:** For each D, check if feasible:
- Sort tiles by one dimension
- Use sliding window or greedy to find K tiles

### 📝 Step-by-Step Solution

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Tile {
    int w, h;
};

bool canSelect(vector<Tile>& tiles, int K, int maxDiff) {
    int N = tiles.size();
    
    // Try all possible "anchor" tiles
    for (int i = 0; i < N; i++) {
        vector<int> validTiles;
        
        for (int j = 0; j < N; j++) {
            int widthDiff = abs(tiles[i].w - tiles[j].w);
            int heightDiff = abs(tiles[i].h - tiles[j].h);
            int diff = max(widthDiff, heightDiff);
            
            if (diff <= maxDiff) {
                validTiles.push_back(j);
            }
        }
        
        if (validTiles.size() >= K) {
            return true;
        }
    }
    
    return false;
}

int main() {
    int N, K;
    cin >> N >> K;
    
    vector<Tile> tiles(N);
    for (int i = 0; i < N; i++) {
        cin >> tiles[i].w >> tiles[i].h;
    }
    
    // Binary search on answer
    int left = 0, right = 100000;
    int answer = right;
    
    while (left <= right) {
        int mid = (left + right) / 2;
        
        if (canSelect(tiles, K, mid)) {
            answer = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    cout << answer << endl;
    
    return 0;
}
```

### 💡 Key Learning
**Binary Search on Answer:**
- If answer X works, all Y > X work too (monotonic)
- Binary search to find minimum valid X

---

## **Problem 9: String Chain Merging**

### 🎯 Intuition
Think of this as a **graph problem**:
- Each string is a node
- Edge from A to B if A.last = B.first
- Find longest path where start_char = end_char

### 🧠 Approach

**DP Approach:**
`dp[start_digit][end_digit]` = max length of chain starting with start_digit, ending with end_digit

### 📝 Step-by-Step Solution

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N;
    cin >> N;
    
    vector<string> strings(N);
    for (int i = 0; i < N; i++) {
        cin >> strings[i];
    }
    
    // dp[start][end] = max length with chain starting 'start', ending 'end'
    map<pair<char, char>, int> dp;
    
    // Initialize: each string alone
    for (auto& s : strings) {
        char start = s[0];
        char end = s[s.size() - 1];
        dp[{start, end}] = max(dp[{start, end}], (int)s.size());
    }
    
    // Try combining chains
    bool updated = true;
    while (updated) {
        updated = false;
        map<pair<char, char>, int> newDp = dp;
        
        for (auto& [key1, len1] : dp) {
            char end1 = key1.second;
            
            for (auto& [key2, len2] : dp) {
                char start2 = key2.first;
                
                if (end1 == start2) {
                    // Can connect
                    char newStart = key1.first;
                    char newEnd = key2.second;
                    int newLen = len1 + len2;
                    
                    if (newLen > newDp[{newStart, newEnd}]) {
                        newDp[{newStart, newEnd}] = newLen;
                        updated = true;
                    }
                }
            }
        }
        
        dp = newDp;
    }
    
    // Find max length where start = end
    int answer = 0;
    for (auto& [key, len] : dp) {
        if (key.first == key.second) {
            answer = max(answer, len);
        }
    }
    
    cout << answer << endl;
    
    return 0;
}
```

### 💡 Key Learning
**DP on Graphs:**
- State represents both endpoints of path
- Combine paths by matching endpoints

---

## **Problem 10: Car Parking (Geometric Moves)**

### 🎯 Intuition
Key insight: In k drives, you can move total distance = 1+2+3+...+k = k(k+1)/2

But you can "waste" moves by going back and forth!

**Constraint:** You need k(k+1)/2 ≥ distance AND same parity

### 🧠 Approach

1. Calculate Manhattan distance for each car
2. Check parity (all must be same)
3. Binary search on k

### 📝 Step-by-Step Solution

```cpp
#include <bits/stdc++.h>
using namespace std;

long long manhattanDist(long long x1, long long y1, long long x2, long long y2) {
    return abs(x1 - x2) + abs(y1 - y2);
}

bool canReach(long long k, long long dist) {
    long long totalMoves = k * (k + 1) / 2;
    
    if (totalMoves < dist) return false;
    
    // Check parity
    long long excess = totalMoves - dist;
    return excess % 2 == 0;
}

int main() {
    int N;
    long long M, P, Q;
    cin >> N >> M >> P >> Q;
    
    vector<long long> distances;
    
    for (int i = 0; i < N; i++) {
        long long x, y;
        cin >> x >> y;
        distances.push_back(manhattanDist(x, y, P, Q));
    }
    
    // Check parity consistency
    int parity = distances[0] % 2;
    for (long long d : distances) {
        if (d % 2 != parity) {
            cout << -1 << endl;
            return 0;
        }
    }
    
    // Binary search on k
    long long maxDist = *max_element(distances.begin(), distances.end());
    
    long long left = 0, right = 2000000000LL;
    long long answer = -1;
    
    while (left <= right) {
        long long mid = (left + right) / 2;
        
        bool allCanReach = true;
        for (long long d : distances) {
            if (!canReach(mid, d)) {
                allCanReach = false;
                break;
            }
        }
        
        if (allCanReach) {
            answer = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    cout << answer << endl;
    
    return 0;
}
```

### 💡 Key Learning
**Parity Checks:**
- Some problems have hidden parity constraints
- If distances have different parity → impossible

---

## **Problem 11: Robot Sum (Garbage Collection)**

### 🎯 Intuition
This is a **DP cost optimization** problem.

Key: Waiting cost accumulates based on uncleaned garbage.

### 🧠 Approach

`dp[i]` = minimum cost to clean [0, i)

For each position j, decide: deploy new robot or let previous robot continue

### 📝 Step-by-Step Solution

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M;
    cin >> N >> M;
    
    vector<long long> G(N);
    for (int i = 0; i < N; i++) {
        cin >> G[i];
    }
    
    // dp[i] = min cost to clean [0, i)
    vector<long long> dp(N + 1, LLONG_MAX);
    dp[0] = 0;
    
    for (int j = 1; j <= N; j++) {
        // Option: deploy robot at position i, clean [i, j)
        for (int i = 0; i < j; i++) {
            if (dp[i] == LLONG_MAX) continue;
            
            // Cost of robot cleaning [i, j)
            long long robotCost = M; // deployment
            
            // Calculate waiting cost
            // When robot at position k, sum of garbage at [k+1, N) accumulates
            long long totalGarbage = 0;
            for (int k = 0; k < N; k++) {
                totalGarbage += G[k];
            }
            
            // Remove already cleaned
            for (int k = 0; k < i; k++) {
                totalGarbage -= G[k];
            }
            
            // Robot cleans [i, j), at each step waiting cost accumulates
            long long waitCost = 0;
            for (int k = i; k < j; k++) {
                waitCost += totalGarbage;
                totalGarbage -= G[k];
            }
            
            dp[j] = min(dp[j], dp[i] + robotCost + waitCost);
        }
    }
    
    cout << dp[N] << endl;
    
    return 0;
}
```

### 💡 Key Learning
**DP with accumulated costs:**
- Track total cost including "waiting" penalties
- Try all split points

---

## **Problem 12: Gift Certificates (Digit DP)**

### 🎯 Intuition
Count numbers from 1 to A with digit sum = S.

This is **Digit DP** - build number digit by digit.

### 🧠 Approach

**State:** `dp[pos][sum][tight][started]`
- pos: current position
- sum: current digit sum
- tight: are we still bounded by A?
- started: have we placed a non-zero digit?

### 📝 Step-by-Step Solution

```cpp
#include <bits/stdc++.h>
using namespace std;

const int MOD = 1e9 + 7;
string A;
int S;
long long dp[105][1005][2][2];
bool vis[105][1005][2][2];

long long solve(int pos, int sum, int tight, int started) {
    if (sum > S) return 0; // exceeded target sum
    
    if (pos == A.size()) {
        if (started && sum == S) return 1;
        return 0;
    }
    
    if (vis[pos][sum][tight][started]) {
        return dp[pos][sum][tight][started];
    }
    
    vis[pos][sum][tight][started] = true;
    
    int limit = tight ? (A[pos] - '0') : 9;
    long long result = 0;
    
    for (int digit = 0; digit <= limit; digit++) {
        int newTight = tight && (digit == limit) ? 1 : 0;
        int newStarted = started || (digit > 0) ? 1 : 0;
        int newSum = sum + digit;
        
        result = (result + solve(pos + 1, newSum, newTight, newStarted)) % MOD;
    }
    
    return dp[pos][sum][tight][started] = result;
}

int main() {
    cin >> A >> S;
    
    memset(vis, false, sizeof(vis));
    
    cout << solve(0, 0, 1, 0) << endl;
    
    return 0;
}
```

### 💡 Key Learning
**Digit DP Template:**
1. Build number left to right
2. Track if we're still bounded (tight)
3. Track if we've started (to avoid leading zeros)
4. Memoize state

---

## **Problem 13: 2-Array with Max D Score**

### 🎯 Intuition
Score_A(D) = #(elements ≤ D) + 2×#(elements > D)
           = |A| + #(elements > D)

Maximize: Score_A(D) - Score_B(D)
        = const + #(A[i] > D) - #(B[i] > D)

### 🧠 Approach

1. Collect all unique values from both arrays
2. Try each value as D
3. Count elements > D in each array
4. Track maximum difference

### 📝 Step-by-Step Solution

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    cin >> n;
    
    vector<int> A(n);
    for (int i = 0; i < n; i++) cin >> A[i];
    
    cin >> m;
    vector<int> B(m);
    for (int i = 0; i < m; i++) cin >> B[i];
    
    // Collect all possible D values
    set<int> candidates;
    candidates.insert(0);
    for (int x : A) candidates.insert(x);
    for (int x : B) candidates.insert(x);
    
    int maxDiff = INT_MIN;
    int bestD = 0;
    
    for (int D : candidates) {
        int scoreA = 0, scoreB = 0;
        
        for (int x : A) {
            if (x <= D) scoreA += 1;
            else scoreA += 2;
        }
        
        for (int x : B) {
            if (x <= D) scoreB += 1;
            else scoreB += 2;
        }
        
        int diff = scoreA - scoreB;
        if (diff > maxDiff) {
            maxDiff = diff;
            bestD = D;
        }
    }
    
    cout << bestD << endl;
    
    return 0;
}
```

### 💡 Key Learning
**Optimization by trying candidates:**
- Instead of trying all possible values
- Try only "critical" values where answer might change

---

## **Problem 14: Partition Array (Meet in the Middle)**

### 🎯 Intuition
Classic **Meet in the Middle** technique for subset problems.

Can't try all 2^(2N) subsets, but can split into two halves!

### 🧠 Approach

1. Split array into two halves
2. Generate all subset sums for each half
3. Combine using binary search

### 📝 Step-by-Step Solution

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cin >> n;
    n *= 2; // array size is 2N
    
    vector<int> nums(n);
    long long totalSum = 0;
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
        totalSum += nums[i];
    }
    
    int half = n / 2;
    
    // Generate all subsets of first half
    // Map: size → list of sums
    map<int, vector<long long>> left;
    
    for (int mask = 0; mask < (1 << half); mask++) {
        int count = __builtin_popcount(mask);
        long long sum = 0;
        
        for (int i = 0; i < half; i++) {
            if (mask & (1 << i)) {
                sum += nums[i];
            }
        }
        
        left[count].push_back(sum);
    }
    
    // Sort for binary search
    for (auto& [k, v] : left) {
        sort(v.begin(), v.end());
    }
    
    // Generate all subsets of second half
    long long minDiff = LLONG_MAX;
    
    for (int mask = 0; mask < (1 << half); mask++) {
        int count = __builtin_popcount(mask);
        long long sum = 0;
        
        for (int i = 0; i < half; i++) {
            if (mask & (1 << i)) {
                sum += nums[half + i];
            }
        }
        
        // Need total of half elements
        int needFromLeft = half - count;
        
        if (left.find(needFromLeft) == left.end()) continue;
        
        auto& candidates = left[needFromLeft];
        
        // We want leftSum + sum ≈ totalSum/2
        long long target = totalSum / 2 - sum;
        
        // Binary search for closest
        auto it = lower_bound(candidates.begin(), candidates.end(), target);
        
        // Check both neighbors
        if (it != candidates.end()) {
            long long total = *it + sum;
            long long diff = abs(2 * total - totalSum);
            minDiff = min(minDiff, diff);
        }
        
        if (it != candidates.begin()) {
            --it;
            long long total = *it + sum;
            long long diff = abs(2 * total - totalSum);
            minDiff = min(minDiff, diff);
        }
    }
    
    cout << minDiff << endl;
    
    return 0;
}
```

### 💡 Key Learning
**Meet in the Middle:**
- Reduces 2^N to 2 × 2^(N/2)
- Essential for subset problems with N ≤ 40

---

## **Problem 15: Traveling Salesman Problem (TSP)**

### 🎯 Intuition
Visit all cities exactly once, return to start, minimize distance.

Classic **Bitmask DP** problem!

### 🧠 Approach

**State:** `dp[mask][i]` = min cost to visit cities in 'mask', currently at city i

**Transition:** Try going to unvisited city j

### 📝 Step-by-Step Solution

```cpp
#include <bits/stdc++.h>
using namespace std;

const int INF = 1e9;

int main() {
    int N;
    cin >> N;
    
    vector<vector<int>> dist(N, vector<int>(N));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cin >> dist[i][j];
        }
    }
    
    // dp[mask][i] = min cost to visit cities in mask, ending at i
    vector<vector<int>> dp(1 << N, vector<int>(N, INF));
    
    // Start at city 0
    dp[1][0] = 0;
    
    // Try all masks
    for (int mask = 1; mask < (1 << N); mask++) {
        for (int i = 0; i < N; i++) {
            if (!(mask & (1 << i))) continue; // i not in mask
            if (dp[mask][i] == INF) continue;
            
            // Try going to city j
            for (int j = 0; j < N; j++) {
                if (mask & (1 << j)) continue; // already visited
                
                int newMask = mask | (1 << j);
                dp[newMask][j] = min(dp[newMask][j], dp[mask][i] + dist[i][j]);
            }
        }
    }
    
    // Return to start city 0
    int fullMask = (1 << N) - 1;
    int answer = INF;
    
    for (int i = 0; i < N; i++) {
        answer = min(answer, dp[fullMask][i] + dist[i][0]);
    }
    
    cout << answer << endl;
    
    return 0;
}
```

### 💡 Key Learning
**Bitmask DP Pattern:**
- State = (subset of items, current position)
- Use bits to represent subset
- Works for N ≤ 20

---

## 🎓 General Learning Tips

### DP Patterns to Master:
1. **Linear DP:** `dp[i]` = best answer ending at i
2. **Interval DP:** `dp[i][j]` = answer for range [i,j]
3. **Bitmask DP:** `dp[mask]` = answer for subset
4. **Digit DP:** Build numbers digit by digit

### When to Use Each:
- **BFS:** Shortest path in unweighted graph
- **Dijkstra:** Shortest path with weights
- **Binary Search:** "Can we achieve X?" is monotonic
- **Meet in Middle:** Subset problems, N ≤ 40
- **Greedy:** Exchange argument works

### Debugging Tips:
1. Start with small examples
2. Print intermediate states
3. Check base cases carefully
4. Verify state transitions

Good luck with your exam! 🚀

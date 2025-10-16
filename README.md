
---

# 5 — Model Monitor & Drift Detector (model_monitor.py)

**File:** `src/model_monitor.py`
```python
#!/usr/bin/env python3
"""
Model Monitor & Drift Detector - model_monitor.py

- Track incoming features and predictions
- Compute population statistics and detect distribution drift via KL-divergence (approx)
- Alert when drift exceeds threshold
"""

import math
import random
from collections import defaultdict, deque

class FeatureWindow:
    def __init__(self, maxlen=1000):
        self.values = deque(maxlen=maxlen)

    def push(self, v):
        self.values.append(float(v))

    def histogram(self, bins=20):
        if not self.values:
            return [0]*bins
        mn = min(self.values); mx = max(self.values)
        if mn == mx:
            arr = [0]*bins
            arr[0] = len(self.values)
            return arr
        width = (mx - mn) / bins
        hist = [0]*bins
        for v in self.values:
            idx = min(bins-1, int((v - mn) / width))
            hist[idx] += 1
        return [h / len(self.values) for h in hist]

def kl_div(p, q, eps=1e-9):
    s = 0.0
    for a,b in zip(p,q):
        s += a * math.log((a + eps) / (b + eps))
    return s

class Monitor:
    def __init__(self, feature_names):
        self.refs = {f: FeatureWindow(maxlen=2000) for f in feature_names}
        self.live = {f: FeatureWindow(maxlen=500) for f in feature_names}
        self.threshold = 0.5

    def ingest(self, features: dict):
        for k, v in features.items():
            if k in self.refs:
                self.live[k].push(v)

    def update_reference(self):
        # snapshot live into reference (periodic)
        for k in self.refs:
            self.refs[k].values = deque(list(self.live[k].values), maxlen=self.refs[k].values.maxlen)

    def check_drift(self):
        drifts = {}
        for k in self.refs:
            p = self.refs[k].histogram()
            q = self.live[k].histogram()
            d = kl_div(p, q)
            drifts[k] = d
            if d > self.threshold:
                print(f"[DRIFT] feature={k} KL={d:.3f}")
        return drifts

def demo():
    mon = Monitor(["age", "income", "score"])
    # bootstrap reference with normal data
    for _ in range(2000):
        mon.refs["age"].push(random.gauss(40,10))
        mon.refs["income"].push(random.gauss(50_000, 10_000))
        mon.refs["score"].push(random.gauss(0.6, 0.1))
    # simulate streaming: no drift first
    for _ in range(300):
        mon.ingest({"age": random.gauss(40,10), "income": random.gauss(50_000, 10_000), "score": random.gauss(0.6,0.1)})
    mon.check_drift()
    # introduce drift on income
    for _ in range(300):
        mon.ingest({"age": random.gauss(40,10), "income": random.gauss(70_000, 10_000), "score": random.gauss(0.6,0.1)})
    mon.check_drift()

if __name__ == "__main__":
    demo()

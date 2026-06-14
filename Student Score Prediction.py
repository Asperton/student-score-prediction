"""
Student Score Prediction — small Python OLS demo

This file replaces the original React/JSX content with a standalone Python script
that trains a linear regression (Ordinary Least Squares) on the included sample
data and prints coefficients, per-student predictions, and overall MAE.
"""

from typing import List, Tuple


SAMPLE_DATA = [
    {"id": 1, "name": "Priya S.", "studyHours": 6, "attendance": 90, "prevScore": 78, "sleep": 7, "score": 82},
    {"id": 2, "name": "Rahul M.", "studyHours": 3, "attendance": 65, "prevScore": 55, "sleep": 5, "score": 58},
    {"id": 3, "name": "Anjali K.", "studyHours": 8, "attendance": 95, "prevScore": 88, "sleep": 8, "score": 91},
    {"id": 4, "name": "Dev P.", "studyHours": 5, "attendance": 80, "prevScore": 70, "sleep": 7, "score": 74},
    {"id": 5, "name": "Sneha R.", "studyHours": 2, "attendance": 55, "prevScore": 45, "sleep": 4, "score": 48},
    {"id": 6, "name": "Arjun B.", "studyHours": 7, "attendance": 88, "prevScore": 82, "sleep": 8, "score": 86},
    {"id": 7, "name": "Meera T.", "studyHours": 4, "attendance": 72, "prevScore": 62, "sleep": 6, "score": 65},
    {"id": 8, "name": "Kiran J.", "studyHours": 9, "attendance": 98, "prevScore": 92, "sleep": 8, "score": 95},
    {"id": 9, "name": "Pooja V.", "studyHours": 1, "attendance": 50, "prevScore": 40, "sleep": 5, "score": 42},
    {"id": 10, "name": "Rohit G.", "studyHours": 6, "attendance": 85, "prevScore": 75, "sleep": 7, "score": 79},
    {"id": 11, "name": "Nisha A.", "studyHours": 5, "attendance": 78, "prevScore": 68, "sleep": 6, "score": 71},
    {"id": 12, "name": "Aditya C.", "studyHours": 7, "attendance": 92, "prevScore": 85, "sleep": 8, "score": 88},
    {"id": 13, "name": "Kavya L.", "studyHours": 3, "attendance": 60, "prevScore": 52, "sleep": 5, "score": 55},
    {"id": 14, "name": "Vikram N.", "studyHours": 8, "attendance": 94, "prevScore": 90, "sleep": 7, "score": 93},
    {"id": 15, "name": "Ishaan S.", "studyHours": 4, "attendance": 70, "prevScore": 60, "sleep": 6, "score": 63},
]


def transpose(M: List[List[float]]) -> List[List[float]]:
    return [list(col) for col in zip(*M)]


def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]


def matvec(A: List[List[float]], v: List[float]) -> List[float]:
    return [sum(a * b for a, b in zip(row, v)) for row in A]


def solve_linear(A: List[List[float]], b: List[float]) -> List[float]:
    # Gaussian elimination with partial pivoting
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        # pivot
        max_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[max_row] = M[max_row], M[col]
        if abs(M[col][col]) < 1e-12:
            continue
        # normalize pivot row
        pivot = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pivot
        # eliminate
        for i in range(n):
            if i == col:
                continue
            factor = M[i][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                M[i][j] -= factor * M[col][j]
    return [M[i][n] for i in range(n)]


def train_model(data: List[dict]) -> List[float]:
    # X: rows of [1, studyHours, attendance, prevScore, sleep]
    X = [[1.0, d["studyHours"], d["attendance"], d["prevScore"], d["sleep"]] for d in data]
    y = [d["score"] for d in data]
    Xt = transpose(X)
    XtX = [[sum(a * b for a, b in zip(col_i, col_j)) for col_j in Xt] for col_i in Xt]
    XtY = [sum(a * b for a, b in zip(col, y)) for col in Xt]
    coeffs = solve_linear(XtX, XtY)
    return coeffs


def predict(coeffs: List[float], features: dict) -> float:
    x = [1.0, features["studyHours"], features["attendance"], features["prevScore"], features["sleep"]]
    val = sum(c * xi for c, xi in zip(coeffs, x))
    return max(0.0, min(100.0, val))


def mean_absolute_error(trues: List[float], preds: List[float]) -> float:
    return sum(abs(t - p) for t, p in zip(trues, preds)) / len(trues)


if __name__ == "__main__":
    coeffs = train_model(SAMPLE_DATA)
    print("Learned coefficients (bias, study, attendance, prevScore, sleep):")
    print([round(c, 4) for c in coeffs])
    print()

    preds = [predict(coeffs, s) for s in SAMPLE_DATA]
    actuals = [s["score"] for s in SAMPLE_DATA]
    mae = mean_absolute_error(actuals, preds)
    print(f"Mean Absolute Error (MAE): {mae:.3f} points")
    print()

    # Print per-student prediction table
    print(f"{'Name':<16}  {'Study':>5} {'Attend':>6} {'Prev':>5} {'Sleep':>5}  {'Actual':>6} {'Pred':>6} {'Error':>6}")
    print("-" * 72)
    for s, p in zip(SAMPLE_DATA, preds):
        err = abs(p - s["score"]) 
        print(f"{s['name']:<16}  {s['studyHours']:5.1f} {s['attendance']:6.1f} {s['prevScore']:5.1f} {s['sleep']:5.1f}  {s['score']:6.1f} {p:6.1f} {err:6.2f}")

    # Example: predict for a custom student
    sample_features = {"studyHours": 5, "attendance": 80, "prevScore": 70, "sleep": 7}
    pred_sample = predict(coeffs, sample_features)
    print()
    print(f"Example prediction for features {sample_features}: {pred_sample:.1f}")

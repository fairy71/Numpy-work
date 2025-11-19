import numpy as np
arr = np.random.randint(1, 51, size=(6, 6))
print("Original:\n", arr)

arr[arr < 10] = 0
arr[arr > 40] = 100
print("\nModified:\n", arr)

center = arr[1:4, 1:4]
print("\nCenter 3x3:\n", center)

print("\nRow means:", arr.mean(axis=1))
print("Col means:", arr.mean(axis=0))

a = np.array([1, 2, 3, 4])
b = np.array([[10], [20], [30]])

print("\na + b =\n", a + b)
print("a * b =\n", a * b)
print("b / sum(a) =\n", b / a.sum())

arr = np.random.randint(0, 101, size=100)

div_3_5 = arr[(arr % 3 == 0) & (arr % 5 == 0)]
print("\nDivisible by 3 & 5:\n", div_3_5)

arr_even_neg = arr.copy()
arr_even_neg[arr_even_neg % 2 == 0] *= -1
print("\nEven numbers negative:\n", arr_even_neg)

count_range = np.sum((arr >= 25) & (arr <= 75))
print("\nCount between 25–75:", count_range)

A = np.random.rand(3, 4)
B = np.random.rand(4, 3)

C = A @ B
print("\nA @ B =\n", C)

Ct = C.T
print("\nTranspose:\n", Ct)

det = np.linalg.det(Ct)
print("\nDeterminant:", det)

arr = np.random.rand(30) * 100
sorted_arr = np.sort(arr)
print("\nSorted:\n", sorted_arr)

largest5 = np.partition(arr, -5)[-5:]
print("\n5 largest:\n", largest5)

closest = arr[np.abs(arr - arr.mean()).argmin()]
print("\nClosest to mean:", closest)

unique_vals = np.unique(arr)
print("\nUnique values count:", len(unique_vals))

arr = np.arange(1, 101).reshape(10, 10)
col_view = arr[:, 0]
print("\nView of first column:", col_view)

col_view[:] = -999
print("\nAfter modifying view, original array changed:\n", arr)

col_copy = arr[:, 0].copy()
col_copy[:] = 777
print("\nCopy changed (original unaffected):\n", arr)

data = np.random.rand(1000, 5)

mean = data.mean(axis=0)
median = np.median(data, axis=0)
std = data.std(axis=0)

print("\nMeans:", mean)
print("Medians:", median)
print("STDs:", std)

z_norm = (data - mean) / std
print("\nZ-normalized sample row:\n", z_norm[0])

min_vals = data.min(axis=0)
max_vals = data.max(axis=0)
minmax_norm = (data - min_vals) / (max_vals - min_vals)
print("\nMinmax-normalized sample row:\n", minmax_norm[0])

flips = np.random.randint(0, 2, 10000)
heads = flips.sum()

p_5200 = np.mean([
    np.random.randint(0, 2, 10000).sum() == 5200
    for _ in range(500)
])
print("\nProbability of 5200 heads:", p_5200)

diff = np.diff(np.where(np.concatenate(([flips[0]],
                                        flips[:-1] != flips[1:], [True])))[0])
longest_streak = diff.max()
print("Longest streak of identical outcomes:", longest_streak)

experiments = [np.random.randint(0, 2, 10000).sum() for _ in range(100)]
print("Sample experiment head counts:\n", experiments[:10])

p = np.poly1d([4, -3, 2, -1])
print("\np(5) =", p(5))

print("Derivative:", p.deriv())
print("Integral:", p.integ())
print("Roots:", p.r)

q = np.poly1d([1, -2])
print("p*q =", p * q)

A = np.random.rand(5, 5)

eigvals, eigvecs = np.linalg.eig(A)
print("\nEigenvalues:\n", eigvals)

invA = np.linalg.inv(A)
print("\nA * A^-1:\n", A @ invA)

Q, R = np.linalg.qr(A)
print("\nQR decomposition:\nQ=\n", Q, "\nR=\n", R)

U, S, Vt = np.linalg.svd(A)
print("\nSVD:\nU=\n", U, "\nS=", S, "\nVt=\n", Vt)

arr = np.arange(100).reshape(10, 10)

even_rows_odd_cols = arr[::2, 1::2]
print("\nEven rows, odd cols:\n", even_rows_odd_cols)

border = np.concatenate([arr[0], arr[-1], arr[1:-1, 0], arr[1:-1, -1]])
print("\nBorder elements:\n", border)

diag = np.diag(arr)
print("\nDiagonal:\n", diag)

img = np.random.randint(0, 256, size=(256, 256))

flip_h = img[:, ::-1]
flip_v = img[::-1, :]
rot90 = np.rot90(img, 1)
rot180 = np.rot90(img, 2)
rot270 = np.rot90(img, 3)

kernel = np.ones((3, 3)) / 9
blur = np.zeros_like(img, float)

for i in range(1, 255):
    for j in range(1, 255):
        blur[i, j] = (img[i-1:i+2, j-1:j+2] * kernel).sum()

print("\nBlur sample:", blur[100, 100])

arr = np.random.rand(100, 100)

np.save("matrix.npy", arr)
np.savetxt("matrix.csv", arr, delimiter=",")

load_npy = np.load("matrix.npy")
load_csv = np.loadtxt("matrix.csv", delimiter=",")

print("\nFiles equal check:", np.allclose(load_npy, load_csv))

arr = np.random.randint(1, 101, size=1_000_000)

unique, counts = np.unique(arr, return_counts=True)
values_exact_10 = unique[counts == 10]
print("\nValues appearing exactly 10 times:\n", values_exact_10)

most_freq = unique[counts.argmax()]
print("\nMost frequent value:", most_freq)

cumsum_gt_50 = np.cumsum(arr * (arr > 50))
print("\nCumsum of values > 50 sample:", cumsum_gt_50[:10])

def normalize_rows(M):
    row_sums = M.sum(axis=1)
    if np.any(row_sums == 0):
        raise ValueError("A row sums to zero!")
    return M / row_sums[:, None]

mat = np.random.rand(4, 5)
print("\nRow-normalized matrix:\n", normalize_rows(mat))

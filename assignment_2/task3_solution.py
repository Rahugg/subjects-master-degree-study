import numpy as np

def generate_temps(seed=42):
    rng = np.random.default_rng(seed)
    T = rng.integers(-10, 36, size=(7, 24, 3))
    return T

def indexing_examples(T):
    city2 = T[:, :, 1]
    day1 = T[0, :, :]
    val = int(T[2, 12, 1])
    return city2, day1, val

def operations(T):
    avg_city = T.mean(axis=(0,1))
    max_per_day = T.max(axis=(1,2))
    hour_lowest = int(T.mean(axis=(0,2)).argmin())
    return avg_city, max_per_day, hour_lowest

def transformations(T):
    M = T.reshape(7, -1)
    daily_profiles = T.mean(axis=1)
    city_means = daily_profiles.mean(axis=0)
    centered = daily_profiles - city_means
    corr = np.corrcoef(centered.reshape(7, -1).T)[0:3,0:3]
    return M, corr

def demo():
    T = generate_temps()
    city2, day1, val = indexing_examples(T)
    avg_city, max_per_day, hour_lowest = operations(T)
    M, corr = transformations(T)
    print(city2.shape, day1.shape, val)
    print("Avg weekly per city:", avg_city)
    print("Max per day:", max_per_day)
    print("Hour with lowest avg temp:", hour_lowest)
    print("Reshaped:", M.shape)
    print("Correlation between cities:\n", corr)

if __name__ == "__main__":
    demo()

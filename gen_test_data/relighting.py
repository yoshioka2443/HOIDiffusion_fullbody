import math
import torch

def deringing(coeffs, window):
    deringed_coeffs = torch.zeros_like(coeffs)
    deringed_coeffs[:, 0] += coeffs[:, 0]
    deringed_coeffs[:, 1:1 + 3] += \
        coeffs[:, 1:1 + 3] * math.pow(math.sin(math.pi * 1.0 / window) / (math.pi * 1.0 / window), 4.0)
    deringed_coeffs[:, 4:4 + 5] += \
        coeffs[:, 4:4 + 5] * math.pow(math.sin(math.pi * 2.0 / window) / (math.pi * 2.0 / window), 4.0)
    deringed_coeffs[:, 9:9 + 7] += \
        coeffs[:, 9:9 + 7] * math.pow(math.sin(math.pi * 3.0 / window) / (math.pi * 3.0 / window), 4.0)
    return deringed_coeffs
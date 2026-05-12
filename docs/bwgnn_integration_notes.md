# BWGNN Integration Notes

## External Source Review

**Reference**: `external/Rethinking-Anomaly-Detection/BWGNN.py`

### Core Modules

1. **PolyConv**: Polynomial graph convolution
   - Uses theta coefficients from Beta wavelet
   - Implements `feat * D^{-1/2} A D^{-1/2}` propagation
   - Each polynomial captures different spectral band

2. **calculate_theta2(d)**: Beta wavelet coefficient computation
   - Uses `sympy` and `scipy.special.beta`
   - Returns list of polynomial coefficients for order d

3. **BWGNN**: Main model
   - Linear → ReLU → Linear → ReLU
   - Multiple PolyConv (one per theta)
   - Concatenate → Linear → ReLU → Linear

### Key Ideas

- Beta wavelets provide band-pass filtering at different frequencies
- Each PolyConv with different theta captures different spectral characteristics
- High theta (close to d) → high-frequency response
- Low theta (close to 0) → low-frequency response

## Our Implementation

### Simplifications

1. **PyG instead of DGL**: Use PyG's message passing
2. **No sympy dependency**: Pre-compute theta coefficients
3. **Expose spectral response**: Add `high_freq_response` to extras

### Architecture

```
Input → Linear → ReLU → Linear → ReLU
  ↓
PolyConv[0] (low-freq) → h0
PolyConv[1] (mid-freq) → h1
PolyConv[2] (high-freq) → h2
  ↓
Concat[h0, h1, h2] → Linear → ReLU → Linear → logits
  ↓
extras = {"high_freq_response": h2.norm(dim=-1)}
```

### Theta Coefficients (d=2)

For order d=2, we have 3 Beta wavelets:
- theta[0]: low-frequency (x/2)^0 * (1-x/2)^2
- theta[1]: mid-frequency (x/2)^1 * (1-x/2)^1
- theta[2]: high-frequency (x/2)^2 * (1-x/2)^0

Pre-computed coefficients:
```python
theta_0 = [1.0, -1.0, 0.25]  # low-freq
theta_1 = [0.0, 1.0, -0.5]   # mid-freq
theta_2 = [0.0, 0.0, 0.25]   # high-freq
```

### Extras

- `high_freq_response`: L2 norm of high-frequency PolyConv output
- `band_energy`: Optional, energy per frequency band

### Differences from Original

| Aspect | Original | Ours |
|--------|----------|------|
| Framework | DGL | PyG |
| Graph input | DGL graph | edge_index |
| Theta computation | sympy + scipy | Pre-computed |
| Spectral response | Not exposed | In extras |
| Output | logits only | BaseModelOutput |

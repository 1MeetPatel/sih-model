# Geospatial Pipeline & Geometric Road Width Estimation

## 1. Ground Sample Distance (GSD) & CRS Handling

### Metric Projected CRS (e.g. UTM Zones / EPSG:32643)
$$\text{GSD} = \frac{|\Delta x| + |\Delta y|}{2} \quad (\text{meters/pixel})$$

### Angular Geographic CRS (e.g. WGS84 / EPSG:4326)
Pixels represent angular degrees $(\Delta^\circ)$. At latitude $\phi$:
$$\Delta x_m = \Delta^\circ_{\text{lon}} \cdot 111,139 \cdot \cos(\phi), \quad \Delta y_m = \Delta^\circ_{\text{lat}} \cdot 111,139$$
$$\text{GSD} = \frac{\Delta x_m + \Delta y_m}{2}$$

---

## 2. Geometric Road Width Estimation via Euclidean Distance Transform

Rather than assuming all segmented pixels qualify as full roads, the system measures physical width continuously along the road centerline:

1. **Euclidean Distance Transform (EDT)**:
   For every pixel $p$ inside the binary road mask $M$:
   $$D(p) = \min_{q \notin M} \|p - q\|_2 \quad (\text{distance to closest background pixel})$$

2. **Centerline Radius Sampling**:
   Along each centerline skeleton point $c \in S$:
   $$R(c) = D(c) \quad (\text{pixel radius})$$
   $$\text{Physical Width}(c) = 2 \cdot R(c) \cdot \text{GSD} \quad (\text{meters})$$

3. **Segment Aggregation**:
   Each connected road segment $k$ has:
   - $\text{Mean Width}_k = \frac{1}{|S_k|} \sum_{c \in S_k} \text{Width}(c)$
   - $\text{Median Width}_k = \text{median}_{c \in S_k}(\text{Width}(c))$
   - $\text{Length}_k = |S_k| \cdot \text{GSD}$

4. **$\ge 6.1\text{m}$ (20-Foot) Filter**:
   A road segment qualifies if and only if:
   $$\text{Mean Width}_k \ge 6.096\text{ m} \quad \text{and} \quad \text{Length}_k \ge 15.0\text{ m}$$

---

## 3. Temporal Road Change Detection

```
BEFORE Road Network (LineStrings)      AFTER Road Network (LineStrings)
              │                                      │
              └───────────────┬──────────────────────┘
                              ▼
                Spatial Buffer & CRS Alignment
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
   NEW ROADS: AFTER - Buffer(BEFORE)    REMOVED ROADS: BEFORE - Buffer(AFTER)
```
- **New Roads**: Generated when a road segment in AFTER has no spatial overlap with the buffered BEFORE network ($\text{buffer} = 3.5\text{ m}$).
- **Removed Roads**: Generated when a road segment in BEFORE is absent in the AFTER network.
- **Attributes**: `change_type`, `length_m`, `width_m`, `confidence`, and georeferenced `LineString` geometries.

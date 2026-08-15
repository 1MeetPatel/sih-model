
import os, math, argparse
import numpy as np
import cv2
import rasterio
import rasterio.features
import geopandas as gpd
from shapely.geometry import shape, LineString
from skimage.morphology import skeletonize
from PIL import Image

def calculate_and_verify_gsd(raster_path, max_allowed_gsd=0.80):
    with rasterio.open(raster_path) as src:
        tf = src.transform
        crs = src.crs
        res_x, res_y = abs(tf.a), abs(tf.e)
        if crs.is_geographic:
            bounds = src.bounds
            mid_lat = (bounds.bottom + bounds.top) / 2.0
            gsd = ((res_x * 111412.84 * math.cos(math.radians(mid_lat))) +
                   (res_y * (111132.954 - 559.822 * math.cos(2 * math.radians(mid_lat))))) / 2.0
        else:
            gsd = (res_x + res_y) / 2.0
        print('[*] Raster: ' + raster_path)
        print('[*] CRS: ' + crs.to_string())
        print('[*] GSD: %.4f m/pixel' % gsd)
        if gsd > max_allowed_gsd:
            raise ValueError('Insufficient GSD (%.3f m/px). Max: %.2f m/px.' % (gsd, max_allowed_gsd))
        return gsd

def predict_road_probability(raster_path):
    with rasterio.open(raster_path) as src:
        r = src.read(1).astype(np.float32)
        g = src.read(2).astype(np.float32)
        b = src.read(3).astype(np.float32)
        transform = src.transform
        crs = src.crs
        profile = src.profile
    def to_uint8(band):
        mn, mx = band.min(), band.max()
        return ((band - mn) / (mx - mn + 1e-6) * 255).astype(np.uint8)
    rgb = np.dstack([to_uint8(r), to_uint8(g), to_uint8(b)])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[:,:,1], hsv[:,:,2]
    road_hsv = (s < 60) & (v > 40) & (v < 160)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    road_lab = (np.abs(a_ch.astype(np.int16) - 128) < 12) & \
               (np.abs(b_ch.astype(np.int16) - 128) < 12) & \
               (l_ch > 50) & (l_ch < 180)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 90)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edge_dil = cv2.dilate(edges, kernel, iterations=1)
    combined = (road_hsv.astype(np.float32) * 100 +
                road_lab.astype(np.float32) * 100 +
                (edge_dil > 0).astype(np.float32) * 55)
    mx = combined.max()
    prob_map = np.clip(combined / mx if mx > 0 else combined, 0, 1).astype(np.float32)
    prob_map = cv2.GaussianBlur(prob_map, (9, 9), 0)
    print('[*] Probability map: %dx%d px' % (prob_map.shape[1], prob_map.shape[0]))
    return prob_map, transform, crs, profile

def filter_roads_by_width(prob_map, gsd, threshold=0.35, min_w=4.5, max_w=7.5):
    binary = (prob_map >= threshold).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    width_m = 2.0 * dist * gsd
    skeleton = skeletonize(binary > 0)
    target_skel = skeleton & (width_m >= min_w) & (width_m <= max_w)
    corridor = np.zeros_like(binary, dtype=np.uint8)
    ys, xs = np.where(target_skel)
    for y, x in zip(ys, xs):
        rad = int(round(dist[y, x]))
        if rad > 0:
            cv2.circle(corridor, (x, y), rad, 1, thickness=-1)
    corridor = np.bitwise_and(corridor, binary)
    n_px = int(corridor.sum())
    print('[*] 20-ft road pixels: %d  (~%.1f m^2)' % (n_px, n_px * gsd * gsd))
    return target_skel.astype(np.uint8), corridor.astype(np.uint8), width_m

def vectorize_and_export(skeleton_mask, corridor_mask, width_map, transform, crs, output_path):
    shapes_gen = rasterio.features.shapes(corridor_mask, mask=corridor_mask > 0, transform=transform)
    polygons = [shape(g) for g, v in shapes_gen if v == 1]
    contours, _ = cv2.findContours(skeleton_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    records = []
    for cnt in contours:
        if len(cnt) < 2:
            continue
        pts = cnt.squeeze(axis=1)
        avg_w = float(np.mean([width_map[p[1], p[0]] for p in pts]))
        geo_pts = [transform * (p[0] + 0.5, p[1] + 0.5) for p in pts]
        line = LineString(geo_pts)
        if line.length > 0:
            records.append(dict(geometry=line, feature_type='Road_Centerline_20ft',
                                avg_width_m=round(avg_w,2), avg_width_ft=round(avg_w*3.28084,2)))
    for poly in polygons:
        records.append(dict(geometry=poly, feature_type='Road_Corridor_20ft',
                            avg_width_m=None, avg_width_ft=None))
    if not records:
        print('[!] No 20-ft road features to export.')
        return gpd.GeoDataFrame(crs=crs)
    gdf = gpd.GeoDataFrame(records, crs=crs)
    gdf.to_file(output_path, driver='GeoJSON')
    print('[+] GeoJSON saved -> %s (%d features)' % (output_path, len(records)))
    return gdf

def generate_overlay(raster_path, corridor_mask, output_path, color=(0,255,255), opacity=0.70):
    with rasterio.open(raster_path) as src:
        r, g, b = src.read(1), src.read(2), src.read(3)
    def norm(band):
        mn, mx = band.min(), band.max()
        return ((band - mn) / (mx - mn + 1e-6) * 255).astype(np.uint8)
    base = np.dstack([norm(r), norm(g), norm(b)])
    overlay = base.copy()
    mask = corridor_mask > 0
    for c, val in enumerate(color):
        overlay[mask, c] = np.clip(base[mask, c] * (1 - opacity) + val * opacity, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(output_path, format='PNG')
    print('[+] Overlay saved  -> ' + output_path)

def detect_changes(t1_path, t2_path, gsd, output_path):
    print('[*] Change detection (T1 vs T2)...')
    prob1, tf1, crs1, _ = predict_road_probability(t1_path)
    prob2, tf2, crs2, _ = predict_road_probability(t2_path)
    _, mask1, _ = filter_roads_by_width(prob1, gsd)
    _, mask2, _ = filter_roads_by_width(prob2, gsd)
    if mask1.shape != mask2.shape:
        mask1 = cv2.resize(mask1, (mask2.shape[1], mask2.shape[0]), interpolation=cv2.INTER_NEAREST)
    new_roads = np.bitwise_and(mask2 == 1, mask1 == 0).astype(np.uint8)
    removed   = np.bitwise_and(mask1 == 1, mask2 == 0).astype(np.uint8)
    records = []
    for arr, label in [(new_roads, 'New_Road_Constructed'), (removed, 'Road_Decommissioned')]:
        for geom, v in rasterio.features.shapes(arr, mask=arr > 0, transform=tf2):
            records.append(dict(geometry=shape(geom), change_type=label))
    if records:
        gpd.GeoDataFrame(records, crs=crs2).to_file(output_path, driver='GeoJSON')
        print('[+] Changes saved  -> %s (%d features)' % (output_path, len(records)))
    else:
        print('[*] No significant changes detected.')

def main():
    parser = argparse.ArgumentParser(description='20-Foot Road Detection Pipeline')
    parser.add_argument('--input',          required=True)
    parser.add_argument('--output-geojson', default='roads_20ft_extracted.geojson')
    parser.add_argument('--output-overlay', default='roads_20ft_overlay.png')
    parser.add_argument('--temporal-t2',    default=None)
    args = parser.parse_args()
    print('=' * 60)
    print('  20-Foot Road Detection Pipeline')
    print('=' * 60)
    gsd = calculate_and_verify_gsd(args.input)
    prob_map, transform, crs, _ = predict_road_probability(args.input)
    skeleton, corridor, width_map = filter_roads_by_width(prob_map, gsd)
    vectorize_and_export(skeleton, corridor, width_map, transform, crs, args.output_geojson)
    generate_overlay(args.input, corridor, args.output_overlay)
    if args.temporal_t2:
        detect_changes(args.input, args.temporal_t2, gsd, 'road_changes.geojson')
    print('=' * 60)
    print('  [DONE] Pipeline completed successfully.')
    print('=' * 60)

if __name__ == '__main__':
    main()

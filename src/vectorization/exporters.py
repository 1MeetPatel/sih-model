"""
GIS Format Exporters (GeoPackage, GeoJSON, Shapefile)
"""

import os
import geopandas as gpd
from ..utils.logging import get_logger

logger = get_logger("GISExporter")


class GISExporter:
    """
    Exports GeoDataFrames to standard GIS interchange formats.
    """

    @staticmethod
    def export_all(
        gdf: gpd.GeoDataFrame,
        output_dir: str,
        base_name: str = "roads",
        export_gpkg: bool = True,
        export_geojson: bool = True,
        export_shp: bool = False
    ):
        os.makedirs(output_dir, exist_ok=True)

        if export_gpkg:
            gpkg_path = os.path.join(output_dir, f"{base_name}.gpkg")
            try:
                gdf.to_file(gpkg_path, layer=base_name, driver="GPKG")
                logger.info(f"[+] Exported GeoPackage -> {gpkg_path}")
            except Exception as e:
                logger.warning(f"GeoPackage export warning: {e}")

        if export_geojson:
            geojson_path = os.path.join(output_dir, f"{base_name}.geojson")
            try:
                # If projected CRS, convert to WGS84 EPSG:4326 for standard GeoJSON compatibility
                if gdf.crs and gdf.crs.is_projected:
                    gdf_wgs84 = gdf.to_crs(epsg=4326)
                    gdf_wgs84.to_file(geojson_path, driver="GeoJSON")
                else:
                    gdf.to_file(geojson_path, driver="GeoJSON")
                logger.info(f"[+] Exported GeoJSON -> {geojson_path}")
            except Exception as e:
                logger.warning(f"GeoJSON export warning: {e}")

        if export_shp:
            shp_path = os.path.join(output_dir, f"{base_name}.shp")
            try:
                gdf.to_file(shp_path, driver="ESRI Shapefile")
                logger.info(f"[+] Exported ESRI Shapefile -> {shp_path}")
            except Exception as e:
                logger.warning(f"Shapefile export warning: {e}")

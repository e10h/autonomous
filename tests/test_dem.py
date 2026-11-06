import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin
from autonomous_state_machine.dem import prepare_geotiff
from autonomous_state_machine.terrain import TerrainGrid


def write_dem(path, z, crs="EPSG:32612", resolution=0.2):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=z.shape[0],
        width=z.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(500000, 4250000, resolution, resolution),
        nodata=-9999,
    ) as ds:
        ds.write(z.astype("float32"), 1)


def test_tiled_gradients_match_whole_dem_at_seams(tmp_path):
    rr, cc = np.mgrid[:32, :32]
    z = 0.1 * cc * 0.2 - 0.2 * rr * 0.2
    source = tmp_path / "plane.tif"
    write_dem(source, z)
    prepare_geotiff(source, tmp_path / "small", vertical_scale=1, native_tile_size=8)
    prepare_geotiff(source, tmp_path / "large", vertical_scale=1, native_tile_size=64)
    a, b = TerrainGrid.open(tmp_path / "small"), TerrainGrid.open(tmp_path / "large")
    np.testing.assert_array_equal(a.blocked, b.blocked)
    for x, y in zip(a.gradients, b.gradients):
        np.testing.assert_allclose(x, y, equal_nan=True)
    np.testing.assert_allclose(a.gx_min[1:-1, 1:-1], 0.1, atol=1e-6)
    np.testing.assert_allclose(a.gy_min[1:-1, 1:-1], 0.2, atol=1e-6)
    assert a.dy < 0 and isinstance(a.blocked, np.memmap)


def test_coarse_prior_preserves_single_native_pixel_cliff_and_nodata(tmp_path):
    z = np.zeros((32, 32), dtype="float32")
    z[12, 12] = 3
    z[20, 20] = -9999
    source = tmp_path / "hazards.tif"
    write_dem(source, z)
    prepare_geotiff(
        source, tmp_path / "map", resolution_m=0.8, vertical_scale=1, native_tile_size=8
    )
    terrain = TerrainGrid.open(tmp_path / "map")
    assert terrain.blocked[3, 3] and terrain.blocked[5, 5]
    assert terrain.metadata["native_resolution_m"] == 0.2
    assert terrain.metadata["resolution_m"] == 0.8


def test_no_upsampling_or_geographic_degree_slopes(tmp_path):
    source = tmp_path / "map.tif"
    write_dem(source, np.zeros((8, 8)))
    with pytest.raises(ValueError, match="finer"):
        prepare_geotiff(source, tmp_path / "fine", 0.1)
    assert not (tmp_path / "fine").exists()
    write_dem(source, np.zeros((8, 8)), crs="EPSG:4326")
    with pytest.raises(ValueError, match="projected"):
        prepare_geotiff(source, tmp_path / "degrees")


def test_crop_preserves_metric_location_and_source_identity(tmp_path):
    source = tmp_path / "map.tif"
    write_dem(source, np.zeros((32, 32)))
    meta = prepare_geotiff(
        source, tmp_path / "crop", bounds=(500001, 4249996, 500004, 4249999)
    )
    terrain = TerrainGrid.open(tmp_path / "crop")
    assert terrain.x0 == pytest.approx(500001.1)
    assert terrain.y0 == pytest.approx(4249998.9)
    assert len(meta["source_sha256"]) == 64
    with pytest.raises(FileExistsError):
        prepare_geotiff(source, tmp_path / "crop")


def test_feet_based_crs_requires_explicit_reprojection(tmp_path):
    source = tmp_path / "feet.tif"
    write_dem(source, np.zeros((8, 8)), crs="EPSG:2229")
    with pytest.raises(ValueError, match="metres"):
        prepare_geotiff(source, tmp_path / "map")


def test_multiband_image_is_not_treated_as_elevation(tmp_path):
    source = tmp_path / 'image.tif'
    with rasterio.open(source, 'w', driver='GTiff', height=8, width=8, count=3,
                       dtype='float32', crs='EPSG:32612', transform=from_origin(500000,4250000,1,1)) as ds:
        ds.write(np.zeros((3,8,8),dtype='float32'))
    with pytest.raises(ValueError, match='single-band'):
        prepare_geotiff(source, tmp_path / 'map')

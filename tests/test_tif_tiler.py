"""tif_tiler 的單元測試：切片原點覆蓋完整性、空白偵測、view 切片。"""
import numpy as np
import pytest

from app_utils.tool import tif_tiler


class TestComputeAxisOrigins:
    def test_small_image_single_tile(self):
        assert tif_tiler.compute_axis_origins(800, 1600, 320) == [0]

    def test_exact_tile(self):
        assert tif_tiler.compute_axis_origins(1600, 1600, 320) == [0]

    @pytest.mark.parametrize("size", [1601, 3199, 3200, 4000, 20000])
    @pytest.mark.parametrize("overlap", [0, 320])
    def test_full_coverage(self, size, overlap):
        tile = 1600
        origins = tif_tiler.compute_axis_origins(size, tile, overlap)
        # 嚴格遞增、皆在合法範圍
        assert origins == sorted(set(origins))
        assert origins[0] == 0
        assert all(0 <= o <= size - tile for o in origins)
        # 完整覆蓋 [0, size)
        covered = np.zeros(size, dtype=bool)
        for o in origins:
            covered[o : o + tile] = True
        assert covered.all()
        # 相鄰切片至少保有要求的重疊
        for a, b in zip(origins, origins[1:]):
            assert a + tile - b >= overlap

    def test_invalid_overlap(self):
        with pytest.raises(ValueError):
            tif_tiler.compute_axis_origins(3200, 1600, 1600)

    def test_grid_row_major(self):
        grid = tif_tiler.compute_tile_origins(3200, 1601, 1600, 320)
        xs = tif_tiler.compute_axis_origins(3200, 1600, 320)
        ys = tif_tiler.compute_axis_origins(1601, 1600, 320)
        assert len(grid) == len(xs) * len(ys)
        assert grid[0] == (0, 0)
        assert grid[1] == (xs[1], 0)  # 列優先


class TestIterTiles:
    def test_views_match_source(self):
        img = np.random.randint(0, 255, (2000, 3000, 3), dtype=np.uint8)
        seen = 0
        for seq, x0, y0, view in tif_tiler.iter_tiles(img, 1600, 320):
            assert view.shape[0] <= 1600 and view.shape[1] <= 1600
            assert np.shares_memory(view, img)  # 零複製
            assert (view == img[y0 : y0 + view.shape[0], x0 : x0 + view.shape[1]]).all()
            assert seq == seen + 1
            seen = seq
        # xs: [0, 1280, 1400]、ys: [0, 400]（最後一片 clamp 到 dim - tile）
        assert seen == 6

    def test_small_image_one_tile(self):
        img = np.zeros((500, 700, 3), dtype=np.uint8)
        tiles = list(tif_tiler.iter_tiles(img, 1600, 320))
        assert len(tiles) == 1
        assert tiles[0][3].shape == (500, 700, 3)


class TestIsBlank:
    def test_all_black(self):
        assert tif_tiler.is_blank(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_all_white(self):
        assert tif_tiler.is_blank(np.full((100, 100, 3), 255, dtype=np.uint8))

    def test_uniform_gray(self):
        assert tif_tiler.is_blank(np.full((100, 100, 3), 128, dtype=np.uint8))

    def test_mostly_nodata_with_speck(self):
        tile = np.zeros((100, 100, 3), dtype=np.uint8)
        tile[50, 50] = (10, 200, 30)  # 單一雜點
        assert tif_tiler.is_blank(tile)

    def test_real_content(self):
        rng = np.random.default_rng(0)
        tile = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        assert not tif_tiler.is_blank(tile)

    def test_half_nodata_half_content(self):
        rng = np.random.default_rng(0)
        tile = np.zeros((100, 100, 3), dtype=np.uint8)
        tile[:, 50:] = rng.integers(1, 255, (100, 50, 3), dtype=np.uint8)
        assert not tif_tiler.is_blank(tile)


def test_tile_file_name_convention():
    assert tif_tiler.tile_file_name("3074571", 7, 5932, 4262, 1600, 1600) == "3074571_0007_5932_4262_1600x1600.png"

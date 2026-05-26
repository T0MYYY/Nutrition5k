import tempfile, os, csv
import pytest
from nutrition5k_pkg.data.metadata import load_dish_metadata, get_train_val_test_split


def _write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def test_load_single_csv():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'cafe1.csv')
        _write_csv(p, [
            ['dish_1234567890', 300.0, 250.0, 15.0, 30.0, 20.0, 1,
             'ingr_0000000001', 'chicken', 100.0, 200.0, 10.0, 0.0, 20.0],
        ])
        meta = load_dish_metadata(p)
    assert 'dish_1234567890' in meta
    r = meta['dish_1234567890']
    assert r['calories'] == pytest.approx(300.0)
    assert r['mass'] == pytest.approx(250.0)
    assert r['fat'] == pytest.approx(15.0)
    assert r['carb'] == pytest.approx(30.0)
    assert r['protein'] == pytest.approx(20.0)


def test_load_two_csvs():
    with tempfile.TemporaryDirectory() as d:
        p1 = os.path.join(d, 'cafe1.csv')
        p2 = os.path.join(d, 'cafe2.csv')
        _write_csv(p1, [['dish_0000000001', 100.0, 80.0, 5.0, 10.0, 8.0, 0]])
        _write_csv(p2, [['dish_0000000002', 200.0, 160.0, 10.0, 20.0, 15.0, 0]])
        meta = load_dish_metadata(p1, p2)
    assert len(meta) == 2


def test_split_ratios():
    dish_ids = [f'dish_{i:010d}' for i in range(100)]
    train, val, test = get_train_val_test_split(dish_ids, val_ratio=0.1, test_ratio=0.1, seed=42)
    assert len(train) + len(val) + len(test) == 100
    assert 8 <= len(test) <= 12   # ~10% ± 2
    assert 8 <= len(val) <= 12


def test_split_is_deterministic():
    dish_ids = [f'dish_{i:010d}' for i in range(200)]
    a = get_train_val_test_split(dish_ids, seed=42)
    b = get_train_val_test_split(dish_ids, seed=42)
    assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


def test_split_no_overlap():
    dish_ids = [f'dish_{i:010d}' for i in range(200)]
    train, val, test = get_train_val_test_split(dish_ids, seed=42)
    assert not (set(train) & set(test))
    assert not (set(train) & set(val))
    assert not (set(val) & set(test))

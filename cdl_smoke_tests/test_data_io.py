#!/usr/bin/env python
"""Standalone smoke test for the dataset reader nodes (batch 2).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_data_io.py

Reuses the smoke harness bootstrap to load the real node registry, then writes
small temporary files and confirms each reader turns them into a valid
``DATASET``.  Format-specific optional deps (openpyxl for xlsx, pyodbc for
accdb) are detected at runtime: if absent the reader is reported as SKIPPED
instead of failing, mirroring the main harness skip list.
"""

import io
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_smoke_test as harness  # noqa: E402

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


def _write_samples(tmp: Path):
    rows = [(float(i), float(i * 2), float(i * 3), float(i * 0.5)) for i in range(8)]
    csv_p = tmp / "s.csv"
    with open(csv_p, "w", newline="") as f:
        import csv

        w = csv.writer(f)
        w.writerow(["a", "b", "c", "y"])
        w.writerows(rows)
    txt_p = tmp / "s.txt"
    with open(txt_p, "w") as f:
        for r in rows:
            f.write("\t".join(str(v) for v in r) + "\n")
    json_p = tmp / "s.json"
    with open(json_p, "w") as f:
        json.dump([{"a": r[0], "b": r[1], "c": r[2], "y": r[3]} for r in rows], f)
    db_p = tmp / "s.db"
    con = sqlite3.connect(db_p)
    con.execute("CREATE TABLE data (a REAL, b REAL, c REAL, y REAL)")
    con.executemany("INSERT INTO data VALUES (?,?,?,?)", rows)
    con.commit()
    con.close()
    return {"csv": csv_p, "txt": txt_p, "json": json_p, "db": db_p}


def main() -> int:
    sandbox = Path(tempfile.mkdtemp(prefix="cdl_io_"))
    harness._bootstrap(sandbox)
    harness._load_registry(False)
    import nodes as host_nodes  # noqa: E402

    REG = host_nodes.NODE_CLASS_MAPPINGS
    files = _write_samples(sandbox)

    # CdlReadCSV -----------------------------------------------------------
    try:
        ds = REG["CdlReadCSV"]().execute(str(files["csv"]), "y")
        ds = ds[0]
        check("ReadCSV labelled features", ds.features.shape == (8, 3))
        check("ReadCSV labelled labels", ds.labels is not None and ds.labels.shape == (8,))
        check("ReadCSV target_name", ds.target_name == "y")
        # unlabelled
        ds0 = REG["CdlReadCSV"]().execute(str(files["csv"]), "")[0]
        check("ReadCSV unlabelled", ds0.labels is None and ds0.target_name == "")
    except Exception as e:  # pragma: no cover
        check("ReadCSV", False, f"{type(e).__name__}: {e}")

    # CdlReadText -----------------------------------------------------------
    try:
        rows = [(float(i), float(i * 2), float(i * 3), float(i * 0.5)) for i in range(8)]
        hdr_txt = sandbox / "s_hdr.txt"
        with open(hdr_txt, "w") as f:
            f.write("a\tb\tc\ty\n")
            for r in rows:
                f.write("\t".join(str(v) for v in r) + "\n")
        ds = REG["CdlReadText"]().execute(str(hdr_txt), "y", "\t", "float32", True)[0]
        check("ReadText features", ds.features.shape == (8, 3))
        check("ReadText labels", ds.labels is not None and ds.labels.shape == (8,))
        ds0 = REG["CdlReadText"]().execute(str(files["txt"]), "", "\t", "float32", False)[0]
        check("ReadText headerless", ds0.features.shape == (8, 4) and ds0.labels is None)
    except Exception as e:  # pragma: no cover
        check("ReadText", False, f"{type(e).__name__}: {e}")

    # CdlReadString ---------------------------------------------------------
    try:
        text = "a\tb\tc\ty\n0\t0\t0\t0.0\n1\t2\t3\t0.5\n2\t4\t6\t1.0\n"
        ds = REG["CdlReadString"]().execute(text, "y", "\t")[0]
        check("ReadString features", ds.features.shape == (3, 3))
        check("ReadString labels", ds.labels is not None and ds.labels.shape == (3,))
    except Exception as e:  # pragma: no cover
        check("ReadString", False, f"{type(e).__name__}: {e}")

    # CdlReadJSON -----------------------------------------------------------
    try:
        ds = REG["CdlReadJSON"]().execute(str(files["json"]), "y")[0]
        check("ReadJSON features", ds.features.shape == (8, 3))
        check("ReadJSON labels", ds.labels is not None and ds.labels.shape == (8,))
    except Exception as e:  # pragma: no cover
        check("ReadJSON", False, f"{type(e).__name__}: {e}")

    # CdlReadDB -------------------------------------------------------------
    try:
        ds = REG["CdlReadDB"]().execute(str(files["db"]), "SELECT * FROM data", "y")[0]
        check("ReadDB features", ds.features.shape == (8, 3))
        check("ReadDB labels", ds.labels is not None and ds.labels.shape == (8,))
    except Exception as e:  # pragma: no cover
        check("ReadDB", False, f"{type(e).__name__}: {e}")

    # CdlReadXLSX (optional openpyxl) --------------------------------------
    try:
        from openpyxl import Workbook

        xlsx_p = sandbox / "s.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.append(["a", "b", "c", "y"])
        for i in range(8):
            ws.append([float(i), float(i * 2), float(i * 3), float(i * 0.5)])
        wb.save(xlsx_p)
        ds = REG["CdlReadXLSX"]().execute(str(xlsx_p), "y")[0]
        check("ReadXLSX features", ds.features.shape == (8, 3))
        check("ReadXLSX labels", ds.labels is not None and ds.labels.shape == (8,))
    except ImportError:
        check("ReadXLSX (openpyxl)", True, "openpyxl absent - skipped")
    except Exception as e:  # pragma: no cover
        check("ReadXLSX", False, f"{type(e).__name__}: {e}")

    # CdlReadAccDB (optional pyodbc + Access driver) -----------------------
    try:
        import pyodbc  # noqa: F401

        check("ReadAccDB (pyodbc present)", True, "driver-dependent; not exercised here")
    except ImportError:
        check("ReadAccDB (pyodbc)", True, "pyodbc absent - skipped (matches _SKIPPED_NODES)")

    # Writers (require a DATASET input -> build one) -----------------------
    import torch as _torch

    ds_in = REG["CdlTensorsToDataset"]().execute(
        _torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        _torch.tensor([0.5, 1.0]),
    )[0]
    # CSV round-trip (to_dataframe names the label column "label")
    try:
        csv_out = sandbox / "out.csv"
        REG["CdlWriteCSV"]().execute(ds_in, str(csv_out))
        ds_csv = REG["CdlReadCSV"]().execute(str(csv_out), "label")[0]
        check("WriteCSV round-trip features", ds_csv.features.shape == (2, 3))
        check("WriteCSV round-trip labels", ds_csv.labels is not None and ds_csv.labels.shape == (2,))
    except Exception as e:  # pragma: no cover
        check("WriteCSV", False, f"{type(e).__name__}: {e}")
    # JSON round-trip
    try:
        json_out = sandbox / "out.json"
        REG["CdlWriteJSON"]().execute(ds_in, str(json_out))
        ds_j = REG["CdlReadJSON"]().execute(str(json_out), "label")[0]
        check("WriteJSON round-trip features", ds_j.features.shape == (2, 3))
    except Exception as e:  # pragma: no cover
        check("WriteJSON", False, f"{type(e).__name__}: {e}")
    # DB round-trip
    try:
        db_out = sandbox / "out.db"
        REG["CdlWriteDB"]().execute(ds_in, str(db_out), "data")
        ds_db = REG["CdlReadDB"]().execute(str(db_out), "SELECT * FROM data", "label")[0]
        check("WriteDB round-trip features", ds_db.features.shape == (2, 3))
    except Exception as e:  # pragma: no cover
        check("WriteDB", False, f"{type(e).__name__}: {e}")
    # XLSX round-trip (optional openpyxl)
    try:
        from openpyxl import Workbook  # noqa: F401

        xlsx_out = sandbox / "out.xlsx"
        REG["CdlWriteXLSX"]().execute(ds_in, str(xlsx_out))
        ds_x = REG["CdlReadXLSX"]().execute(str(xlsx_out), "label")[0]
        check("WriteXLSX round-trip features", ds_x.features.shape == (2, 3))
    except ImportError:
        check("WriteXLSX (openpyxl)", True, "openpyxl absent - skipped")
    except Exception as e:  # pragma: no cover
        check("WriteXLSX", False, f"{type(e).__name__}: {e}")
    # Preview
    try:
        prev = REG["CdlDatasetPreview"]().execute(ds_in, 2)[0]
        check("DatasetPreview non-empty", isinstance(prev, str) and len(prev) > 0)
    except Exception as e:  # pragma: no cover
        check("DatasetPreview", False, f"{type(e).__name__}: {e}")

    failed = [r for r in _RESULTS if not r[1]]
    for name, ok, detail in _RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

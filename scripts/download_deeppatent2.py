"""DeepPatent2 (Harvard Dataverse, DOI 10.7910/DVN/UG4SBD) 연도별 다운로드.

연도별로 split 아카이브(2019.aa, 2019.ab, ...)를 받아 병합 후 tar 해제한다.
전체 287GB이므로 --years 로 필요한 연도만 받는 것을 권장.

  python scripts/download_deeppatent2.py --years 2013            # 1개 연도(검증용)
  python scripts/download_deeppatent2.py --years 2018 2019 2020  # 여러 연도
  python scripts/download_deeppatent2.py --list                  # 연도별 용량만 조회
"""
import argparse
import os
import tarfile
import zipfile

import requests

DOI = "doi:10.7910/DVN/UG4SBD"
BASE = "https://dataverse.harvard.edu"
API_LIST = f"{BASE}/api/datasets/:persistentId/?persistentId={DOI}"
API_FILE = f"{BASE}/api/access/datafile"


def list_files():
    r = requests.get(API_LIST, timeout=60)
    r.raise_for_status()
    files = r.json()["data"]["latestVersion"]["files"]
    out = []
    for f in files:
        df = f["dataFile"]
        label = f.get("label") or df.get("filename")
        out.append({"label": label, "id": df["id"], "size": df.get("filesize", 0)})
    return out


def human(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def download_file(fid, dest):
    if os.path.exists(dest):
        print(f"  skip (exists): {os.path.basename(dest)}")
        return
    tmp = dest + ".part"
    with requests.get(f"{API_FILE}/{fid}", stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    os.rename(tmp, dest)
    print(f"  done: {os.path.basename(dest)} ({human(os.path.getsize(dest))})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", default=[])
    ap.add_argument("--out", default="models/deeppatent2")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--keep-parts", action="store_true", help="병합 후 원본 split 파일 유지")
    args = ap.parse_args()

    files = list_files()

    if args.list or not args.years:
        by_year = {}
        for f in files:
            yr = f["label"].split(".")[0]
            by_year.setdefault(yr, [0, 0])
            by_year[yr][0] += 1
            by_year[yr][1] += f["size"]
        print("연도별 용량 (part 수 / 크기):")
        for yr in sorted(by_year):
            n, sz = by_year[yr]
            print(f"  {yr}: {n} parts, {human(sz)}")
        if not args.years:
            print("\n--years 2013 처럼 연도를 지정해 다운로드하세요.")
            return

    raw = os.path.join(args.out, "raw")
    ext = os.path.join(args.out, "extracted")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(ext, exist_ok=True)

    for yr in args.years:
        parts = sorted([f for f in files if f["label"].startswith(f"{yr}.")],
                       key=lambda x: x["label"])
        if not parts:
            print(f"[{yr}] 해당 연도 파일 없음 — 건너뜀")
            continue
        total = human(sum(p["size"] for p in parts))
        print(f"[{yr}] {len(parts)} parts, {total} 다운로드")
        for p in parts:
            download_file(p["id"], os.path.join(raw, p["label"]))

        # split 파트 병합 (DeepPatent2 아카이브는 실제로 ZIP을 split한 것)
        merged = os.path.join(raw, f"{yr}.archive")
        print(f"[{yr}] 병합 -> {merged}")
        with open(merged, "wb") as out:
            for p in parts:
                with open(os.path.join(raw, p["label"]), "rb") as fh:
                    while True:
                        buf = fh.read(1 << 22)
                        if not buf:
                            break
                        out.write(buf)
        ydir = os.path.join(ext, yr)
        print(f"[{yr}] 압축 해제(zip) -> {ydir}")
        if zipfile.is_zipfile(merged):
            with zipfile.ZipFile(merged) as zf:
                zf.extractall(ydir)
        else:
            with tarfile.open(merged) as tar:
                tar.extractall(ydir)
        os.remove(merged)

        # zip 내부에 design{YYYY}.json + Segmented.tar.gz(+Original.tar.gz)가 중첩됨.
        # 학습에 필요한 Segmented(세그먼트 도면) tar.gz를 마저 해제.
        for root, _, fnames in os.walk(ydir):
            for fn in fnames:
                if fn.lower() == "segmented.tar.gz":
                    seg = os.path.join(root, fn)
                    print(f"[{yr}] 세그먼트 도면 해제 -> {root}")
                    with tarfile.open(seg) as tar:
                        tar.extractall(root)
                    os.remove(seg)
                elif fn.lower() == "original.tar.gz":
                    os.remove(os.path.join(root, fn))  # 원본 합성도면은 미사용 → 용량 절약
        if not args.keep_parts:
            for p in parts:
                os.remove(os.path.join(raw, p["label"]))
        print(f"[{yr}] 완료")

    print("모든 연도 처리 완료. 다음: python scripts/build_pairs.py")


if __name__ == "__main__":
    main()

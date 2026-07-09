"""DeepPatent2 추출본(JSON 메타데이터 + 세그먼트 PNG) → data/pairs.jsonl 변환.

각 세그먼트 도면을 한 줄로:
  {"image": <절대경로>, "text": <디자인 명칭>, "design_id": <patentID>, "viewpoint": <aspect>}

- text = object_title(특허 제목 = 디자인의 명칭). 비면 object(물품명)로 대체.
- design_id = patentID  → 같은 출원의 여러 뷰가 이미지↔이미지 positive.
- 영어로 학습(다국어 베이스). 한글 증강은 나중에 별도 컬럼 추가로 가능.

  python scripts/build_pairs.py --src models/deeppatent2/extracted --out data/pairs.jsonl
"""
import argparse
import json
import os


def index_pngs(root):
    """basename(소문자) -> 절대경로. subfigure_file 해석용."""
    idx = {}
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".png"):
                idx[fn.lower()] = os.path.abspath(os.path.join(dp, fn))
    return idx


def iter_records(json_path):
    """JSON 배열 또는 JSON-lines 모두 지원."""
    with open(json_path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            try:
                for r in json.load(f):
                    yield r
                return
            except json.JSONDecodeError:
                f.seek(0)
        for line in f:
            line = line.strip().rstrip(",")
            if line and line not in "[]":
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def clean(s):
    return " ".join(str(s).split()).strip() if s else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/deeppatent2/extracted")
    ap.add_argument("--out", default="data/pairs.jsonl")
    ap.add_argument("--include-aspect", action="store_true",
                    help="text 끝에 뷰포인트 부가 (예: 'sneaker, perspective view')")
    args = ap.parse_args()

    print(f">> PNG 인덱싱: {args.src}")
    png = index_pngs(args.src)
    print(f"   {len(png):,} PNG 발견")

    json_files = []
    for dp, _, files in os.walk(args.src):
        for fn in files:
            if fn.lower().endswith(".json"):
                json_files.append(os.path.join(dp, fn))
    print(f">> JSON 메타데이터 {len(json_files)}개 파싱")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_written = n_no_img = n_no_text = 0
    seen_designs = set()
    with open(args.out, "w", encoding="utf-8") as out:
        for jf in json_files:
            for r in iter_records(jf):
                sub = r.get("subfigure_file") or r.get("figure_file")
                if not sub:
                    continue
                path = png.get(os.path.basename(sub).lower())
                if not path:
                    n_no_img += 1
                    continue
                text = clean(r.get("object_title")) or clean(r.get("object"))
                if not text:
                    n_no_text += 1
                    continue
                if args.include_aspect and clean(r.get("aspect")):
                    text = f"{text}, {clean(r.get('aspect'))}"
                design_id = clean(r.get("patentID")) or os.path.basename(sub).split("-")[0]
                out.write(json.dumps({
                    "image": path,
                    "text": text,
                    "design_id": design_id,
                    "viewpoint": clean(r.get("aspect")),
                    "locarno": clean(r.get("classification_locarno")),
                }, ensure_ascii=False) + "\n")
                n_written += 1
                seen_designs.add(design_id)

    print(f"\n>> 완료: {n_written:,} 쌍 -> {args.out}")
    print(f"   고유 디자인(design_id): {len(seen_designs):,}")
    print(f"   스킵: 이미지없음 {n_no_img:,}, 텍스트없음 {n_no_text:,}")


if __name__ == "__main__":
    main()

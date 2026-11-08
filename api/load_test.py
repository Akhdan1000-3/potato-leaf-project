"""
Uji beban (latency dan stress test) untuk API klasifikasi daun kentang.
Hanya memakai pustaka bawaan Python (tambah Pillow bila tidak memberi --image).

Contoh (API sudah berjalan di 127.0.0.1:8000):
    python api/load_test.py --url http://127.0.0.1:8000 --image contoh.jpg --levels 1,5,10,20 --duration 20

Tiap level = jumlah pengguna serentak. Hasil dicetak sebagai tabel dan disimpan ke CSV
(default results/load_test.csv): jumlah request, error, throughput, rata-rata, p50, p95, p99, maksimum.
"""
import argparse
import csv
import http.client
import io
import statistics
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

BOUNDARY = "----potato" + uuid.uuid4().hex


def multipart(image: bytes) -> bytes:
    head = (
        f"--{BOUNDARY}\r\n"
        'Content-Disposition: form-data; name="file"; filename="leaf.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    return head + image + f"\r\n--{BOUNDARY}--\r\n".encode()


def sample_image() -> bytes:
    from PIL import Image  # hanya dipakai bila --image tidak diberikan
    import random

    rnd = random.Random(1)
    img = Image.new("RGB", (256, 256))
    img.putdata([(rnd.randint(0, 120), rnd.randint(80, 255), rnd.randint(0, 120)) for _ in range(256 * 256)])
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


def new_connection(u):
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    return cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=30)


def worker(u, path, body, stop_at, results, lock):
    headers = {"Content-Type": f"multipart/form-data; boundary={BOUNDARY}", "Content-Length": str(len(body))}
    conn = new_connection(u)
    local = []
    while time.time() < stop_at:
        t0 = time.perf_counter()
        status = 0
        try:
            conn.request("POST", path, body=body, headers=headers)
            resp = conn.getresponse()
            resp.read()
            status = resp.status
        except Exception:
            conn.close()
            conn = new_connection(u)
        local.append(((time.perf_counter() - t0) * 1000, status))
    conn.close()
    with lock:
        results.extend(local)


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = min(len(sorted_vals) - 1, int(round(p / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[k]


def run_level(url, model, body, users, duration):
    u = urlparse(url)
    path = f"/v1/predict?model={model}"
    results, lock = [], threading.Lock()
    stop_at = time.time() + duration
    t_start = time.time()
    threads = [threading.Thread(target=worker, args=(u, path, body, stop_at, results, lock)) for _ in range(users)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t_start
    ok = sorted(ms for ms, st in results if st == 200)
    errors = sum(1 for _, st in results if st != 200)
    total = len(results)
    return {
        "users": users,
        "duration_s": round(elapsed, 1),
        "requests": total,
        "errors": errors,
        "error_rate_pct": round(100 * errors / total, 2) if total else 0.0,
        "throughput_rps": round(total / elapsed, 2) if elapsed else 0.0,
        "mean_ms": round(statistics.mean(ok), 1) if ok else 0.0,
        "p50_ms": round(pct(ok, 50), 1),
        "p95_ms": round(pct(ok, 95), 1),
        "p99_ms": round(pct(ok, 99), 1),
        "max_ms": round(max(ok), 1) if ok else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--image", help="path gambar daun untuk dikirim (JPEG/PNG)")
    ap.add_argument("--model", default="mobilenetv2", choices=["mobilenetv2", "efficientnetb0"])
    ap.add_argument("--levels", default="1,5,10,20", help="jumlah pengguna serentak, dipisah koma")
    ap.add_argument("--duration", type=int, default=20, help="detik per level")
    ap.add_argument("--out", default="results/load_test.csv")
    args = ap.parse_args()

    image = Path(args.image).read_bytes() if args.image else sample_image()
    body = multipart(image)
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    print(f"Target {args.url} | model {args.model} | {args.duration} detik per level | gambar {len(image) / 1024:.0f} KB")
    run_level(args.url, args.model, body, 1, 2)  # pemanasan singkat, tidak dicatat
    rows = []
    header = f"{'users':>5} {'req':>6} {'err':>4} {'err%':>6} {'rps':>7} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}"
    print(header)
    for n in levels:
        r = run_level(args.url, args.model, body, n, args.duration)
        rows.append(r)
        print(f"{r['users']:>5} {r['requests']:>6} {r['errors']:>4} {r['error_rate_pct']:>6} {r['throughput_rps']:>7} "
              f"{r['mean_ms']:>8} {r['p50_ms']:>8} {r['p95_ms']:>8} {r['p99_ms']:>8} {r['max_ms']:>8}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nHasil disimpan ke {out}")


if __name__ == "__main__":
    main()

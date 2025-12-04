#!/usr/bin/env python3
# hbn_eeg_dl.py
# Public S3 -> local directory, resumable and restartable.
# Stdlib only.

"""
Util script to pull the entire HBN dataset from the public aws bucket into the current working directory.
When executed with default parameters, the script will do the following:
- Create folders bucket_tars and bucket_tars.staging
- Using 4 parallel workers, download each subject into bucket_tars.staging. When done, the finished .tar.gz is moved into bucket_tars
- Do this for all subjects

The script is robust to crashes / restarts, and will continue from where it was left of.
The staging area guarantees that no partial downloads land in bucket_tars.

As the time of writing, the bucket contains 4576 archives that are 5.6 TiB large in sum.
The downloaded EEG data will not be used for this project.
If disk space is a concern, it may be removed using a command like:
find . -type d -name 'EEG' -exec rm -rf {} +

LLM Code
"""


import argparse
import os
import sys
import time
import threading
import socket
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

BUCKET_DEFAULT = "fcp-indi"
PREFIX_DEFAULT = "data/Archives/HBN/EEG/"
BASE_URL_TPL = "https://{bucket}.s3.amazonaws.com"
NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

# ---------- utils
def human(n):
    for unit in ["B","KiB","MiB","GiB","TiB"]:
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def file_size(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return 0

# ---------- list S3 (ListObjectsV2, unsigned)
def list_s3(bucket, prefix, base_url):
    url_base = f"{base_url}"
    params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
    token = None
    results = []

    while True:
        qp = params.copy()
        if token:
            qp["continuation-token"] = token
        url = url_base + "?" + urlencode(qp)
        req = Request(url, headers={"User-Agent":"python-s3-list/1.0"})
        with urlopen(req, timeout=60) as r:
            data = r.read()
        root = ET.fromstring(data)

        for c in root.findall("s3:Contents", NAMESPACE):
            key = c.find("s3:Key", NAMESPACE).text
            size = int(c.find("s3:Size", NAMESPACE).text)
            if key.endswith(".tar.gz"):
                # Store relative path under prefix
                rel = key[len(prefix):]
                results.append((rel, size))

        is_trunc = root.findtext("s3:IsTruncated", default="false", namespaces=NAMESPACE).lower() == "true"
        if not is_trunc:
            break
        token_el = root.find("s3:NextContinuationToken", NAMESPACE)
        if token_el is None or not token_el.text:
            break
        token = token_el.text
    return results

# ---------- downloader
class Downloader:
    def __init__(self, bucket, prefix, dest_dir, workers, max_attempts, chunk_size, timeout):
        self.bucket = bucket
        self.prefix = prefix if prefix.endswith("/") else prefix + "/"
        self.base_url = BASE_URL_TPL.format(bucket=bucket)
        self.dest_dir = os.path.abspath(dest_dir)
        self.stage_dir = self.dest_dir + ".staging"
        self.ok_log = os.path.join(os.getcwd(), "eeg_ok.log")
        self.fail_log = os.path.join(os.getcwd(), "eeg_failures.log")
        self.manifest_path = os.path.join(os.getcwd(), "eeg_manifest.tsv")
        self.workers = workers
        self.max_attempts = max_attempts
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.print_lock = threading.Lock()
        self.log_lock = threading.Lock()
        ensure_dir(self.dest_dir)
        ensure_dir(self.stage_dir)

    def log(self, path, line):
        with self.log_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def url_for(self, rel_key):
        # Build URL with proper encoding
        key = self.prefix + rel_key
        return f"{self.base_url}/{quote(key)}"

    def download_one(self, rel_key, size):
        final_path = os.path.join(self.dest_dir, rel_key)
        stage_path = os.path.join(self.stage_dir, rel_key) + ".part"
        ensure_dir(os.path.dirname(final_path))
        ensure_dir(os.path.dirname(stage_path))

        # Skip if already complete
        if os.path.exists(final_path) and file_size(final_path) == size:
            self.log(self.ok_log, f"SKIP\t{rel_key}")
            return True

        # If a previous staged file already has full size, just finalize
        if os.path.exists(stage_path) and file_size(stage_path) == size:
            os.replace(stage_path, final_path)
            self.log(self.ok_log, f"OK\t{rel_key}")
            return True

        # Reset bogus partial larger than expected
        if os.path.exists(stage_path) and file_size(stage_path) > size:
            try:
                os.remove(stage_path)
            except OSError:
                pass

        url = self.url_for(rel_key)
        attempt = 0

        while attempt < self.max_attempts:
            attempt += 1
            start_sz = file_size(stage_path)
            headers = {"User-Agent": "python-s3-dl/1.0"}
            if start_sz > 0 and start_sz < size:
                headers["Range"] = f"bytes={start_sz}-"

            req = Request(url, headers=headers)
            try:
                with urlopen(req, timeout=self.timeout) as resp:
                    code = resp.getcode()
                    if start_sz > 0 and code not in (206, 200):
                        raise HTTPError(url, code, "Unexpected status for range", hdrs=resp.headers, fp=None)
                    # Append if resuming, else write new
                    mode = "ab" if start_sz > 0 else "wb"
                    with open(stage_path, mode) as out:
                        while True:
                            chunk = resp.read(self.chunk_size)
                            if not chunk:
                                break
                            out.write(chunk)
                # Verify size
                if file_size(stage_path) == size:
                    os.replace(stage_path, final_path)
                    self.log(self.ok_log, f"OK\t{rel_key}")
                    with self.print_lock:
                        print(f"OK   {rel_key}")
                    return True
                else:
                    raise IOError("size_mismatch_after_download")

            except (HTTPError, URLError, ConnectionResetError, TimeoutError, IOError) as e:
                # Backoff and retry
                back = min(60, 2 ** attempt)
                with self.print_lock:
                    got = file_size(stage_path)
                    print(f"RETRY {rel_key} attempt {attempt}/{self.max_attempts} "
                          f"status={getattr(e, 'code', 'NA')} size={got}/{size} sleep={back}s")
                time.sleep(back)
                continue

        self.log(self.fail_log, f"FAIL\t{rel_key}")
        with self.print_lock:
            print(f"FAIL {rel_key}")
        return False

    def run(self):
        socket.setdefaulttimeout(self.timeout)
        # Build manifest
        with self.print_lock:
            print(f"Listing s3://{self.bucket}/{self.prefix} ...")
        files = list_s3(self.bucket, self.prefix, self.base_url)
        # Persist manifest for reference
        with open(self.manifest_path, "w", encoding="utf-8") as mf:
            for rel, sz in files:
                mf.write(f"{sz}\t{rel}\n")

        total = len(files)
        total_bytes = sum(sz for _, sz in files)
        with self.print_lock:
            print(f"Found {total} files, {human(total_bytes)}")

        # Submit downloads
        ok = 0
        fail = 0
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(self.download_one, rel, sz): (rel, sz) for rel, sz in files}
            for fut in as_completed(futs):
                try:
                    if fut.result():
                        ok += 1
                    else:
                        fail += 1
                except Exception as e:
                    fail += 1
        with self.print_lock:
            print(f"Done. OK={ok} FAIL={fail}  Logs: {self.ok_log} {self.fail_log}")

def parse_args():
    p = argparse.ArgumentParser(description="Resumable public S3 downloader (EEG).")
    p.add_argument("--bucket", default=BUCKET_DEFAULT)
    p.add_argument("--prefix", default=PREFIX_DEFAULT)
    p.add_argument("--dest", default=os.path.join(os.getcwd(), "bucket_tars"))
    p.add_argument("--workers", type=int, default=4, help="parallel downloads")
    p.add_argument("--max-attempts", type=int, default=8)
    p.add_argument("--chunk-mib", type=int, default=8)
    p.add_argument("--timeout", type=int, default=60, help="per-request timeout seconds")
    return p.parse_args()

def main():
    args = parse_args()
    dl = Downloader(
        bucket=args.bucket,
        prefix=args.prefix,
        dest_dir=args.dest,
        workers=args.workers,
        max_attempts=args.max_attempts,
        chunk_size=args.chunk_mib * 1024 * 1024,
        timeout=args.timeout,
    )
    dl.run()

if __name__ == "__main__":
    main()

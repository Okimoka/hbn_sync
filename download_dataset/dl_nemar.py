#!/usr/bin/env python3
# dl_nemar.py
# Resumable NeMAR zip downloader with staging + parallel workers (stdlib only).

"""
Util script to pull all nemar HBN-EEG datasets from nemar into the current working directory.
This is an adapted version of dl_bucket.py. Its source code contains more information.

The script is robust to crashes / restarts, and will continue from where it was left of.

The datasets are 1.7 TiB large in sum.

LLM Code
"""


import argparse
import os
import sys
import time
import threading
import socket
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from concurrent.futures import ThreadPoolExecutor, as_completed

# Base URL used in your bash script
BASE_URL_DEFAULT = (
    "https://nemar.org/dataexplorer/download?filepath=/data/nemar/openneuro//zip_files"
)

# Default HBN EEG datasets to download when no IDs are given
DEFAULT_DSIDS = [
    "ds005505", # Release 1
    "ds005506", # Release 2
    "ds005507", # Release 3
    "ds005508", # Release 4
    "ds005509", # Release 5
    "ds005510", # Release 6
    "ds005511", # Release 7
    "ds005512", # Release 8
    "ds005514", # Release 9
    "ds005515", # Release 10
    "ds005516", # Release 11
]


def human(n):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
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


class Downloader:
    """
    Download dsXXXXXX.zip archives from NeMAR with:
    - staging area (dest_dir.staging)
    - resumable downloads (HTTP Range)
    - multiple worker threads
    """

    def __init__(self, base_url, dest_dir, workers, max_attempts, chunk_size, timeout):
        self.base_url = base_url.rstrip("/")
        self.dest_dir = os.path.abspath(dest_dir)
        self.stage_dir = self.dest_dir + ".staging"

        self.ok_log = os.path.join(os.getcwd(), "nemar_ok.log")
        self.fail_log = os.path.join(os.getcwd(), "nemar_failures.log")

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

    def url_for(self, dsid):
        return f"{self.base_url}/{dsid}.zip"

    def download_one(self, dsid):
        final_path = os.path.join(self.dest_dir, f"{dsid}.zip")
        stage_path = os.path.join(self.stage_dir, f"{dsid}.zip.part")
        ensure_dir(os.path.dirname(final_path))
        ensure_dir(os.path.dirname(stage_path))

        # Already complete?
        if os.path.exists(final_path):
            self.log(self.ok_log, f"SKIP\t{dsid}")
            return True

        attempt = 0

        while attempt < self.max_attempts:
            attempt += 1
            start_sz = file_size(stage_path)
            headers = {"User-Agent": "python-nemar-dl/1.1"}
            if start_sz > 0:
                headers["Range"] = f"bytes={start_sz}-"

            url = self.url_for(dsid)

            with self.print_lock:
                print(
                    f"START {dsid} attempt {attempt}/{self.max_attempts} "
                    f"from {human(start_sz)}"
                )

            req = Request(url, headers=headers)

            try:
                with urlopen(req, timeout=self.timeout) as resp:
                    code = resp.getcode()
                    if start_sz > 0 and code not in (206, 200):
                        raise HTTPError(
                            url,
                            code,
                            "Unexpected status for range",
                            hdrs=resp.headers,
                            fp=None,
                        )

                    # Try to infer total size from headers (if present)
                    total_size = None
                    cl = resp.getheader("Content-Length")
                    if cl is not None:
                        try:
                            cl_val = int(cl)
                            total_size = start_sz + cl_val if start_sz > 0 else cl_val
                        except ValueError:
                            total_size = None

                    cr = resp.getheader("Content-Range")
                    if cr and "/" in cr:
                        try:
                            total_size = int(cr.split("/")[-1])
                        except ValueError:
                            pass

                    downloaded = start_sz
                    last_report = time.time()
                    mode = "ab" if start_sz > 0 else "wb"

                    with open(stage_path, mode) as out:
                        while True:
                            chunk = resp.read(self.chunk_size)
                            if not chunk:
                                break
                            out.write(chunk)
                            downloaded += len(chunk)

                            # Progress line every ~30s
                            now = time.time()
                            if now - last_report >= 30:
                                with self.print_lock:
                                    if total_size:
                                        print(
                                            f"PROG  {dsid}: "
                                            f"{human(downloaded)} / {human(total_size)}"
                                        )
                                    else:
                                        print(
                                            f"PROG  {dsid}: "
                                            f"{human(downloaded)} downloaded"
                                        )
                                last_report = now

                got = file_size(stage_path)
                if total_size is not None and got != total_size:
                    raise IOError(
                        f"size_mismatch_after_download {got} != {total_size}"
                    )

                os.replace(stage_path, final_path)
                self.log(self.ok_log, f"OK\t{dsid}")
                with self.print_lock:
                    print(f"OK   {dsid}")
                return True

            except (HTTPError, URLError, ConnectionResetError, OSError) as e:
                got = file_size(stage_path)
                back = min(60, 2**attempt)
                with self.print_lock:
                    print(
                        f"RETRY {dsid} attempt {attempt}/{self.max_attempts} "
                        f"status={getattr(e, 'code', 'NA')} size={human(got)} "
                        f"sleep={back}s ({e})"
                    )
                time.sleep(back)
                continue

        self.log(self.fail_log, f"FAIL\t{dsid}")
        with self.print_lock:
            print(f"FAIL {dsid}")
        return False

    def run(self, dsids):
        socket.setdefaulttimeout(self.timeout)

        # Deduplicate while preserving order
        seen = set()
        norm_dsids = []
        for dsid in dsids:
            if dsid not in seen:
                seen.add(dsid)
                norm_dsids.append(dsid)

        with self.print_lock:
            print(
                f"Datasets to download ({len(norm_dsids)}): "
                + ", ".join(norm_dsids)
            )
            print(f"Base URL:    {self.base_url}")
            print(f"Dest dir:    {self.dest_dir}")
            print(f"Staging dir: {self.stage_dir}")
            print(f"Workers:     {self.workers}")

        ok = 0
        fail = 0

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(self.download_one, dsid): dsid for dsid in norm_dsids}
            for fut in as_completed(futs):
                dsid = futs[fut]
                try:
                    if fut.result():
                        ok += 1
                    else:
                        fail += 1
                except Exception as e:
                    fail += 1
                    with self.print_lock:
                        print(f"UNEXPECTED ERROR {dsid}: {e}")

        with self.print_lock:
            print(f"Done. OK={ok} FAIL={fail}  Logs: {self.ok_log} {self.fail_log}")


def normalize_ids(raw_ids):
    """
    Convert numeric IDs to ds%06d, normalize ds-prefix forms.
    E.g. 5511 -> ds005511, ds5511 -> ds005511, ds005511 -> ds005511
    """
    norm = []
    for arg in raw_ids:
        a = arg.strip()
        if a.isdigit():
            dsid = f"ds{int(a):06d}"
        elif a.lower().startswith("ds") and a[2:].isdigit():
            dsid = f"ds{int(a[2:]):06d}"
        else:
            dsid = a
        norm.append(dsid)
    return norm


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Resumable NeMAR zip downloader with staging and parallel workers. "
            "If no IDs are given, downloads the default HBN EEG set."
        )
    )
    p.add_argument(
        "ids",
        nargs="*",
        help=(
            "Dataset IDs (e.g. 5511 or ds005511). "
            "Leave empty to use built-in defaults."
        ),
    )
    p.add_argument(
        "--base-url",
        default=BASE_URL_DEFAULT,
        help="Base NeMAR download URL (normally leave as default).",
    )
    p.add_argument(
        "--dest",
        default=os.path.join(os.getcwd(), "nemar_zips"),
        help="Destination directory for .zip files.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel downloads (default 4).",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=8,
        help="Max retry attempts per file.",
    )
    p.add_argument(
        "--chunk-mib",
        type=int,
        default=8,
        help="Download chunk size in MiB.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request timeout in seconds.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.ids:
        raw_ids = args.ids
    else:
        raw_ids = DEFAULT_DSIDS

    dsids = normalize_ids(raw_ids)

    if args.workers < 1:
        print("Error: --workers must be >= 1", file=sys.stderr)
        sys.exit(1)

    dl = Downloader(
        base_url=args.base_url,
        dest_dir=args.dest,
        workers=args.workers,
        max_attempts=args.max_attempts,
        chunk_size=args.chunk_mib * 1024 * 1024,
        timeout=args.timeout,
    )
    dl.run(dsids)


if __name__ == "__main__":
    main()

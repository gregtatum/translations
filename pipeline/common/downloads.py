import gzip
import io
import json
import multiprocessing
import os
import shutil
import subprocess
import threading
import time
from contextlib import ExitStack, contextmanager
from io import BufferedReader
from pathlib import Path
from typing import Any, Callable, Generator, Generic, Iterable, Literal, Optional, TypeVar, Union
from zipfile import ZipFile

import requests
from zstandard import ZstdCompressor, ZstdDecompressor

from pipeline.common import format_bytes
from pipeline.common.logging import get_logger

logger = get_logger(__file__)


def stream_download_to_file(url: str, destination: Union[str, Path]) -> None:
    """
    Streams a download to a file, and retries several times if there are any failures. The
    destination file must not already exist.
    """
    if os.path.exists(destination):
        raise Exception(f"That file already exists: {destination}")

    logger.info(f"Destination: {destination}")

    # If this is mocked for a test, use the locally mocked path.
    mocked_location = get_mocked_downloads_file_path(url)
    if mocked_location:
        shutil.copy(mocked_location, destination)
        return

    with open(destination, "wb") as file, DownloadChunkStreamer(url) as chunk_streamer:
        for chunk in chunk_streamer.download_chunks():
            file.write(chunk)


def get_mocked_downloads_file_path(url: str) -> Optional[str]:
    """If there is a mocked download, get the path to the file, otherwise return None"""
    mocked_downloads_str = os.environ.get("MOCKED_DOWNLOADS")
    if not mocked_downloads_str:
        return None

    mocked_downloads = json.loads(mocked_downloads_str)

    if not isinstance(mocked_downloads, dict):
        raise Exception(
            "Expected the mocked downloads to be a json object mapping the URL to file path"
        )

    source_file = mocked_downloads.get(url)
    if not source_file:
        print("MOCKED_DOWNLOADS:", json.dumps(mocked_downloads, indent=2))
        raise Exception(f"Received a URL that was not in MOCKED_DOWNLOADS {url}")

    if not os.path.exists(source_file):
        raise Exception(f"The source file specified did not exist {source_file}")

    logger.info("Mocking a download.")
    logger.info(f"   url: {url}")
    logger.info(f"  file: {source_file}")

    return source_file


def location_exists(location: str):
    """
    Checks if a location (url or file path) exists.
    """
    if get_mocked_downloads_file_path(location):
        return True

    if location.startswith("http://") or location.startswith("https://"):
        response = requests.head(location, allow_redirects=True)
        return response.ok
    return os.path.exists(location)


def attempt_mocked_request(url: str) -> Optional[BufferedReader]:
    """
    If there are mocked download, use that.
    """
    file_path = get_mocked_downloads_file_path(url)
    if file_path:
        return open(file_path, "rb")
    return None


def get_download_size(url: str) -> int:
    """Get the total bytes of a file to download."""
    mocked_file_path = get_mocked_downloads_file_path(url)
    if mocked_file_path:
        return os.path.getsize(mocked_file_path)

    response = requests.head(url, allow_redirects=True)
    size = response.headers.get("content-length", 0)
    return int(size)


class RemoteDecodingLineStreamer:
    """
    Base class to stream lines directly from a remote file.
    """

    def __init__(self, url: str) -> None:
        self.url = url

        self.decoding_stream = None
        self.byte_chunk_stream = None
        self.line_stream = None

    def __enter__(self):
        mocked_request = attempt_mocked_request(self.url)
        if mocked_request:
            # We are in a test.
            logger.info(f"Using a mocked download: {self.url}")
            self.byte_chunk_stream = mocked_request
            self.decoding_stream = self.decode(self.byte_chunk_stream)
        else:
            self.byte_chunk_stream = DownloadChunkStreamer(self.url).__enter__()
            self.decoding_stream = self.decode(self.byte_chunk_stream)

        self.line_stream = io.TextIOWrapper(self.decoding_stream, encoding="utf-8")

        return self.line_stream

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        if self.line_stream:
            self.line_stream.close()
        if self.decoding_stream:
            self.decoding_stream.close()
        if self.byte_chunk_stream:
            self.byte_chunk_stream.close()

    def decode(self, byte_stream: Any) -> Any:
        # This byte stream requires no decoding, so just pass it on through.
        return byte_stream


class RemoteGzipLineStreamer(RemoteDecodingLineStreamer):
    """
    Stream lines directly from a remote gzip file. The line includes the newlines separator.

    Usage:

        with RemoteGzipLineStreamer(url) as lines:
            for line in lines:
                print(line)
    """

    def decode(self, byte_stream):
        return gzip.GzipFile(fileobj=byte_stream)


class RemoteZstdLineStreamer(RemoteDecodingLineStreamer):
    """
    Stream lines directly from a remote zstd file. The line includes the newlines separator.

    Usage:

        with RemoteZstdLineStreamer(url) as lines:
            for line in lines:
                print(line)
    """

    def decode(self, byte_stream):
        return ZstdDecompressor().stream_reader(byte_stream)


class DownloadChunkStreamer(io.IOBase):
    """
    Streams a download as chunks, and retries several times if there are any failures. This
    clas implements io.IOBase so it can be used as a file reader.

    Iterator over chunks directly:

        with DownloadChunkStreamer(url) as chunk_streamer:
            for chunk in chunk_streamer.download_chunks():
                f.write(chunk)

    Or pass it as a file handle:

        with DownloadChunkStreamer(url) as f:
             gzip.GzipFile(fileobj=f)
    """

    def __init__(self, url: str, total_retries=3, timeout_sec=10.0, wait_before_retry_sec=60.0):
        self.url = url
        self.response = None

        # How many retry attempts should there be, and how long to wait between retries.
        self.total_retries = total_retries
        self.wait_before_retry_sec = wait_before_retry_sec

        # How long to wait for a response to timeout? This is the time that no new data is received.
        self.timeout_sec = timeout_sec

        self.report_every = 0.05  # What percentage of the download to report updates?
        self.next_report_percent = self.report_every  # The next report percentage.

        self.downloaded_bytes = 0
        self.chunk_bytes = 8 * 1024

        # The buffered `read` data.
        self.buffer = b""

        # The Generator result of _download_chunks.
        self.chunk_iter: Optional[Generator[bytes, None, None]] = None

    def __enter__(self):
        """
        On enter, kick off the download, and store the chunk iterator. This iterator
        handles the restarts for Requests.
        """
        self.chunk_iter = self.download_chunks()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """
        Close out the response, and cancel any iterators.
        """
        if self.response:
            self.response.close()

        self.response = None
        self.chunk_iter = None

    def read(self, size=-1) -> bytes:
        """
        This method implements the io.IOBase read method. It buffers the chunks until the `size`
        requirement is fulfilled. It is backed by the chunks_iter created by the download_chunks
        method.
        """
        if not self.chunk_iter:
            # The chunk iterator was consumed. Return an empty byte object to indicate the download
            # is complete.
            return b""

        if size < 0:
            # Load everything into the buffer, and return it.
            for chunk in self.chunk_iter:
                self.buffer += chunk
            result = self.buffer
            self.buffer = b""
            return result

        # Load the buffer with requested amount of data to read. -1 indicates load everything.
        while len(self.buffer) < size:
            chunk = next(self.chunk_iter, None)
            if chunk:
                self.buffer += chunk
            else:
                # The stream ended.
                break

        # Return the requested read amount, and divide up the remaining buffer.
        result = self.buffer[:size]
        self.buffer = self.buffer[size:]

        return result

    def readable(self):
        return True

    def download_chunks(self) -> Generator[bytes, None, None]:
        """
        This method is the generator that is responsible for running the request, and retrying
        when there is a failure. It yields the fixed size byte chunks, and exposes a generator
        to be consumed. This generator can be used directly in a for loop, or the entire class
        can be passed in as a file handle.
        """
        next_report_percent = self.report_every
        total_bytes = 0

        for retry in range(self.total_retries):
            if retry > 0:
                logger.error(f"Remaining retries: {self.total_retries - retry}")

            try:
                headers = {}
                if self.downloaded_bytes > 0:
                    # Pick up the download from where it was before.
                    headers = {"Range": f"bytes={self.downloaded_bytes}-"}

                self.response = requests.get(
                    self.url, headers=headers, stream=True, timeout=self.timeout_sec
                )
                self.response.raise_for_status()

                # Report the download size.
                if not total_bytes and "content-length" in self.response.headers:
                    total_bytes = int(self.response.headers["content-length"])
                    logger.info(f"Download size: {total_bytes:,} bytes")

                for chunk in self.response.iter_content(chunk_size=self.chunk_bytes):
                    if not chunk:
                        continue

                    self.downloaded_bytes += len(chunk)

                    # Report the percentage downloaded every `report_every` percentage.
                    if total_bytes and self.downloaded_bytes >= next_report_percent * total_bytes:
                        logger.info(
                            f"{self.downloaded_bytes / total_bytes * 100.0:.0f}% downloaded "
                            f"({self.downloaded_bytes}/{total_bytes} bytes)"
                        )
                        next_report_percent += self.report_every

                    yield chunk

                # The download is complete.
                self.close()
                logger.info("100% downloaded - Download finished.")
                return

            except requests.exceptions.Timeout as error:
                logger.error(f"The connection timed out: {error}.")

            except requests.exceptions.RequestException as error:
                # The RequestException is the generic error that catches all classes of "requests"
                # errors. Don't attempt to be be smart about this, just attempt again until
                # the retries are done.
                logger.error(f"A download error occurred: {error}")

            # Close out the response on an error. It will be recreated when retrying.
            if self.response:
                self.response.close()
                self.response = None

            logger.info(f"Retrying in {self.wait_before_retry_sec} sec")
            time.sleep(self.wait_before_retry_sec)

        self.close()
        raise Exception("The download failed.")

    def decode(self, byte_stream) -> Generator[bytes, None, None]:
        """Pass through the byte stream. This method can be specialized by child classes."""
        return byte_stream


@contextmanager
def _read_lines_multiple_files(
    files: list[Union[str, Path]],
    encoding: str,
    path_in_archive: Optional[str],
    on_enter_location: Optional[Callable[[str], None]] = None,
) -> Generator[Generator[str, None, None], None, None]:
    """
    Iterates through each line in multiple files, combining it into a single stream.
    """

    stack = None

    def iter(stack: ExitStack):
        for file_path in files:
            logger.info(f"Reading lines from: {file_path}")
            lines = stack.enter_context(
                read_lines(file_path, path_in_archive, on_enter_location, encoding=encoding)
            )
            yield from lines
            stack.close()

    try:
        stack = ExitStack()
        yield iter(stack)
    finally:
        if stack:
            stack.close()


@contextmanager
def _read_lines_single_file(
    location: Path | str,
    encoding: str,
    path_in_archive: Optional[str] = None,
    on_enter_location: Optional[Callable[[str], None]] = None,
) -> Generator[Generator[str, None, None], None, None]:
    """
    A smart function to efficiently stream lines from a local or remote file.
    The location can either be a URL or a local file system path.
    It handles gzip, zst, and plain text files.

    Args:
        location - URL or file path
        path_in_archive  - The path to a file in a zip archive
        on_enter_location - A lambda for when a new location is entered
    """
    location = str(location)
    if on_enter_location:
        on_enter_location(location)

    if location.startswith("http://") or location.startswith("https://"):
        # If this is mocked for a test, use the locally mocked path.
        mocked_location = get_mocked_downloads_file_path(location)
        if mocked_location:
            location = mocked_location

    stack = ExitStack()

    try:
        if location.startswith("http://") or location.startswith("https://"):
            # This is a remote file.

            response = requests.head(location, allow_redirects=True)
            content_type = response.headers.get("Content-Type")
            if content_type == "application/gzip":
                yield stack.enter_context(RemoteGzipLineStreamer(location))  # type: ignore[reportReturnType]

            elif content_type == "application/zstd":
                yield stack.enter_context(RemoteZstdLineStreamer(location))  # type: ignore[reportReturnType]

            elif content_type == "application/zip":
                raise Exception("Streaming a zip from a remote location is supported.")

            elif content_type == "text/plain":
                yield stack.enter_context(RemoteDecodingLineStreamer(location))  # type: ignore[reportReturnType]

            elif location.endswith(".gz") or location.endswith(".gzip"):
                yield stack.enter_context(RemoteGzipLineStreamer(location))  # type: ignore[reportReturnType]

            elif location.endswith(".zst"):
                yield stack.enter_context(RemoteZstdLineStreamer(location))  # type: ignore[reportReturnType]
            else:
                # Treat as plain text.
                yield stack.enter_context(RemoteDecodingLineStreamer(location))  # type: ignore[reportReturnType]

        else:  # noqa: PLR5501
            # This is a local file.
            if location.endswith(".gz") or location.endswith(".gzip"):
                yield stack.enter_context(gzip.open(location, "rt", encoding=encoding))  # type: ignore[reportReturnType]

            elif location.endswith(".zst"):
                input_file = stack.enter_context(open(location, "rb"))
                zst_reader = stack.enter_context(ZstdDecompressor().stream_reader(input_file))
                yield stack.enter_context(io.TextIOWrapper(zst_reader, encoding=encoding))  # type: ignore[reportReturnType]

            elif location.endswith(".zip"):
                if not path_in_archive:
                    raise Exception("Expected a path into the zip file.")
                zip = stack.enter_context(ZipFile(location, "r"))
                if path_in_archive not in zip.namelist():
                    raise Exception(f"Path did not exist in the zip file: {path_in_archive}")
                file = stack.enter_context(zip.open(path_in_archive, "r"))
                yield stack.enter_context(io.TextIOWrapper(file, encoding=encoding))  # type: ignore[reportReturnType]
            else:
                # Treat as plain text.
                yield stack.enter_context(open(location, "rt", encoding=encoding))  # type: ignore[reportReturnType]
    finally:
        stack.close()


def read_lines(
    location_or_locations: Union[Path, str, list[Union[str, Path]]],
    path_in_archive: Optional[str] = None,
    on_enter_location: Optional[Callable[[str], None]] = None,
    encoding="utf-8",
):
    """
    A smart function to efficiently stream lines from a local or remote file.
    The location can either be a URL or a local file system path.
    It handles gzip, zst, and plain text files.
    It can also handle a list of files.

    Args:
        location_or_locations - A single URL or file path, or a list
        path_in_archive  - The path to a file in a zip archive

    Usage:
        with read_lines("output.txt.gz") as lines:
            for line in lines:
                print(line)

        paths = [
            "http://example.com/file.txt.gz",
            "path/to/file.txt.zst",
        ]
        with read_lines(paths) as lines:
            for line in lines:
                print(line)
    """

    if isinstance(location_or_locations, list):
        return _read_lines_multiple_files(
            location_or_locations, encoding, path_in_archive, on_enter_location
        )

    return _read_lines_single_file(
        location_or_locations, encoding, path_in_archive, on_enter_location
    )


@contextmanager
def write_lines(path: Path | str, encoding="utf-8"):
    """
    A smart function to create a context to write lines to a file. It works on .zst, .gz, and
    raw text files. It reads the extension to determine the file type. If writing out a raw
    text file, for instance a sample of a dataset that is just used for viewing, include a
    "byte order mark" so that the browser can properly detect the encoding.

    with write_lines("output.txt.gz") as output:
        output.write("writing a line\n")
        output.write("writing a second lines\n")
    """

    stack = None
    try:
        path = str(path)
        stack = ExitStack()

        if path.endswith(".zst"):
            file = stack.enter_context(open(path, "wb"))
            compressor = stack.enter_context(ZstdCompressor().stream_writer(file))
            yield stack.enter_context(io.TextIOWrapper(compressor, encoding=encoding))
        elif path.endswith(".gz"):
            yield stack.enter_context(gzip.open(path, "wt", encoding=encoding))
        else:
            yield stack.enter_context(open(path, "wt", encoding=encoding))

    finally:
        if stack:
            stack.close()


def count_lines(path: Path | str) -> int:
    """
    Similar to wc -l, this counts the lines in a file. However, this command does so regardless
    of the compression strategy used on the file.
    """
    with read_lines(path) as lines:
        return sum(1 for _ in lines)


def is_file_empty(path: Path | str) -> bool:
    """
    Attempts to read a line to determine if a file is empty or not. Works on local or remote files
    as well as compressed or uncompressed files.
    """
    with read_lines(path) as lines:
        try:
            next(lines)
            return False
        except StopIteration:
            return True


def get_file_size(location: Path | str) -> int:
    """Get the size of a file, whether it is remote or local."""
    if isinstance(location, str) and (
        location.startswith("http://") or location.startswith("https://")
    ):
        return get_download_size(location)
    return os.path.getsize(location)


def get_human_readable_file_size(location: Path | str) -> tuple[str, int]:
    """Get the size of a file in a human-readable string, and the numeric bytes."""
    bytes = get_file_size(location)
    return format_bytes(bytes), bytes


def compress_file(
    path: Path | str,
    keep_original: bool = True,
    compressed_path: Optional[Path | str] = None,
    compression: Literal["zst", "gz"] = "zst",
) -> Path:
    """
    Compresses a file to .zst or .gz format. It returns the path of the compressed file.
    "zst" is the preferred compression scheme.
    """
    path = Path(path)

    if compression == "zst":
        if not compressed_path:
            compressed_path = Path(str(path) + ".zst")
        cctx = ZstdCompressor()
        with open(path, "rb") as infile:
            with open(compressed_path, "wb") as outfile:
                outfile.write(cctx.compress(infile.read()))

    elif compression == "gz":
        if not compressed_path:
            compressed_path = Path(str(path) + ".gz")
        with open(path, "rb") as infile:
            with gzip.open(compressed_path, "wb") as outfile:
                outfile.write(infile.read())

    else:
        raise ValueError(f"Unsupported compression format: {compression}")

    if not keep_original:
        # Delete the original file
        path.unlink()

    return Path(compressed_path)


def decompress_file(
    path: Union[str, Path],
    keep_original: bool = True,
    decompressed_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Decompresses a .gz or .zst file. It returns the path of the decompressed file.
    """
    path = Path(path)

    if decompressed_path:
        decompressed_path = Path(decompressed_path)
    else:
        # Remove the original suffix
        decompressed_path = path.with_suffix("")

    with ExitStack() as stack:
        decompressed_file = stack.enter_context(decompressed_path.open("wb"))

        if path.suffix == ".gz":
            compressed_file = stack.enter_context(gzip.open(str(path), "rb"))
            decompressed_file.write(compressed_file.read())
            while True:
                # Write the data out in chunks so that all of the it doesn't need to be
                # into memory.
                chunk = compressed_file.read(10_240)
                if not chunk:
                    break
                decompressed_file.write(chunk)

        elif path.suffix == ".zst":
            compressed_file = stack.enter_context(open(path, "rb"))
            for chunk in ZstdDecompressor().read_to_iter(compressed_file):
                # Write the data out in chunks so that all of the it doesn't need to be
                # into memory.
                decompressed_file.write(chunk)
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

    if not keep_original:
        # Delete the original file
        path.unlink()

    return decompressed_path


# def split_on_line_aligned_chunks(
#     file_path: Path | str, chunk_bytes: int
# ) -> Generator[bytes, None, None]:
#     """
#     Yields bytes that span chunks of the original file, aligned to line boundaries of at
#     least chunk_bytes in size.
#     """
#     file_path = Path(file_path)
#     with ExitStack() as stack:
#         base_file = stack.enter_context(file_path.open("rb"))
#         if file_path.suffix == ".zst":
#             base_file = stack.enter_context(ZstdDecompressor().stream_reader(base_file))
#         start = 0

#         while True:
#             base_file.seek(start + chunk_bytes)
#             extra = base_file.readline()

#             if not extra:
#                 # The readline returned empty, we've reached the end of the file.
#                 # Seek to end of file to determine final byte offset
#                 base_file.seek(0, 2)
#                 end = base_file.tell()
#                 if start < end:
#                     f = stack.enter_context(file_path.open("rb"))
#                     f.seek(start)
#                     yield f.read(end - start)
#                 break

#             end = base_file.tell()
#             f = stack.enter_context(file_path.open("rb"))
#             f.seek(start)
#             yield f.read(end - start)
#             start = end


def split_on_line_aligned_chunks(
    file_path: Path | str, chunk_bytes: int
) -> Generator[bytes, None, None]:
    """
    Yields slices of a preallocated byte buffer containing line-aligned chunks
    of at least `chunk_bytes` in size. Supports .zst files via streaming.
    """
    file_path = Path(file_path)
    with ExitStack() as stack:
        raw_file = stack.enter_context(file_path.open("rb"))
        if file_path.suffix == ".zst":
            # Wrap the zst stream in a buffered reader to perform "readline" on it.
            stream = stack.enter_context(
                io.BufferedReader(
                    ZstdDecompressor().stream_reader(raw_file)  # type: ignore[reportArgumentType]
                )
            )
        else:
            stream = raw_file

        buffer = bytearray(chunk_bytes)
        offset = 0

        while True:
            line = stream.readline()
            if not line:
                # The readline returned empty, we've reached the end of the file.
                if offset > 0:
                    # There is a final text string to yield.
                    yield bytes(buffer[:offset])
                break

            line_len = len(line)

            if offset + line_len > chunk_bytes:
                # Buffer is full, copy out a slice of the buffer and yield it.
                yield bytes(buffer[:offset])
                offset = 0

            assert (
                line_len <= chunk_bytes
            ), "A line was found that was too large for the chunking buffer."

            # Store the line in the buffer.
            buffer[offset : offset + line_len] = line
            offset += line_len


def split_on_parallel_line_aligned_chunks(
    src_path: Path | str, trg_path: Path | str, chunk_bytes: int
) -> Generator[tuple[bytes, bytes], None, None]:
    """
    Yields byte chunks for the source and target corpora. The byte chunks will both
    contain the same number of sentences. Each chunk will be at least `chunk_bytes`
    in the source stream. Chunks are aligned by line and end on line boundaries.
    """
    src_path = Path(src_path)
    trg_path = Path(trg_path)

    with ExitStack() as stack:
        src_raw = stack.enter_context(src_path.open("rb"))
        trg_raw = stack.enter_context(trg_path.open("rb"))

        if src_path.suffix == ".zst":
            src_stream = stack.enter_context(ZstdDecompressor().stream_reader(src_raw))
        else:
            src_stream = src_raw

        if trg_path.suffix == ".zst":
            trg_stream = stack.enter_context(ZstdDecompressor().stream_reader(trg_raw))
        else:
            trg_stream = trg_raw

        src_buffer = bytearray(chunk_bytes)
        trg_buffer = bytearray(chunk_bytes)
        src_offset = 0
        trg_offset = 0
        line_count = 0

        while True:
            src_line = src_stream.readline()
            trg_line = trg_stream.readline()

            if not src_line and not trg_line:
                if src_offset > 0:
                    yield bytes(src_buffer[:src_offset]), bytes(trg_buffer[:trg_offset])
                break

            assert src_line and trg_line, "Source and target files are misaligned."

            src_len = len(src_line)
            trg_len = len(trg_line)

            assert src_len <= chunk_bytes, "The source line is too long for buffer."
            assert trg_len <= chunk_bytes, "The target line is too long for buffer."

            if src_offset + src_len > chunk_bytes:
                yield bytes(src_buffer[:src_offset]), bytes(trg_buffer[:trg_offset])
                src_offset = 0
                trg_offset = 0

            src_buffer[src_offset : src_offset + src_len] = src_line
            trg_buffer[trg_offset : trg_offset + trg_len] = trg_line
            src_offset += src_len
            trg_offset += trg_len
            line_count += 1


T = TypeVar("T")


def bytes_to_lines(buf: bytes) -> Generator[str, None, None]:
    """
    Converts a bytes object into a generator of UTF-8 decoded lines.
    """
    stream = io.BytesIO(buf)
    for line in stream:
        yield line.decode("utf-8")


def multiprocess_mono_lines(
    file: Path | str,
    on_chunk: Callable[[Generator[str, None, None]], T],
    on_chunk_done: Callable[[T], None],
    chunk_bytes: int = 10_000_000,
    processes: int = multiprocessing.cpu_count(),
):
    """
    Splits a large text or .zst-compressed file into line-aligned byte chunks,
    processes them in parallel using multiple processes, and handles results in
    submission order.
    """
    file = Path(file)
    chunk_gen = split_on_line_aligned_chunks(file, chunk_bytes)

    with multiprocessing.Pool(processes=processes) as pool:

        def chunk_wrapper(chunk: bytes, on_chunk: Callable[[Generator[str, None, None]], T]) -> T:
            return on_chunk(bytes_to_lines(chunk))

        results = pool.imap(lambda c: chunk_wrapper(c, on_chunk), chunk_gen)
        for result in results:
            if on_chunk_done:
                on_chunk_done(result)


def multiprocess_parallel_lines(
    src_file: Path | str,
    trg_file: Path | str,
    on_chunk: Callable[[Generator[str, None, None], Generator[str, None, None]], T],
    on_chunk_done: Callable[[T], None],
    chunk_bytes: int = 10_000_000,
    processes: int = multiprocessing.cpu_count(),
):
    """
    Splits two aligned source and target text or .zst-compressed files into parallel
    line-aligned byte chunks, processes them in parallel using multiple processes,
    and handles results in submission order.
    """

    src_file = Path(src_file)
    trg_file = Path(trg_file)
    chunk_gen = split_on_parallel_line_aligned_chunks(src_file, trg_file, chunk_bytes)

    def chunk_wrapper(pair: tuple[bytes, bytes]) -> T:
        src_chunk, trg_chunk = pair
        return on_chunk(bytes_to_lines(src_chunk), bytes_to_lines(trg_chunk))

    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.imap(chunk_wrapper, chunk_gen)
        for result in results:
            if on_chunk_done:
                on_chunk_done(result)


def stream_input(command: list[str], lines: Iterable[str]) -> list[str]:
    """
    Runs a subprocess by streaming input and handling the output in a separate thread.
    This handles issues with backpressure when the upstream task is faster than the
    downstream task.
    """
    proc = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
    )

    assert proc.stdin is not None
    assert proc.stdout is not None

    # Read from stdout in a separate thread to not block, and to avoid back pressure.
    results: list[str] = []

    def reader():
        assert proc.stdout is not None
        for line in proc.stdout:
            results.append(line)

    thread = threading.Thread(target=reader)
    thread.start()

    for line in lines:
        proc.stdin.write(line)

    proc.stdin.close()
    thread.join()
    proc.stdout.close()
    proc.wait()

    return results

import gc
import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, FILES, MAX_ROWS, PROCESSED_DIR
from src.preprocessor import get_cleaning_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def rebuild_cleaned_files(force: bool = False, chunksize: int = 500_000):
    """
    Rebuild per-file cleaned parquet outputs:
      data/processed/cleaned_0.parquet ... cleaned_N.parquet

    - force=False: skip files that already exist
    - force=True: overwrite all
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    built = 0
    skipped = 0
    failed = 0

    for i, file_name in enumerate(FILES):
        src_path = DATA_DIR / file_name
        dst_path = PROCESSED_DIR / f"cleaned_{i}.parquet"

        if not src_path.exists():
            logger.warning(f"[{i}] Missing source: {src_path}")
            failed += 1
            continue

        if dst_path.exists() and not force:
            logger.info(f"[{i}] Exists, skipping: {dst_path.name}")
            skipped += 1
            continue

        logger.info(f"[{i}] Processing: {file_name}")
        pipeline = get_cleaning_pipeline()  # fresh pipeline per file

        chunks = []
        rows_this_file = 0

        try:
            for chunk in pd.read_csv(src_path, chunksize=chunksize, low_memory=False):
                cleaned = pipeline.fit_transform(chunk)  # rule-based transforms
                chunks.append(cleaned)
                rows_this_file += len(cleaned)

                if MAX_ROWS and (total_rows + rows_this_file) >= MAX_ROWS:
                    logger.info(f"Reached MAX_ROWS={MAX_ROWS}. Truncating build.")
                    break

                del chunk
                gc.collect()

            if not chunks:
                logger.warning(f"[{i}] No data produced after cleaning.")
                failed += 1
                continue

            file_df = pd.concat(chunks, ignore_index=True)

            if MAX_ROWS and len(file_df) + total_rows > MAX_ROWS:
                file_df = file_df.iloc[: MAX_ROWS - total_rows]

            file_df.to_parquet(dst_path, index=False)
            total_rows += len(file_df)
            built += 1

            logger.info(f"[{i}] Saved {dst_path.name} | rows={len(file_df)}")

            del file_df
            del chunks
            gc.collect()

            if MAX_ROWS and total_rows >= MAX_ROWS:
                logger.info(f"Global MAX_ROWS reached ({MAX_ROWS}). Stopping.")
                break

        except Exception as e:
            logger.exception(f"[{i}] Failed processing {file_name}: {e}")
            failed += 1
            continue

    logger.info("=== Rebuild complete ===")
    logger.info(f"Built: {built}, Skipped: {skipped}, Failed: {failed}, Total rows: {total_rows}")


if __name__ == "__main__":
    # Set force=True to overwrite all cleaned files
    rebuild_cleaned_files(force=False, chunksize=500_000)
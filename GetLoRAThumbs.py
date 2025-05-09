import os
import json
import struct
import requests
from PIL import Image
import io
import gradio as gr
from pathlib import Path
import logging
import errno
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
import hashlib
import sqlite3
import base64
import shutil

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log Gradio version for debugging
logger.info(f"Gradio version: {gr.__version__}")

# Optional: Set Civitai API key
CIVITAI_API_KEY = os.getenv("CIVITAI_API_KEY", "")

def init_thumbnail_database(db_path):
    """Initialize SQLite database and ensure blacked_out column exists."""
    logger.debug(f"Initializing thumbnail database at {db_path}")
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            if cursor.fetchone()[0] != 'ok':
                logger.warning(f"Database integrity check failed for {db_path}. Recreating database.")
                conn.close()
                Path(db_path).unlink(missing_ok=True)
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        CREATE TABLE thumbnails (
                            filename TEXT PRIMARY KEY,
                            model_url TEXT,
                            file_hash TEXT,
                            mtime REAL,
                            blacked_out INTEGER DEFAULT 0,
                            metadata TEXT
                        )
                    """)
                    conn.commit()
            else:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS thumbnails (
                        filename TEXT PRIMARY KEY,
                        model_url TEXT,
                        file_hash TEXT,
                        mtime REAL,
                        blacked_out INTEGER DEFAULT 0,
                        metadata TEXT
                    )
                """)
                # Check if metadata column exists
                cursor.execute("PRAGMA table_info(thumbnails)")
                columns = [col[1] for col in cursor.fetchall()]
                if 'metadata' not in columns:
                    logger.info(f"Adding metadata column to thumbnails table in {db_path}")
                    cursor.execute("ALTER TABLE thumbnails ADD COLUMN metadata TEXT")
                conn.commit()
            logger.debug("Thumbnail database initialized")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database {db_path}: {e}")
        raise

def save_thumbnail_to_filesystem(output_dir, cache_dir, filename, image, model_url=None, db_path=None, file_hash=None, mtime=None, blacked_out=0, metadata=None):
    """Save thumbnail to filesystem and metadata to database."""
    logger.debug(f"Saving thumbnail for {filename} to {output_dir}")
    try:
        thumbnail_path = output_dir / f"{filename}.png"
        image.save(thumbnail_path, format='PNG')
        logger.info(f"Saved thumbnail for {filename} to {thumbnail_path}")
        
        if db_path:
            metadata_json = json.dumps(metadata, indent=2) if metadata else "{}"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO thumbnails (filename, model_url, file_hash, mtime, blacked_out, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (filename, model_url, file_hash, mtime, blacked_out, metadata_json))
                conn.commit()
                logger.debug(f"Saved metadata for {filename} to database with model_url: {model_url}, hash: {file_hash}, blacked_out: {blacked_out}, metadata: {metadata_json}")
        return None
    except Exception as e:
        logger.error(f"Error saving thumbnail for {filename}: {e}")
        return f"Error saving thumbnail for {filename}: {e}"

def get_thumbnail_from_filesystem(output_dir, db_path, filename):
    """Retrieve thumbnail from filesystem and metadata from database."""
    logger.debug(f"Checking {output_dir} for thumbnail of {filename}")
    try:
        thumbnail_path = output_dir / f"{filename}.png"
        model_url = None
        file_hash = None
        stored_mtime = None
        blacked_out = 0
        metadata = {}
        if thumbnail_path.exists():
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT model_url, file_hash, mtime, blacked_out, metadata FROM thumbnails WHERE filename = ?", (filename,))
                result = cursor.fetchone()
                if result:
                    model_url, file_hash, stored_mtime, blacked_out, metadata_json = result
                    try:
                        metadata = json.loads(metadata_json) if metadata_json else {}
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse metadata JSON for {filename}")
                        metadata = {}
            try:
                thumbnail = Image.open(thumbnail_path)
                logger.debug(f"Found thumbnail for {filename} at {thumbnail_path} with model_url: {model_url}, hash: {file_hash}, blacked_out: {blacked_out}, metadata: {metadata}")
                return None, thumbnail, model_url, file_hash, stored_mtime, blacked_out, metadata
            except Exception as e:
                logger.error(f"Error opening thumbnail {thumbnail_path}: {e}")
                return f"Error opening thumbnail {thumbnail_path}: {e}", None, None, None, None, 0, {}
        logger.debug(f"No thumbnail found for {filename} at {thumbnail_path}")
        return None, None, None, None, None, 0, {}
    except Exception as e:
        logger.error(f"Error retrieving thumbnail for {filename}: {e}")
        return f"Error retrieving thumbnail for {filename}: {e}", None, None, None, None, 0, {}

def create_black_thumbnail(size=(200, 200)):
    """Create a black 200x200 thumbnail."""
    image = Image.new('RGB', size, color='black')
    return image

def read_safetensors_metadata(file_path):
    """Read metadata from a safetensors file."""
    logger.debug(f"Attempting to read metadata from {file_path}")
    try:
        with open(file_path, 'rb') as f:
            data = f.read(8)
            if len(data) < 8:
                logger.error(f"File {file_path} is too short to read header")
                return f"Error reading {file_path}: File too short", None
            header_size = struct.unpack('<Q', data)[0]
            logger.debug(f"Header size: {header_size} bytes")
            header_data = f.read(header_size)
            if len(header_data) < header_size:
                logger.error(f"File {file_path} has incomplete header")
                return f"Error reading {file_path}: Incomplete header", None
            try:
                header_str = header_data.decode('utf-8')
            except UnicodeDecodeError:
                header_str = header_data.decode('utf-8', errors='replace')
            try:
                header_json = json.loads(header_str)
                metadata = header_json.get("__metadata__", {})
                if not isinstance(metadata, dict):
                    logger.warning(f"Metadata in {file_path} is not a dictionary: {metadata}")
                    metadata = {"raw_metadata": str(metadata)}
                metadata = {k: str(v) for k, v in metadata.items()}
                logger.info(f"Parsed metadata from header in {file_path}: {metadata}")
                return metadata
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON header in {file_path}: {e}")
                return {}
    except Exception as e:
        logger.error(f"Error reading safetensors metadata from {file_path}: {e}")
        return f"Error reading {file_path}: {e}", None

def black_out_thumbnail(directory, selected_file, output_text, file_list_state):
    """Black out a thumbnail, saving original to cache."""
    logger.info(f"Blacking out thumbnail for {selected_file} in {directory}")
    try:
        if not selected_file:
            logger.warning("No file selected in gallery for black out")
            return f"{output_text}\nError: No file selected for black out. Please click an image from the gallery.", gr.update(), file_list_state, None
        if selected_file not in file_list_state:
            logger.error(f"Selected file {selected_file} not in available files")
            return f"{output_text}\nError: Selected file {selected_file} not in available files", gr.update(), file_list_state, None

        directory_path = Path(directory)
        db_path = directory_path / "thumbnails.db"
        output_dir = directory_path / "thumbnails"
        cache_dir = directory_path / "thumbnails" / "cache"
        output_dir.mkdir(exist_ok=True)
        cache_dir.mkdir(exist_ok=True)

        file_path = directory_path / selected_file
        if not file_path.exists():
            logger.error(f"File {file_path} does not exist")
            return f"{output_text}\nError: File {file_path} does not exist", gr.update(), file_list_state, None

        filename_base = selected_file.rsplit('.safetensors', 1)[0]
        error, thumbnail, model_url, file_hash, stored_mtime, blacked_out, metadata = get_thumbnail_from_filesystem(output_dir, db_path, filename_base)
        if error:
            logger.error(error)
            return f"{output_text}\n{error}", gr.update(), file_list_state, None
        if not thumbnail:
            logger.warning(f"No thumbnail available for {filename_base}")
            return f"{output_text}\nNo thumbnail available for {filename_base}", gr.update(), file_list_state, None
        if blacked_out:
            logger.info(f"Thumbnail for {filename_base} is already blacked out")
            return f"{output_text}\nThumbnail for {filename_base} is already blacked out", gr.update(), file_list_state, None

        cache_path = cache_dir / f"{filename_base}.png"
        thumbnail.save(cache_path, format='PNG')
        logger.debug(f"Saved original thumbnail to {cache_path}")

        black_thumbnail = create_black_thumbnail()
        error = save_thumbnail_to_filesystem(output_dir, cache_dir, filename_base, black_thumbnail, model_url, db_path, file_hash, stored_mtime, blacked_out=1, metadata=metadata)
        if error:
            logger.error(error)
            return f"{output_text}\n{error}", gr.update(), file_list_state, None

        all_display_items = []
        for f in sorted(directory_path.glob("*.safetensors")):
            error, thumb, url, _, _, blk, meta = get_thumbnail_from_filesystem(output_dir, db_path, f.name.rsplit('.safetensors', 1)[0])
            all_display_items.append((f.name, thumb, url, blk, meta))
        html_result = generate_html_page(all_display_items, directory_path)
        gallery_items = [
            (
                resize_to_fit(img, 200),
                f"{name} {'(Blacked Out)' if blk else ''}\n---\nMetadata:\n" + (
                    '\n'.join([f"{k}: {v}" for k, v in meta.items()]) if meta else "No metadata available"
                )
            )
            for name, img, _, blk, meta in all_display_items if img is not None
        ]

        logger.info(f"Blacked out thumbnail for {filename_base}")
        return f"{output_text}\nBlacked out thumbnail for {filename_base}\n{html_result}", gr.update(value=gallery_items), file_list_state, None
    except Exception as e:
        logger.error(f"Error blacking out thumbnail for {selected_file}: {e}")
        return f"{output_text}\nError blacking out thumbnail for {selected_file}: {e}", gr.update(), file_list_state, None

def restore_thumbnail(directory, selected_file, output_text, file_list_state):
    """Restore a blacked-out thumbnail from cache or re-fetch."""
    logger.info(f"Restoring thumbnail for {selected_file} in {directory}")
    try:
        if not selected_file:
            logger.warning("No file selected in gallery for restore")
            return f"{output_text}\nError: No file selected for restore. Please click an image from the gallery.", gr.update(), file_list_state, None
        if selected_file not in file_list_state:
            logger.error(f"Selected file {selected_file} not in available files")
            return f"{output_text}\nError: Selected file {selected_file} not in available files", gr.update(), file_list_state, None

        directory_path = Path(directory)
        db_path = directory_path / "thumbnails.db"
        output_dir = directory_path / "thumbnails"
        cache_dir = output_dir / "cache"
        file_path = directory_path / selected_file

        if not file_path.exists():
            logger.error(f"File {file_path} does not exist")
            return f"{output_text}\nError: File {file_path} does not exist", gr.update(), file_list_state, None

        filename_base = selected_file.rsplit('.safetensors', 1)[0]
        error, thumbnail, model_url, file_hash, stored_mtime, blacked_out, metadata = get_thumbnail_from_filesystem(output_dir, db_path, filename_base)
        if error:
            logger.error(error)
            return f"{output_text}\n{error}", gr.update(), file_list_state, None
        if not blacked_out:
            logger.info(f"Thumbnail for {filename_base} is not blacked out")
            return f"{output_text}\nThumbnail for {filename_base} is not blacked out", gr.update(), file_list_state, None

        cache_path = cache_dir / f"{filename_base}.png"
        if cache_path.exists():
            logger.debug(f"Restoring thumbnail from cache: {cache_path}")
            restored_thumbnail = Image.open(cache_path)
            error = save_thumbnail_to_filesystem(output_dir, cache_dir, filename_base, restored_thumbnail, model_url, db_path, file_hash, stored_mtime, blacked_out=0, metadata=metadata)
            if error:
                logger.error(error)
                return f"{output_text}\n{error}", gr.update(), file_list_state, None
            logger.info(f"Restored thumbnail for {filename_base} from cache")
        else:
            logger.debug(f"No cached thumbnail found for {filename_base}, re-fetching")
            metadata = read_safetensors_metadata(file_path)
            if isinstance(metadata, tuple):
                logger.warning(f"Metadata read failed: {metadata[0]}")
                return f"{output_text}\n{metadata[0]}", gr.update(), file_list_state, None

            error, thumbnail = extract_embedded_thumbnail(metadata, file_path)
            if thumbnail:
                error = save_thumbnail_to_filesystem(output_dir, cache_dir, filename_base, thumbnail, model_url, db_path, file_hash, stored_mtime, blacked_out=0, metadata=metadata)
                if error:
                    return f"{output_text}\n{error}", gr.update(), file_list_state, None
                logger.info(f"Restored thumbnail for {filename_base} from embedded metadata")
            else:
                thumbnail_url = metadata.get('ss_thumbnail_url')
                if thumbnail_url:
                    error, thumbnail = get_thumbnail_from_url(thumbnail_url)
                    if thumbnail:
                        error = save_thumbnail_to_filesystem(output_dir, cache_dir, filename_base, thumbnail, model_url, db_path, file_hash, stored_mtime, blacked_out=0, metadata=metadata)
                        if error:
                            return f"{output_text}\n{error}", gr.update(), file_list_state, None
                        logger.info(f"Restored thumbnail for {filename_base} from URL")
                    else:
                        logger.warning(f"Failed to fetch thumbnail from URL: {thumbnail_url}")
                else:
                    hashes = []
                    if file_hash:
                        hashes.append(('file_sha256', file_hash))
                    if metadata.get('ss_new_sd_model_hash'):
                        hashes.append(('ss_new_sd_model_hash', metadata['ss_new_sd_model_hash']))
                    if metadata.get('ss_sd_model_hash'):
                        hashes.append(('ss_sd_model_hash', metadata['ss_sd_model_hash']))
                    if hashes:
                        error, thumbnail, new_model_url = search_thumbnail_online(hashes, filename_base, file_path)
                        if thumbnail:
                            model_url = new_model_url or model_url
                            error = save_thumbnail_to_filesystem(output_dir, cache_dir, filename_base, thumbnail, model_url, db_path, file_hash, stored_mtime, blacked_out=0, metadata=metadata)
                            if error:
                                return f"{output_text}\n{error}", gr.update(), file_list_state, None
                            logger.info(f"Restored thumbnail for {filename_base} from online search")
                        else:
                            logger.warning(f"No thumbnail found online for {filename_base}")
                            return f"{output_text}\nNo thumbnail found to restore for {filename_base}", gr.update(), file_list_state, None

        all_display_items = []
        for f in sorted(directory_path.glob("*.safetensors")):
            error, thumb, url, _, _, blk, meta = get_thumbnail_from_filesystem(output_dir, db_path, f.name.rsplit('.safetensors', 1)[0])
            all_display_items.append((f.name, thumb, url, blk, meta))
        html_result = generate_html_page(all_display_items, directory_path)
        gallery_items = [
            (
                resize_to_fit(img, 200),
                f"{name} {'(Blacked Out)' if blk else ''}\n---\nMetadata:\n" + (
                    '\n'.join([f"{k}: {v}" for k, v in meta.items()]) if meta else "No metadata available"
                )
            )
            for name, img, _, blk, meta in all_display_items if img is not None
        ]

        logger.info(f"Restored thumbnail for {filename_base}")
        return f"{output_text}\nRestored thumbnail for {filename_base}\n{html_result}", gr.update(value=gallery_items), file_list_state, None
    except Exception as e:
        logger.error(f"Error restoring thumbnail for {selected_file}: {e}")
        return f"{output_text}\nError restoring thumbnail for {selected_file}: {e}", gr.update(), file_list_state, None

def clear_old_thumbnails(directory, output_text):
    """Remove thumbnails and database entries for files no longer in the directory."""
    logger.info(f"Clearing old thumbnails in {directory}")
    try:
        directory_path = Path(directory)
        db_path = directory_path / "thumbnails.db"
        output_dir = directory_path / "thumbnails"
        cache_dir = output_dir / "cache"
        output_dir.mkdir(exist_ok=True)
        cache_dir.mkdir(exist_ok=True)

        current_files = {f.name for f in directory_path.glob("*.safetensors")}
        
        init_thumbnail_database(db_path)
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filename FROM thumbnails")
            db_thumbnails = {row[0] for row in cursor.fetchall()}
        
        fs_thumbnails = {f.stem for f in output_dir.glob("*.png")}
        cached_thumbnails = {f.stem for f in cache_dir.glob("*.png")}
        
        removed_count = 0
        for filename in db_thumbnails:
            if filename not in {f.rsplit('.safetensors', 1)[0] for f in current_files}:
                cursor.execute("DELETE FROM thumbnails WHERE filename = ?", (filename,))
                logger.debug(f"Removed {filename} from database")
                removed_count += 1
        
        for filename in fs_thumbnails:
            if filename not in {f.rsplit('.safetensors', 1)[0] for f in current_files}:
                thumbnail_path = output_dir / f"{filename}.png"
                thumbnail_path.unlink(missing_ok=True)
                logger.debug(f"Removed {filename}.png from filesystem")
                removed_count += 1
        
        for filename in cached_thumbnails:
            if filename not in {f.rsplit('.safetensors', 1)[0] for f in current_files}:
                cache_path = cache_dir / f"{filename}.png"
                cache_path.unlink(missing_ok=True)
                logger.debug(f"Removed {filename}.png from cache")
                removed_count += 1
        
        conn.commit()
        
        all_display_items = []
        for filename in sorted(current_files):
            error, thumbnail, model_url, _, _, blacked_out, metadata = get_thumbnail_from_filesystem(output_dir, db_path, filename.rsplit('.safetensors', 1)[0])
            all_display_items.append((filename, thumbnail, model_url, blacked_out, metadata))
        html_result = generate_html_page(all_display_items, directory_path)
        gallery_items = [
            (
                resize_to_fit(img, 200),
                f"{name} {'(Blacked Out)' if blk else ''}\n---\nMetadata:\n" + (
                    '\n'.join([f"{k}: {v}" for k, v in meta.items()]) if meta else "No metadata available"
                )
            )
            for name, img, _, blk, meta in all_display_items if img is not None
        ]

        logger.info(f"Cleared {removed_count} old thumbnails")
        return f"{output_text}\nCleared {removed_count} old thumbnails\n{html_result}", gr.update(value=gallery_items), list(sorted(current_files)), None
    except Exception as e:
        logger.error(f"Error clearing old thumbnails: {e}")
        return f"{output_text}\nError clearing old thumbnails: {e}", gr.update(), [], None

def compute_file_sha256(file_path):
    """Compute SHA256 hash of a file."""
    logger.debug(f"Computing SHA256 hash for {file_path}")
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        file_hash = sha256_hash.hexdigest()
        logger.debug(f"Computed SHA256 hash: {file_hash}")
        return file_hash
    except Exception as e:
        logger.error(f"Error computing SHA256 hash for {file_path}: {e}")
        return None

def extract_embedded_thumbnail(metadata, file_path):
    """Check for embedded thumbnail in metadata."""
    logger.debug(f"Checking for embedded thumbnail in {file_path}")
    thumbnail_data = metadata.get('ss_thumbnail')
    if thumbnail_data:
        logger.debug("Found embedded thumbnail data")
        try:
            img_data = base64.b64decode(thumbnail_data)
            logger.debug(f"Decoded thumbnail data size: {len(img_data)} bytes")
            return None, Image.open(io.BytesIO(img_data))
        except Exception as e:
            logger.error(f"Error decoding thumbnail for {file_path}: {e}")
            return f"Error decoding thumbnail for {file_path}: {e}", None
    logger.debug("No embedded thumbnail found")
    return None, None

@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, json.JSONDecodeError))
)
def get_thumbnail_from_url(url):
    """Download thumbnail from a URL with retries."""
    logger.debug(f"Attempting to download thumbnail from {url}")
    headers = {"Authorization": f"Bearer {CIVITAI_API_KEY}"} if CIVITAI_API_KEY else {}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        logger.debug(f"HTTP response status: {response.status_code}")
        if response.status_code == 200:
            return None, Image.open(io.BytesIO(response.content))
        else:
            logger.warning(f"Failed to download thumbnail from {url}: Status {response.status_code}")
            return f"Failed to download thumbnail from {url}: Status {response.status_code}", None
    except Exception as e:
        logger.error(f"Error downloading thumbnail from {url}: {e}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, json.JSONDecodeError))
)
def search_thumbnail_online(hashes, file_name, file_path):
    """Search for thumbnail and model URL online using hashes."""
    headers = {"Authorization": f"Bearer {CIVITAI_API_KEY}"} if CIVITAI_API_KEY else {}
    for hash_type, model_hash in hashes:
        civitai_model_api = f"https://civitai.com/api/v1/model-versions/by-hash/{model_hash}"
        logger.debug(f"Searching online for thumbnail with {hash_type} hash {model_hash}")
        try:
            response = requests.get(civitai_model_api, headers=headers, timeout=5)
            logger.debug(f"Model API response status: {response.status_code}")
            if response.status_code != 200:
                logger.warning(f"Model API request failed for {hash_type} hash {model_hash}: Status {response.status_code}")
                continue

            data = response.json()
            model_id = data.get('modelId')
            if not model_id:
                logger.debug(f"No modelId found for {hash_type} hash")
                continue

            model_url = f"https://civitai.com/models/{model_id}"
            logger.debug(f"Constructed model URL: {model_url}")

            if data.get('images') and len(data['images']) > 0:
                image_url = data['images'][0].get('url')
                logger.debug(f"Found image URL: {image_url}")
                if image_url:
                    error, thumbnail = get_thumbnail_from_url(image_url)
                    if thumbnail:
                        return error, thumbnail, model_url
                    else:
                        logger.debug(f"Failed to retrieve thumbnail for {image_url}")

            civitai_images_api = f"https://civitai.com/api/v1/images?modelId={model_id}&nsfw=X"
            logger.debug(f"Falling back to images API: {civitai_images_api}")
            response = requests.get(civitai_images_api, headers=headers, timeout=5)
            logger.debug(f"Images API response status: {response.status_code}")
            if response.status_code == 200:
                images_data = response.json()
                if images_data.get('items') and len(images_data['items']) > 0:
                    image_url = images_data['items'][0].get('url')
                    logger.debug(f"Found image URL in images API: {image_url}")
                    if image_url:
                        error, thumbnail = get_thumbnail_from_url(image_url)
                        if thumbnail:
                            return error, thumbnail, model_url
                logger.debug(f"No items found in images API response")
            else:
                logger.warning(f"Images API request failed: Status {response.status_code}")

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {hash_type} hash {model_hash}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error searching online for {hash_type} hash {model_hash}: {e}")
            continue

    logger.debug(f"No thumbnail or model URL found for {file_name}")
    return f"No thumbnail found for {file_name}", None, None

def generate_html_page(items, output_dir):
    """Generate a static HTML page with thumbnails, metadata, and Civitai links in four columns."""
    logger.debug(f"Generating HTML page at {output_dir / 'thumbnails.html'}")
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LoRA and Model Thumbnails</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #1A1A1A;
                color: #E0E0E0;
                margin: 0;
                padding: 20px;
                padding-top: 20px;
                min-height: 100vh;
                text-align: center;
            }
            h1 {
                color: #E0E0E0;
                text-align: center;
                margin-top: 0;
                margin-bottom: 20px;
            }
            .thumbnail-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 200px;
                width: 100%;
                max-width: 1600px;
                margin: 0 auto;
                justify-items: center;
                align-items: start;
            }
            .thumbnail-container {
                margin: 0;
                text-align: center;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .thumbnail-container h3 {
                margin: 10px 0;
                color: #E0E0E0;
                word-break: break-word;
                max-width: 200px;
            }
            .thumbnail-container img {
                max-width: 400px;
                max-height: 400px;
                object-fit: contain;
                border: 1px solid #4A4A4A;
                border-radius: 4px;
            }
            .thumbnail-container a img:hover {
                opacity: 0.8;
                cursor: pointer;
            }
            .thumbnail-container p {
                color: #A0A0A0;
                margin: 5px 0;
            }
            .thumbnail-container a {
                text-decoration: none;
                color: #66B0FF;
            }
            .thumbnail-container a:hover {
                text-decoration: underline;
                color: #99C7FF;
            }
            .metadata-box {
                max-width: 400px;
                max-height: 200px;
                overflow-y: auto;
                background-color: #2A2A2A;
                padding: 10px;
                border: 1px solid #4A4A4A;
                border-radius: 4px;
                text-align: left;
                font-size: 12px;
                color: #E0E0E0;
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
        <h1>LoRA and Model Thumbnails</h1>
        <div class="thumbnail-grid">
    """
    
    for filename, img, model_url, _, metadata in items:
        html_content += '<div class="thumbnail-container">'
        html_content += f'<h3>{filename}</h3>'
        if img:
            thumbnail_path = f"thumbnails/{filename.rsplit('.safetensors', 1)[0]}.png"
            if model_url:
                html_content += f'<a href="{model_url}" target="_blank">'
                html_content += f'<img src="{thumbnail_path}" alt="{filename}" title="Click to view on Civitai">'
                html_content += '</a>'
            else:
                html_content += f'<img src="{thumbnail_path}" alt="{filename}">'
        else:
            html_content += '<p>No thumbnail available</p>'
            if model_url:
                html_content += f'<p><a href="{model_url}" target="_blank">View on Civitai</a></p>'
        metadata_text = "\n".join([f"{k}: {v}" for k, v in metadata.items()]) if metadata else "No metadata available"
        html_content += f'<div class="metadata-box">{metadata_text}</div>'
        html_content += '</div>'
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    try:
        with open(output_dir / 'thumbnails.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Saved to {output_dir}\\thumbnails.html")
        return f"Saved to {output_dir}\\thumbnails.html"
    except Exception as e:
        logger.error(f"Error saving thumbnails.html: {e}")
        return f"Error saving thumbnails.html: {e}"

def resize_to_fit(image, max_size):
    """Resize image to fit within max_size while preserving aspect ratio."""
    logger.debug(f"Resizing image to fit within {max_size}x{max_size}")
    if image.size[0] > max_size or image.size[1] > max_size:
        ratio = min(max_size / image.size[0], max_size / image.size[1])
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        logger.debug(f"Resized from {image.size} to {new_size}")
        return resized_image
    return image

def process_lora_directory(directory, output_text, progress=gr.Progress()):
    """Process all LoRA and base model safetensors files."""
    logger.info(f"Starting processing for directory: {directory}")
    if not os.path.isdir(directory):
        logger.error(f"Invalid directory: {directory} - {os.strerror(errno.errorcode.get(errno.EACCES, 'Unknown error'))}")
        return f"{output_text}\nError: {directory} is not a valid directory", [], gr.update(), None

    db_path = Path(directory) / "thumbnails.db"
    init_thumbnail_database(db_path)
    output_dir = Path(directory) / "thumbnails"
    cache_dir = output_dir / "cache"
    output_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    all_results = []
    all_display_items = []

    files = sorted(Path(directory).glob("*.safetensors"))
    total_files = len(files)
    logger.debug(f"Found {total_files} safetensors files")

    for i, file_path in enumerate(progress.tqdm(files, desc="Processing files")):
        logger.info(f"Processing file: {file_path.name}")
        result = f"Processing {file_path.name}\n"
        thumbnail = None
        model_url = None
        file_hash = None
        mtime = os.path.getmtime(file_path)
        metadata = None

        filename_base = file_path.name.rsplit('.safetensors', 1)[0]
        error, thumbnail, model_url, stored_hash, stored_mtime, blacked_out, metadata = get_thumbnail_from_filesystem(output_dir, db_path, filename_base)
        if error:
            result += error + "\n"
        if thumbnail and stored_mtime == mtime and not blacked_out:
            result += f"Retrieved thumbnail for {file_path.name} from filesystem (cached)\n"
            if model_url:
                result += f"Retrieved Civitai model page: {model_url}\n"
            file_hash = stored_hash
            all_results.append(result)
            all_display_items.append((file_path.name, thumbnail, model_url, blacked_out, metadata))
            continue

        metadata = read_safetensors_metadata(file_path)
        if isinstance(metadata, tuple):
            logger.warning(f"Metadata read failed: {metadata[0]}")
            result += metadata[0] + "\n"
            all_results.append(result)
            all_display_items.append((file_path.name, None, None, 0, {}))
            continue
        else:
            result += f"Metadata read successfully\n"
            logger.debug("Metadata read successfully")

        error, thumbnail = extract_embedded_thumbnail(metadata, file_path)
        if error:
            result += error + "\n"
        if thumbnail:
            file_hash = compute_file_sha256(file_path) if not file_hash else file_hash
            error = save_thumbnail_to_filesystem(output_dir, cache_dir, filename_base, thumbnail, model_url, db_path, file_hash, mtime, blacked_out=0, metadata=metadata)
            if error:
                result += error + "\n"
            else:
                result += f"Saved thumbnail for {file_path.name} to filesystem\n"
            all_results.append(result)
            all_display_items.append((file_path.name, thumbnail, model_url, 0, metadata))
            continue

        thumbnail_url = metadata.get('ss_thumbnail_url')
        if thumbnail_url:
            logger.debug(f"Found thumbnail URL in metadata: {thumbnail_url}")
            error, thumbnail = get_thumbnail_from_url(thumbnail_url)
            if error:
                result += error + "\n"
            if thumbnail:
                file_hash = compute_file_sha256(file_path) if not file_hash else file_hash
                error = save_thumbnail_to_filesystem(output_dir, cache_dir, filename_base, thumbnail, model_url, db_path, file_hash, mtime, blacked_out=0, metadata=metadata)
                if error:
                    result += error + "\n"
                else:
                    result += f"Saved thumbnail for {file_path.name} to filesystem\n"
                all_results.append(result)
                all_display_items.append((file_path.name, thumbnail, model_url, 0, metadata))
                continue

        hashes = []
        file_hash = compute_file_sha256(file_path) if not file_hash else file_hash
        if file_hash:
            hashes.append(('file_sha256', file_hash))
        if metadata.get('ss_new_sd_model_hash'):
            hashes.append(('ss_new_sd_model_hash', metadata['ss_new_sd_model_hash']))
        if metadata.get('ss_sd_model_hash'):
            hashes.append(('ss_sd_model_hash', metadata['ss_sd_model_hash']))

        if hashes:
            logger.debug(f"Using hashes for online search: {[(h[0], h[1][:10] + '...') for h in hashes]}")
            error, thumbnail, model_url = search_thumbnail_online(hashes, file_path.name, file_path)
            if error:
                result += error + "\n"
            if thumbnail:
                error = save_thumbnail_to_filesystem(output_dir, cache_dir, filename_base, thumbnail, model_url, db_path, file_hash, mtime, blacked_out=0, metadata=metadata)
                if error:
                    result += error + "\n"
                else:
                    result += f"Saved thumbnail for {file_path.name} to filesystem\n"
                if model_url:
                    result += f"Found Civitai model page: {model_url}\n"
            all_results.append(result)
            all_display_items.append((file_path.name, thumbnail, model_url, 0, metadata))
            continue

        logger.info(f"No thumbnail or model URL found for {file_path.name}")
        result += f"No thumbnail or model URL found for {file_path.name}\n"
        all_results.append(result)
        all_display_items.append((file_path.name, None, None, 0, metadata))

    all_results.append("Processing complete: All files processed.")
    html_result = generate_html_page(all_display_items, Path(directory))
    all_results.append(html_result)
    result_text = "\n".join(all_results)
    gallery_items = [
        (
            resize_to_fit(img, 200),
            f"{name} {'(Blacked Out)' if blk else ''}\n---\nMetadata:\n" + (
                '\n'.join([f"{k}: {v}" for k, v in meta.items()]) if meta else "No metadata available"
            )
        )
        for name, img, _, blk, meta in all_display_items if img is not None
    ]
    files = list(sorted(f.name for f in Path(directory).glob("*.safetensors")))
    logger.info("Processing complete")
    return result_text, files, gr.update(value=gallery_items), None

def reprocess_directory(directory, clear_thumbnails_folder, output_text, progress=gr.Progress()):
    """Reprocess all files, overwriting thumbnails.db."""
    logger.info(f"Reprocessing directory: {directory}")
    if not os.path.isdir(directory):
        logger.error(f"Invalid directory: {directory} - {os.strerror(errno.errorcode.get(errno.EACCES, 'Unknown error'))}")
        return f"{output_text}\nError: {directory} is not a valid directory", [], gr.update(), None

    directory_path = Path(directory)
    db_path = directory_path / "thumbnails.db"
    output_dir = directory_path / "thumbnails"
    cache_dir = output_dir / "cache"

    if db_path.exists():
        db_path.unlink()
        logger.debug("Deleted existing thumbnails.db")

    if clear_thumbnails_folder and output_dir.exists():
        shutil.rmtree(output_dir)
        logger.debug("Deleted existing thumbnails folder")
    output_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    return process_lora_directory(directory, output_text, progress)

def gradio_interface():
    """Create Gradio UI for LoRA thumbnail extraction with custom CSS for Dark Mode."""
    logger.debug("Initializing Gradio interface with custom CSS for Dark Mode")
    
    dark_mode_css = """
    .gradio-container {
        background-color: #1A1A1A !important;
        color: #E0E0E0 !important;
    }
    .block, .block .inner, .block .content {
        background-color: #1A1A1A !important;
        color: #E0E0E0 !important;
    }
    h1, h2, h3, h4, h5, h6, label {
        color: #E0E0E0 !important;
    }
    .block input, .block textarea, .block select {
        background-color: #2A2A2A !important;
        color: #E0E0E0 !important;
        border: 1px solid #4A4A4A !important;
    }
    .block button {
        background-color: #4A4A4A !important;
        color: #E0E0E0 !important;
        border: 1px solid #666666 !important;
    }
    .block button:hover {
        background-color: #666666 !important;
    }
    .gallery-item, .gallery .inner {
        background-color: #1A1A1A !important;
    }
    .gallery-item img {
        border: 1px solid #4A4A4A !important;
    }
    .gallery-item .caption, .gallery .caption {
        color: #000000 !important;
        background-color: rgba(255, 255, 255, 0.7) !important;
        padding: 2px 5px !important;
        margin: 0 !important;
        white-space: pre-wrap !important;
        font-size: 12px !important;
    }
    """
    
    with gr.Blocks(css=dark_mode_css, head="""
        <script>
            function setCaptionColor() {
                const captions = document.querySelectorAll('.gallery-item .caption, .gallery .caption');
                captions.forEach(caption => {
                    caption.style.color = '#000000';
                    caption.style.backgroundColor = 'rgba(255, 255, 255, 0.7)';
                    caption.style.whiteSpace = 'pre-wrap';
                    caption.style.fontSize = '12px';
                });
            }
            window.addEventListener('load', setCaptionColor);
            const observer = new MutationObserver(setCaptionColor);
            observer.observe(document.body, { childList: true, subtree: true });
        </script>
    """) as demo:
        gr.Markdown("# LoRA and Model Metadata and Thumbnail Extractor")
        directory_input = gr.Textbox(label="Directory Path", placeholder="Enter the path to your safetensors files directory")
        api_key_input = gr.Textbox(label="Civitai API Key (Optional)", placeholder="Enter your Civitai API key", type="password")
        process_button = gr.Button("Process Files")
        clear_thumbnails_button = gr.Button("Clear Old Thumbnails")
        reprocess_button = gr.Button("Reprocess All Files")
        clear_thumbnails_folder = gr.Checkbox(label="Clear Thumbnails Folder on Reprocess", value=False)
        output_text = gr.Textbox(label="Processing Log", lines=10)
        
        gr.Markdown("## Thumbnails (Click an image to select)")
        gallery = gr.Gallery(
            label="Thumbnails",
            columns=4,
            rows=3,
            object_fit="contain",
            height="auto",
            allow_preview=True,
            elem_classes=["gallery"],
            show_label=True,
            show_share_button=False,
            show_download_button=False,
            preview=True,
            format="png",
        )
        file_list_state = gr.State(value=[])
        selected_file_state = gr.State(value=None)
        with gr.Row():
            blackout_button = gr.Button("Black Out Selected")
            restore_button = gr.Button("Restore Selected")

        def set_api_key(api_key, output_text):
            global CIVITAI_API_KEY
            CIVITAI_API_KEY = api_key
            return f"{output_text}\nAPI key set successfully" if api_key else f"{output_text}\nNo API key provided"

        def update_display(directory, output_text, progress=gr.Progress()):
            result, files, gallery_update, selected_file_update = process_lora_directory(directory, output_text, progress)
            return result, files, gallery_update, selected_file_update

        def clear_thumbnails(directory, output_text):
            result, gallery_update, files, selected_file_update = clear_old_thumbnails(directory, output_text)
            return result, files, gallery_update, selected_file_update

        def reprocess(directory, clear_thumbnails_folder, output_text, progress=gr.Progress()):
            result, files, gallery_update, selected_file_update = reprocess_directory(directory, clear_thumbnails_folder, output_text, progress)
            return result, files, gallery_update, selected_file_update

        def on_directory_change(directory):
            directory_path = Path(directory)
            if os.path.isdir(directory):
                files = sorted(f.name for f in directory_path.glob("*.safetensors"))
                return files, None
            return [], None

        def on_gallery_select(evt: gr.SelectData):
            logger.debug(f"Gallery selection: index={evt.index}, value={evt.value}")
            selected_file = None
            if evt.index is not None and isinstance(evt.value, dict) and 'caption' in evt.value:
                caption = evt.value['caption']
                # Extract the filename from the caption (before the metadata separator)
                selected_file = caption.split('\n---\n')[0].strip().split(' (Blacked Out)')[0].strip()
                logger.debug(f"Selected file from gallery: {selected_file}")
            else:
                logger.warning(f"Invalid gallery selection data: {evt.value}")
            return selected_file

        api_key_input.change(
            fn=set_api_key,
            inputs=[api_key_input, output_text],
            outputs=output_text
        )

        directory_input.change(
            fn=on_directory_change,
            inputs=directory_input,
            outputs=[file_list_state, selected_file_state]
        )

        process_button.click(
            fn=update_display,
            inputs=[directory_input, output_text],
            outputs=[output_text, file_list_state, gallery, selected_file_state]
        )

        clear_thumbnails_button.click(
            fn=clear_thumbnails,
            inputs=[directory_input, output_text],
            outputs=[output_text, file_list_state, gallery, selected_file_state]
        )

        reprocess_button.click(
            fn=reprocess,
            inputs=[directory_input, clear_thumbnails_folder, output_text],
            outputs=[output_text, file_list_state, gallery, selected_file_state]
        )

        gallery.select(
            fn=on_gallery_select,
            inputs=None,
            outputs=selected_file_state
        )

        blackout_button.click(
            fn=black_out_thumbnail,
            inputs=[directory_input, selected_file_state, output_text, file_list_state],
            outputs=[output_text, gallery, file_list_state, selected_file_state]
        )

        restore_button.click(
            fn=restore_thumbnail,
            inputs=[directory_input, selected_file_state, output_text, file_list_state],
            outputs=[output_text, gallery, file_list_state, selected_file_state]
        )

    return demo

if __name__ == "__main__":
    logger.info("Starting application")
    demo = gradio_interface()
    demo.launch(server_name="0.0.0.0")

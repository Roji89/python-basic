import requests
import os
from urllib.parse import urlparse

def download_photo(url, filename=None):
    """
    Download a photo from a URL and save it to the current directory.
    
    Args:
        url (str): The URL of the photo to download
        filename (str, optional): Custom filename. If None, extracts from URL
    
    Returns:
        str: Path to the downloaded file, or "exists" if file already exists, or None if failed
    """
    try:
        # If no filename provided, extract from URL
        if filename is None:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            # If no filename in URL, use a default
            if not filename or '.' not in filename:
                filename = "downloaded_image.jpg"
        
        # Ensure the filename has .jpg extension
        if not filename.lower().endswith(('.jpg', '.jpeg')):
            filename += '.jpg'
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Check if file already exists
        if os.path.exists(filename):
            print(f"File already exists, skipping: {filename}")
            return "exists"
        
        # Send GET request to download the image
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Write the image data to file
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print(f"Photo downloaded successfully: {filename}")
        return filename
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading photo: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Main execution
if __name__ == "__main__":
    # List of specific photo IDs to download
    # photo_ids = list(range(10548147184, 10548148471))
    photo_ids = [
        10548147223, 10548147229, 10548148404, 10548148407, 10548148439,
        10548148451, 10548148428, 10548148444, 10548148386, 10548147897,
        10548147380, 10548147210, 10548147917, 10548147406, 10548147927,
        10860894059, 10860894060, 10860894061, 10860894062, 10860894063,
        10860894064, 10860894065, 10860894066, 10860894067, 10860894068,
        10860894069, 10860894070, 10860894071, 10860894072, 10860894073,
        10860894074, 10860894075, 10860894076, 10860894077, 10860894078,
        10860894079, 10860894080, 10860894081, 10860894082, 10860894083,
        10860894084, 10860894085, 10860894086, 10860894087, 10860894088
    ]
    
    # Base URL template
    # base_url = "https://pictime5neu1public-pub-fbevadhjh6c7b2d9.a02.azurefd.net/pictures/44/887/44887470/cyh4slwokz9s7if100/lowres/"
    base_url = "https://lafauteauxcouleurs.pic-time.com/-aaronrojajustin/download?mode=hiresphoto&photoId="
    # Download all photos or just one (change index to download different photo)
    photo_index = 0  # Change this to download different photos (0-1286) when download_all is False
    
    successful_downloads = 0
    failed_downloads = 0
    skipped_downloads = 0
    
    for i, photo_id in enumerate(photo_ids):
        # photo_url = f"{base_url}{photo_id}.jpg"
        photo_url = f"{base_url}{photo_id}&systemName=pictime&gui=yes&accessToken="
        custom_filename = f"photo/{photo_id}.jpg"
        print(f"Processing photo {i + 1} of {len(photo_ids)}: {photo_id}")
        
        downloaded_file = download_photo(photo_url, custom_filename)
        
        if downloaded_file == "exists":
            skipped_downloads += 1
            print(f"⚠ Skipped (already exists): {custom_filename}")
        elif downloaded_file:
            successful_downloads += 1
            print(f"✓ Downloaded: {downloaded_file}")
        else:
            failed_downloads += 1
            print(f"✗ Failed to download photo {photo_id}")
    
    print(f"downloaded photo: {(photo_ids)} ")
    print(f"Range: {photo_ids[0]} to {photo_ids[-1]}")

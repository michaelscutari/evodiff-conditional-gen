import requests
import urllib.parse
import os
import time
from typing import Optional, List

# --- Configuration ---
UNIPROT_API_URL = "https://rest.uniprot.org/uniprotkb/stream"
# Define a User-Agent (Good practice for APIs)
HEADERS = {"User-Agent": "Python UniProt Downloader/1.0 (Bio Project)"}
# Output directory for datasets
OUTPUT_DIR = "data"

def build_uniprot_query(
    ec_number: Optional[str] = None,
    reviewed: Optional[bool] = True,
    no_fragments: Optional[bool] = True,
    other_criteria: Optional[List[str]] = None
) -> str:
    """
    Constructs a query string for the UniProt API based on criteria.

    Args:
        ec_number: EC number pattern (e.g., "3.1.1.*" or "1.1.1.1").
        reviewed: If True, query only reviewed (Swiss-Prot) entries.
        no_fragments: If True, exclude protein fragments.
        other_criteria: A list of additional query parts (e.g., ["taxonomy_id:9606"]).

    Returns:
        The formatted UniProt query string.
    """
    query_parts = []
    if reviewed is not None:
        query_parts.append(f"(reviewed:{str(reviewed).lower()})")
    if ec_number:
        query_parts.append(f"(ec:{ec_number})")
    if no_fragments is not None:
        # The key fix: We need to use NOT fragment:true to exclude fragments
        query_parts.append(f"(NOT fragment:true)")
    if other_criteria:
        query_parts.extend(other_criteria)

    query = " AND ".join(filter(None, query_parts))
    if not query:
        raise ValueError("Query cannot be empty. Please provide some criteria.")
    return query

def download_uniprot_data(
    query: str,
    output_filepath: str,
    data_format: str = "fasta",
    fields: Optional[str] = None,
    chunk_size: int = 8192,
    max_retries: int = 3,
    retry_delay: int = 5 # seconds
) -> None:
    """
    Downloads data from UniProt based on a query and saves it to a file.

    Args:
        query: The UniProt query string.
        output_filepath: Path to save the downloaded data.
        data_format: Desired format ('fasta', 'tsv', 'json', etc.).
        fields: Comma-separated list of fields for tsv/json formats.
                Ignored for fasta format via stream endpoint.
        chunk_size: Size of chunks to download in bytes.
        max_retries: Maximum number of retries on connection errors.
        retry_delay: Delay between retries in seconds.
    """
    params = {
        "query": query,
        "format": data_format,
        "compressed": "false", # We'll handle uncompressed data
    }
    # Add fields only if format is not fasta (stream endpoint ignores fields for fasta)
    if data_format != "fasta" and fields:
        params["fields"] = fields

    encoded_params = urllib.parse.urlencode(params)
    request_url = f"{UNIPROT_API_URL}?{encoded_params}"

    print(f"Submitting request to UniProt: {UNIPROT_API_URL} with query parts...")
    # print(f"Full URL (params encoded): {request_url}") # Can be very long

    retries = 0
    while retries < max_retries:
        try:
            with requests.get(request_url, stream=True, headers=HEADERS, timeout=300) as response: # Added timeout
                response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

                print(f"Downloading data ({response.headers.get('Content-Type', 'N/A')})...")
                os.makedirs(os.path.dirname(output_filepath), exist_ok=True) # Ensure directory exists
                with open(output_filepath, "wb") as f:
                    downloaded_bytes = 0
                    start_time = time.time()
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        # Optional: print progress
                        # print(f"\rDownloaded {downloaded_bytes / (1024*1024):.2f} MB", end="")

                end_time = time.time()
                print(f"\nDownload complete. Saved {downloaded_bytes / (1024*1024):.2f} MB "
                      f"to {output_filepath} in {end_time - start_time:.2f} seconds.")
                return # Success, exit function

        except requests.exceptions.RequestException as e:
            retries += 1
            print(f"\nError during download: {e}")
            if retries < max_retries:
                print(f"Retrying in {retry_delay} seconds... ({retries}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Download failed.")
                raise # Re-raise the last exception

def main():
    """Main function to coordinate data downloading."""
    print(f"Script started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure base data directory exists

    # --- Phase 1 Data Collection ---
    # Goal: Get reviewed, non-fragment sequences for EC 3.1.1.* in FASTA format.
    print("\n--- Starting Phase 1 Data Collection ---")
    phase1_ec = "7.*"
    phase1_output_filename = "ec_7_reviewed.fasta"
    phase1_output_filepath = os.path.join(OUTPUT_DIR, phase1_output_filename)
    phase1_format = "fasta"

    print(f"Target: Reviewed, non-fragment enzymes matching EC {phase1_ec}")
    print(f"Output: {phase1_output_filepath}")
    print(f"Format: {phase1_format}")

    # Build the query for Phase 1
    phase1_query = build_uniprot_query(
        ec_number=phase1_ec,
        reviewed=True,
        no_fragments=True
    )
    print(f"Constructed UniProt Query: {phase1_query}")

    # Download the data for Phase 1
    try:
        download_uniprot_data(
            query=phase1_query,
            output_filepath=phase1_output_filepath,
            data_format=phase1_format
        )
        print("Phase 1 data download successful.")
    except Exception as e:
        print(f"Phase 1 data download failed: {e}")
        # Decide if script should stop if Phase 1 fails (usually yes)
        return

    # --- Phase 2 Data Collection (Setup, but not executed by default) ---
    # Goal: Get sequences AND their specific EC numbers (e.g., in TSV format)
    #       for conditional training. Requires 'ec' field.
    print("\n--- Phase 2 Data Collection (Code Prepared, Not Executing Now) ---")
    execute_phase2 = True # <<< SET TO True TO RUN PHASE 2 DOWNLOAD >>>

    if execute_phase2:
        phase2_ec = "7.*" # Could refine to specific sub-subclasses if needed
        phase2_output_filename = "ec_7_reviewed.tsv"
        phase2_output_filepath = os.path.join(OUTPUT_DIR, phase2_output_filename)
        phase2_format = "tsv"
        # Fields needed: Accession ID, the sequence, and the EC number
        phase2_fields = "accession,sequence,ec"

        print(f"Target: Reviewed, non-fragment enzymes matching EC {phase2_ec}")
        print(f"Output: {phase2_output_filepath}")
        print(f"Format: {phase2_format}")
        print(f"Fields: {phase2_fields}")

        # Query might be the same as Phase 1 or adjusted
        phase2_query = build_uniprot_query(
            ec_number=phase2_ec,
            reviewed=True,
            no_fragments=True
        )
        print(f"Constructed UniProt Query: {phase2_query}")

        # Download the data for Phase 2
        try:
            download_uniprot_data(
                query=phase2_query,
                output_filepath=phase2_output_filepath,
                data_format=phase2_format,
                fields=phase2_fields
            )
            print("Phase 2 data download successful.")
        except Exception as e:
            print(f"Phase 2 data download failed: {e}")
    else:
        print("Phase 2 execution is skipped (set 'execute_phase2 = True' to run).")

    print(f"\nScript finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
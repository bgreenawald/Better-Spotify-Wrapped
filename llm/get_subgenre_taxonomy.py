import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

PARENT_GENRES = [
    "blues",
    "children's music",
    "classical",
    "comedy",
    "country",
    "edm",
    "electronica",
    "folk",
    "funk",
    "gospel",
    "hip hop",
    "jazz",
    "metal",
    "new age",
    "opera",
    "pop",
    "r&b",
    "rap",
    "reggae",
    "rock",
    "soul",
    "soundtrack",
    "world",
]

BASE_PROMPT = """Given the following list of 'parent' genres, I will provide you a child genre. Your job is to categorize the child genre into one or more of the parent genres. So, if the parent genre list was

[country, pop, metal] and the child genre was pop-country, you would return

['country', 'pop']

If the child genre is one of the parent genres exactly, return an empty list.

Here is the parent list:

blues
children's music
classical
comedy
country
edm
electronica
folk
funk
gospel
hip hop
jazz
metal
new age
opera
pop
r&b
rap
reggae
rock
soul
soundtrack
world

Here is the child genre:

{genre}

Return ONLY a JSON array with the parent genres that the child genre belongs to. Do not include any explanation or additional text."""


class GenreTaxonomyClassifier:
    def __init__(self, api_key: str | None = None):
        """Initialize the genre classifier with Google Generative AI.

        Args:
            api_key: Google AI API key. If not provided, will look for GOOGLE_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY env var or pass api_key parameter."
            )

        self.client = genai.Client(api_key=self.api_key)

    def prepare_batch_requests_file(
        self, child_genres: list[str], output_file: str = "batch_requests.jsonl"
    ) -> str:
        """Prepare batch requests in JSONL format for Google AI batch processing.

        Args:
            child_genres: List of child genres to classify
            output_file: Path to save the JSONL file

        Returns:
            Path to the created JSONL file
        """
        with open(output_file, "w") as f:
            for i, genre in enumerate(child_genres):
                request_obj = {
                    "key": f"genre-{i}",
                    "request": {
                        "contents": [
                            {
                                "parts": [{"text": BASE_PROMPT.format(genre=genre)}],
                                "role": "user",
                            }
                        ],
                        "generation_config": {
                            "temperature": 0.2,
                            "top_p": 0.8,
                            "top_k": 40,
                            "max_output_tokens": 256,
                            "response_mime_type": "application/json",
                        },
                    },
                }
                f.write(json.dumps(request_obj) + "\n")

        print(f"Created batch request file with {len(child_genres)} requests: {output_file}")
        return output_file

    def submit_batch_job_with_file(
        self, input_file: str, display_name: str = "genre-classification"
    ):
        """Submit a batch job to Google Generative AI using a file and poll for completion.

        Args:
            input_file: Path to JSONL file with requests
            display_name: Display name for the batch job

        Returns:
            The completed batch job with results
        """
        # Upload the input file
        print(f"Uploading batch file: {input_file}")
        uploaded_file = self.client.files.upload(
            file=input_file,
            config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
        )

        print(f"File uploaded successfully: {uploaded_file.name}")

        # Create batch job using the uploaded file
        batch_job = self.client.batches.create(
            model="gemini-2.5-flash",
            src=uploaded_file.name,
            config={"display_name": display_name},
        )

        print(f"Batch job created: {batch_job.name}")

        # Poll for completion
        completed_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }

        while batch_job.state.name not in completed_states:
            print(f"Current state: {batch_job.state.name}")
            time.sleep(30)  # Check every 30 seconds
            batch_job = self.client.batches.get(name=batch_job.name)

        print(f"Job finished with state: {batch_job.state.name}")

        if batch_job.state.name == "JOB_STATE_FAILED":
            raise RuntimeError(f"Batch job failed: {batch_job.error}")

        return batch_job

    def process_batch_with_inline(
        self, child_genres: list[str], display_name: str = "genre-classification-inline"
    ) -> dict[str, list[str]]:
        """Process genres using inline batch requests (for smaller batches).

        Args:
            child_genres: List of child genres to classify
            display_name: Display name for the batch job

        Returns:
            Dictionary mapping child genres to their parent genres
        """
        # Prepare inline requests
        inline_requests = []
        for genre in child_genres:
            request = {
                "contents": [
                    {"parts": [{"text": BASE_PROMPT.format(genre=genre)}], "role": "user"}
                ],
                "generation_config": {
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 256,
                    "response_mime_type": "application/json",
                },
            }
            inline_requests.append(request)

        # Create batch job with inline requests
        batch_job = self.client.batches.create(
            model="gemini-2.5-flash",
            src=inline_requests,
            config={"display_name": display_name},
        )

        print(f"Batch job created: {batch_job.name}")

        # Poll for completion
        completed_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }

        while batch_job.state.name not in completed_states:
            print(f"Current state: {batch_job.state.name}")
            time.sleep(30)
            batch_job = self.client.batches.get(name=batch_job.name)

        print(f"Job finished with state: {batch_job.state.name}")

        if batch_job.state.name != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Batch job did not succeed: {batch_job.state.name}")

        # Process inline responses
        results = {}
        if batch_job.dest and batch_job.dest.inlined_responses:
            for i, inline_response in enumerate(batch_job.dest.inlined_responses):
                if i < len(child_genres):
                    genre = child_genres[i]
                    if inline_response.response:
                        try:
                            # Parse JSON response
                            response_text = inline_response.response.text
                            parent_genres = json.loads(response_text)
                            # Validate that returned genres are in the parent list
                            valid_parents = [g for g in parent_genres if g in PARENT_GENRES]
                            results[genre] = valid_parents
                        except Exception as e:
                            print(f"Error parsing response for genre '{genre}': {e}")
                            results[genre] = []
                    elif inline_response.error:
                        print(f"Error for genre '{genre}': {inline_response.error}")
                        results[genre] = []

        return results

    def process_batch(
        self, child_genres: list[str], use_file: bool = False
    ) -> dict[str, list[str]]:
        """Process a batch of child genres and return their parent genre classifications.

        Args:
            child_genres: List of child genres to classify
            use_file: Whether to use file-based batch processing (for large batches)

        Returns:
            Dictionary mapping child genres to their parent genres
        """
        results = {}

        # Check if it's a single genre that's already a parent
        if len(child_genres) == 1 and child_genres[0] in PARENT_GENRES:
            return {child_genres[0]: []}

        if use_file or len(child_genres) > 100:
            # Use file-based batch processing for large batches
            print(f"Processing {len(child_genres)} genres using file-based batch API...")

            # Prepare batch request file
            request_file = self.prepare_batch_requests_file(child_genres)

            try:
                # Submit and wait for batch job
                batch_job = self.submit_batch_job_with_file(request_file)

                # Process file results
                if batch_job.dest and batch_job.dest.file_name:
                    result_file_name = batch_job.dest.file_name
                    print(f"Downloading results from: {result_file_name}")

                    # Download the result file
                    file_content = self.client.files.download(file=result_file_name)

                    # Parse JSONL results
                    for line in file_content.decode("utf-8").strip().split("\n"):
                        if line:
                            try:
                                result = json.loads(line)
                                # Extract the key and response
                                if "key" in result:
                                    # Extract genre index from key (e.g., "genre-0" -> 0)
                                    key_parts = result["key"].split("-")
                                    if len(key_parts) == 2 and key_parts[1].isdigit():
                                        idx = int(key_parts[1])
                                        if idx < len(child_genres):
                                            genre = child_genres[idx]

                                            if "response" in result:
                                                response_text = result["response"]["candidates"][0][
                                                    "content"
                                                ]["parts"][0]["text"]
                                                parent_genres = json.loads(response_text)
                                                valid_parents = [
                                                    g for g in parent_genres if g in PARENT_GENRES
                                                ]
                                                results[genre] = valid_parents
                                            elif "error" in result:
                                                print(
                                                    f"Error for genre '{genre}': {result['error']}"
                                                )
                                                results[genre] = []
                            except Exception as e:
                                print(f"Error parsing result line: {e}")

            finally:
                # Clean up the request file
                if os.path.exists(request_file):
                    os.remove(request_file)

        else:
            # Use inline batch processing for smaller batches
            results = self.process_batch_with_inline(child_genres)

        return results

    def classify_genre(self, child_genre: str) -> list[str]:
        """Classify a single child genre into parent genres.

        Args:
            child_genre: The child genre to classify

        Returns:
            List of parent genres
        """
        # Check if it's already a parent genre
        if child_genre in PARENT_GENRES:
            return []

        # Use inline batch for single genre
        results = self.process_batch_with_inline([child_genre])
        return results.get(child_genre, [])


def main():
    """Example usage of the genre taxonomy classifier."""

    # Initialize classifier
    classifier = GenreTaxonomyClassifier()

    # Load genres from file if it exists
    genres_file = "llm/all_subgenres.txt"
    if os.path.exists(genres_file):
        with open(genres_file) as f:
            test_genres = [line.strip() for line in f if line.strip()]
    else:
        # Fallback to example genres
        test_genres = [
            "pop-country",
            "death metal",
            "neo-soul",
            "trap",
            "indie rock",
            "jazz fusion",
            "pop",  # This is a parent genre, should return empty
            "electronic dance",
            "folk rock",
            "gospel blues",
        ]

    print(f"\nProcessing {len(test_genres)} genres...")
    print("-" * 50)

    # Process all genres as a batch
    # Use file-based processing for large batches
    use_file = len(test_genres) > 100
    results = classifier.process_batch(test_genres, use_file=use_file)

    # Display results
    for genre, parents in results.items():
        print(f"{genre:30} -> {parents}")

    # Save results to file
    output_file = "genre_taxonomy_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

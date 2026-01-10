import json
import os
import time
from typing import Literal

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
    def __init__(
        self,
        api_key: str | None = None,
        provider: Literal["gemini", "openai"] | None = None,
        model: str | None = None,
    ):
        """Initialize the genre classifier with Gemini or OpenAI.

        Args:
            api_key: API key for the chosen provider. If omitted, pulled from env.
            provider: "gemini" or "openai". Defaults to "gemini".
            model: Optional model name override for the provider.
        """
        load_dotenv()

        self.provider: Literal["gemini", "openai"] = (
            provider or os.getenv("LLM_PROVIDER", "openai").lower()
        )  # type: ignore[assignment]
        if self.provider not in ("gemini", "openai"):
            raise ValueError("provider must be 'gemini' or 'openai'")

        if self.provider == "gemini":
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("Google API key required. Set GOOGLE_API_KEY or pass api_key.")
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.client = genai.Client(api_key=self.api_key)
            self._openai_client = None
        else:
            # OpenAI
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
            # Defer import to avoid hard dependency when using Gemini
            try:
                from openai import OpenAI  # type: ignore

                self._openai_client = OpenAI(api_key=self.api_key)
            except Exception as e:  # pragma: no cover - import-time failure only
                raise RuntimeError(
                    "openai package is required for provider='openai'. Install 'openai'."
                ) from e
            self.client = None

    def prepare_batch_requests_file(
        self,
        child_genres: list[str],
        output_file: str = "batch_requests.jsonl",
    ) -> str:
        """Prepare batch requests JSONL for the selected provider.

        For Gemini, writes file formatted for Google AI Batch API.
        For OpenAI, writes file formatted for OpenAI Batch API.
        """
        with open(output_file, "w") as f:
            for i, genre in enumerate(child_genres):
                if self.provider == "gemini":
                    request_obj = {
                        "key": f"genre-{i}",
                        "request": {
                            "contents": [
                                {
                                    "parts": [{"text": BASE_PROMPT.format(genre=genre)}],
                                    "role": "user",
                                }
                            ],
                            "reasoning": "medium",
                            "generation_config": {
                                "response_mime_type": "application/json",
                            },
                        },
                    }
                else:
                    # OpenAI batch entry using Chat Completions endpoint
                    request_obj = {
                        "custom_id": f"genre-{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a strict genre classifier. "
                                        "Return ONLY a JSON array of allowed parent genres."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": BASE_PROMPT.format(genre=genre),
                                },
                            ],
                            "reasoning_effort": "low",
                        },
                    }
                f.write(json.dumps(request_obj) + "\n")

        print(
            f"Created {self.provider} batch request file with {len(child_genres)} requests: {output_file}"
        )
        return output_file

    def submit_batch_job_with_file(
        self,
        input_file: str,
        display_name: str = "genre-classification",
    ):
        """Submit a batch job using the provider's Batch API and poll for completion."""
        print(f"Uploading batch file: {input_file}")

        if self.provider == "gemini":
            uploaded_file = self.client.files.upload(
                file=input_file,
                config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
            )
            print(f"File uploaded successfully: {uploaded_file.name}")

            batch_job = self.client.batches.create(
                model=self.model,
                src=uploaded_file.name,
                config={"display_name": display_name},
            )

            print(f"Batch job created: {batch_job.name}")

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

            if batch_job.state.name == "JOB_STATE_FAILED":
                raise RuntimeError(f"Batch job failed: {batch_job.error}")

            return batch_job

        # OpenAI provider
        client = self._openai_client
        with open(input_file, "rb") as f:
            batch_input_file = client.files.create(file=f, purpose="batch")
        print(f"File uploaded successfully: {batch_input_file.id}")

        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"Batch job created: {batch.id}")

        terminal_status = {"completed", "failed", "expired", "cancelled"}
        while batch.status not in terminal_status:
            print(f"Current state: {batch.status}")
            time.sleep(30)
            batch = client.batches.retrieve(batch.id)

        print(f"Job finished with state: {batch.status}")

        if batch.status == "failed":
            raise RuntimeError(f"Batch job failed: {batch}")

        return batch

    def process_batch_with_inline(
        self,
        child_genres: list[str],
        display_name: str = "genre-classification-inline",
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
                "reasoning": "medium",
                "generation_config": {
                    "response_mime_type": "application/json",
                },
            }
            inline_requests.append(request)

        # Create batch job with inline requests
        batch_job = self.client.batches.create(
            model=self.model,
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
        self,
        child_genres: list[str],
        use_file: bool = False,
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

        if use_file or len(child_genres) > 100 or self.provider == "openai":
            # Use file-based batch processing for large batches
            print(
                f"Processing {len(child_genres)} genres using {self.provider} file-based batch API..."
            )

            # Prepare batch request file
            request_file = self.prepare_batch_requests_file(child_genres)

            try:
                # Submit and wait for batch job
                batch_job = self.submit_batch_job_with_file(request_file)

                # Process file results
                if self.provider == "gemini":
                    if batch_job.dest and batch_job.dest.file_name:
                        result_file_name = batch_job.dest.file_name
                        print(f"Downloading results from: {result_file_name}")

                        file_content = self.client.files.download(file=result_file_name)

                        for line in file_content.decode("utf-8").strip().split("\n"):
                            if line:
                                try:
                                    result = json.loads(line)
                                    if "key" in result:
                                        key_parts = result["key"].split("-")
                                        if len(key_parts) == 2 and key_parts[1].isdigit():
                                            idx = int(key_parts[1])
                                            if idx < len(child_genres):
                                                genre = child_genres[idx]

                                                if "response" in result:
                                                    response_text = result["response"][
                                                        "candidates"
                                                    ][0]["content"]["parts"][0]["text"]
                                                    parent_genres = json.loads(response_text)
                                                    valid_parents = [
                                                        g
                                                        for g in parent_genres
                                                        if g in PARENT_GENRES
                                                    ]
                                                    results[genre] = valid_parents
                                                elif "error" in result:
                                                    print(
                                                        f"Error for genre '{genre}': {result['error']}"
                                                    )
                                                    results[genre] = []
                                except Exception as e:
                                    print(f"Error parsing result line: {e}")
                else:
                    # OpenAI results
                    client = self._openai_client
                    if not getattr(batch_job, "output_file_id", None):
                        print("No output_file_id found on batch job.")
                        return results
                    file_id = batch_job.output_file_id
                    print(f"Downloading results from: {file_id}")

                    file_response = client.files.content(file_id)
                    # Handle different SDK response types safely
                    if hasattr(file_response, "text"):
                        text = file_response.text
                    elif hasattr(file_response, "content"):
                        content = file_response.content
                        text = (
                            content.decode("utf-8")
                            if isinstance(content, bytes | bytearray)
                            else str(content)
                        )
                    else:
                        # Try to read as bytes
                        text = bytes(file_response).decode("utf-8")

                    for line in str(text).strip().split("\n"):
                        if not line:
                            continue
                        try:
                            result = json.loads(line)
                            if "custom_id" in result:
                                key_parts = result["custom_id"].split("-")
                                if len(key_parts) == 2 and key_parts[1].isdigit():
                                    idx = int(key_parts[1])
                                    if idx < len(child_genres):
                                        genre = child_genres[idx]
                                        if result.get("response") and result["response"].get(
                                            "body"
                                        ):
                                            body = result["response"]["body"]
                                            # Chat Completions parsing
                                            content = (
                                                body.get("choices", [{}])[0]
                                                .get("message", {})
                                                .get("content", "")
                                            )
                                            try:
                                                parent_genres = json.loads(content)
                                            except Exception:
                                                parent_genres = []
                                            valid_parents = [
                                                g for g in parent_genres if g in PARENT_GENRES
                                            ]
                                            results[genre] = valid_parents
                                        elif result.get("error"):
                                            print(f"Error for genre '{genre}': {result['error']}")
                                            results[genre] = []
                        except Exception as e:
                            print(f"Error parsing result line: {e}")

            finally:
                # Clean up the request file
                if os.path.exists(request_file):
                    os.remove(request_file)

        else:
            # Use inline batch processing for smaller batches (Gemini only)
            if self.provider == "gemini":
                results = self.process_batch_with_inline(child_genres)
            else:
                # For OpenAI, prefer file-based batches; inline fallback not implemented
                raise NotImplementedError(
                    "Inline processing for provider='openai' is not implemented."
                )

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

    # Initialize classifier; choose provider via LLM_PROVIDER env var
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

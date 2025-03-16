import logging
import time
from collections.abc import Iterable
from collections.abc import Sequence
from typing import Any
from typing import Optional

import apache_beam as beam
import openai
from apache_beam.io.components.adaptive_throttler import AdaptiveThrottler
from apache_beam.metrics.metric import Metrics
from apache_beam.ml.inference.base import ModelHandler
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.transforms.base import EmbeddingsManager
from apache_beam.ml.transforms.base import _TextEmbeddingHandler
from apache_beam.utils import retry

__all__ = ["OpenAITextEmbeddings"]

_BATCH_SIZE = 20  # OpenAI API can handle larger batches than VertexAI
_MSEC_TO_SEC = 1000
_DEFAULT_MODEL = "text-embedding-ada-002"  # Default embedding model


LOGGER = logging.getLogger("OpenAIEmbeddings")


def _retry_on_openai_error(exception):
    """
    Retry filter that returns True if a returned error is appropriate for retry.
    This includes rate limit errors and server errors.

    Args:
      exception: the returned exception encountered during the request/response
        loop.

    Returns:
      boolean indication whether the exception should be retried
    """
    if isinstance(exception, openai.RateLimitError):
        return True
    if isinstance(exception, openai.APIError):
        return True
    if isinstance(exception, openai.APIConnectionError):
        return True
    return False


class _OpenAITextEmbeddingHandler(ModelHandler):
    """
    Note: Intended for internal use and guarantees no backwards compatibility.
    """
    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.organization = organization
        self.dimensions = dimensions

        # Configure AdaptiveThrottler and throttling metrics for client-side
        # throttling behavior.
        self.throttled_secs = Metrics.counter(
            OpenAITextEmbeddings, "cumulativeThrottlingSeconds")
        self.throttler = AdaptiveThrottler(
            window_ms=1, bucket_ms=1, overload_ratio=2)

    @retry.with_exponential_backoff(
        num_retries=5, retry_filter=_retry_on_openai_error)
    def get_request(
        self,
        text_batch: Sequence[str],
        client: Any,
        throttle_delay_secs: int):
        while self.throttler.throttle_request(time.time() * _MSEC_TO_SEC):
            LOGGER.info(
                "Delaying request for %d seconds due to previous failures",
                throttle_delay_secs)
            time.sleep(throttle_delay_secs)
            self.throttled_secs.inc(throttle_delay_secs)

        try:
            req_time = time.time()
            # Create the request parameters
            params = {
                "model": self.model_name,
                "input": text_batch
            }
            
            # Add dimensions if specified
            if self.dimensions:
                params["dimensions"] = self.dimensions
                
            response = client.embeddings.create(**params)
            self.throttler.successful_request(req_time * _MSEC_TO_SEC)
            
            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]
            return embeddings
            
        except openai.RateLimitError as e:
            LOGGER.warning("Request was rate limited by OpenAI API: %s", str(e))
            raise
        except Exception as e:
            LOGGER.error("Unexpected exception raised as part of request: %s", str(e))
            raise

    def run_inference(
        self,
        batch: Sequence[str],
        model: Any,
        inference_args: Optional[dict[str, Any]] = None,
    ) -> Iterable:
        embeddings = []
        batch_size = _BATCH_SIZE
        for i in range(0, len(batch), batch_size):
            text_batch = batch[i:i + batch_size]
            embeddings_batch = self.get_request(
                text_batch=text_batch, client=model, throttle_delay_secs=5)
            embeddings.extend(embeddings_batch)
        return embeddings

    def load_model(self):
        # Initialize the OpenAI client
        client = openai.OpenAI(
            api_key=self.api_key,
            organization=self.organization
        )
        return client

    def __repr__(self):
        # ModelHandler is internal to the user and is not exposed.
        # Hence we need to override the __repr__ method to expose
        # the name of the class.
        return 'OpenAITextEmbeddings'


class OpenAITextEmbeddings(beam.PTransform):
    """
    A PTransform for generating text embeddings using OpenAI API.
    Processes PCollection of dictionaries with text columns and appends embeddings.
    """
    def __init__(
        self,
        columns: list[str],
        model_name: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        dimensions: Optional[int] = None,
        **kwargs):
        """
        Embedding Config for OpenAI Text Embedding models.
        Text Embeddings are generated for a batch of text using the OpenAI API.
        Embeddings are returned in a list for each text in the batch.

        Args:
          columns: The columns containing the text to be embedded.
          model_name: The OpenAI embedding model to use.
                     Default is "text-embedding-ada-002".
          api_key: The OpenAI API key. If not provided, will use the 
                   OPENAI_API_KEY environment variable.
          organization: The OpenAI organization ID. If not provided, will use 
                       the OPENAI_ORG_ID environment variable.
          dimensions: Optional parameter to specify the dimensions of the 
                     embeddings to return. Only available for some models.
        """
        super().__init__()
        self.columns = columns
        self.model_name = model_name
        self.api_key = api_key
        self.organization = organization
        self.dimensions = dimensions
        self.kwargs = kwargs
        self._embedding_manager = None

    def expand(self, pcoll):
        """
        Expands the PTransform by creating an EmbeddingsManager and applying it.
        """
        # Create the EmbeddingsManager
        embedding_manager = _OpenAIEmbeddingsManager(
            columns=self.columns,
            model_name=self.model_name,
            api_key=self.api_key,
            organization=self.organization,
            dimensions=self.dimensions,
            **self.kwargs
        )
        
        # Apply the PTransform from the manager
        return pcoll | embedding_manager.get_ptransform_for_processing()


class _OpenAIEmbeddingsManager(EmbeddingsManager):
    """
    EmbeddingsManager implementation for OpenAI.
    """
    def __init__(
        self,
        columns: list[str],
        model_name: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        dimensions: Optional[int] = None,
        **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.organization = organization
        self.dimensions = dimensions
        super().__init__(columns=columns, **kwargs)

    def get_model_handler(self) -> ModelHandler:
        return _OpenAITextEmbeddingHandler(
            model_name=self.model_name,
            api_key=self.api_key,
            organization=self.organization,
            dimensions=self.dimensions,
        )

    def get_ptransform_for_processing(self, **kwargs) -> beam.PTransform:
        return RunInference(
            model_handler=_TextEmbeddingHandler(self),
            inference_args=self.inference_args)

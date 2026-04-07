import logging
import config

logger = logging.getLogger(__name__)


class Observability:
    """
    Handles the integration with Langfuse for tracing and observability.
    If enabled in config, it provides a CallbackHandler that LangChain/LangGraph
    uses to log every LLM call, tool invocation, and state transition.
    """

    def __init__(self):
        self._enabled = config.LANGFUSE_ENABLED
        self._handler = None
        self._client = None

        if not self._enabled:
            return

        if not config.LANGFUSE_PUBLIC_KEY or not config.LANGFUSE_SECRET_KEY:
            logger.warning("Langfuse enabled but API keys are missing — skipping")
            self._enabled = False
            return

        try:
            from langfuse import get_client
            from langfuse.langchain import CallbackHandler

            self._client = get_client()

            if self._client.auth_check():
                print("Langfuse client is authenticated and ready!")
            else:
                print("Authentication failed. Please check your credentials and host.")
                self._enabled = False
                return

            self._handler = CallbackHandler()
        except Exception as exc:
            logger.warning("Could not initialize Langfuse: %s", exc)
            self._enabled = False

    def get_handler(self):
        """
        Returns the LangChain-compatible CallbackHandler.
        If Langfuse is disabled, this returns None.
        """
        return self._handler

    def flush(self):
        """
        Force-sends any buffered traces to the Langfuse server.
        Useful when clearing a session to ensure all data is captured.
        """
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                pass
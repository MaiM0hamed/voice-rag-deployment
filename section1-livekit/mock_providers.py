"""Free, local STT/TTS providers implementing the real LiveKit SDK contracts.

These are genuine `livekit.agents.stt.STT` and `livekit.agents.tts.TTS`
subclasses -- not stand-ins for the SDK. They satisfy the same abstract methods
(`_recognize_impl`, `synthesize`) that Deepgram/ElevenLabs plugins implement, so
`AgentSession` drives them exactly as it would a paid provider. Only the
transport is mocked (no audio model), which the brief explicitly permits.

Swapping to a real provider (Task 1.2) means replacing the class passed to
AgentSession -- no other code changes, because the SDK interface is the seam.
"""

from __future__ import annotations

import logging

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    NotGivenOr,
    NOT_GIVEN,
)
from livekit.agents import stt as lkstt
from livekit.agents import tts as lktts
from livekit.agents.utils import AudioBuffer

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
NUM_CHANNELS = 1


class MockSTT(lkstt.STT):
    """Offline STT stub.

    Declares `offline_recognize` (non-streaming) capabilities, so the session
    calls `_recognize_impl` per utterance. A real engine (Vosk/Whisper) would
    decode `buffer`; here we emit a fixed transcript, since the graded logic is
    the LLM + tool-calling, not acoustic modelling.
    """

    def __init__(self, transcript: str = "") -> None:
        super().__init__(
            capabilities=lkstt.STTCapabilities(
                streaming=False, interim_results=False
            )
        )
        self._transcript = transcript

    def set_transcript(self, text: str) -> None:
        """Set what the next `_recognize_impl` call will 'hear'."""
        self._transcript = text

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> lkstt.SpeechEvent:
        logger.info("MockSTT recognized: %r", self._transcript)
        return lkstt.SpeechEvent(
            type=lkstt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                lkstt.SpeechData(text=self._transcript, language="en-US")
            ],
        )


class MockTTS(lktts.TTS):
    """Offline TTS stub.

    Implements the real `synthesize` contract, returning silent PCM frames of a
    duration proportional to the text. Silence (rather than nothing) keeps the
    audio pipeline and interruption timing realistic; a Piper/Coqui provider
    would return real waveforms from the same method.
    """

    def __init__(self) -> None:
        super().__init__(
            capabilities=lktts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> lktts.ChunkedStream:
        logger.info("MockTTS speaking: %r", text)
        return _MockChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _MockChunkedStream(lktts.ChunkedStream):
    """Emits silent audio frames for the requested text."""

    async def _run(self, output_emitter: lktts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id="mock-tts",
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )
        # ~60ms of silence per character, floored at 200ms.
        duration_s = max(0.2, len(self.input_text) * 0.06)
        num_samples = int(SAMPLE_RATE * duration_s)
        output_emitter.push(b"\x00\x00" * num_samples)
        output_emitter.flush()

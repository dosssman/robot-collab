from __future__ import annotations
import os
import dataclasses


@dataclasses.dataclass(frozen=True)
class OpenaiConfig:
    organization: str
    api_key: str

    @classmethod
    def from_env(cls) -> OpenaiConfig:
        return cls(
            organization=os.environ["OPENAI_ORGANIZATION"],
            api_key=os.environ["OPENAI_API_KEY"],
        )

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    whisper_model: str = "mlx-community/whisper-large-v3-turbo"
    wav2vec2_model: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    supported_languages: list[str] = ["es", "hr", "de", "zh"]
    default_language: str = "es"
    upload_dir: str = "uploads"
    claude_command: str = "claude"
    cors_origins: list[str] = ["http://localhost:5173"]


settings = Settings()

from typing import Literal, Optional

from pydantic import BaseModel, Field


class AccountData(BaseModel):
    """
    Raw social media account attributes. All fields are optional; missing
    fields are kept as NaN and imputed by the preprocessor fitted during
    training (same behavior as the offline prediction pipeline).
    """

    # numerical / boolean flag columns
    statuses_count: Optional[float] = None
    followers_count: Optional[float] = None
    friends_count: Optional[float] = None
    favourites_count: Optional[float] = None
    listed_count: Optional[float] = None
    default_profile: Optional[float] = None
    default_profile_image: Optional[float] = None
    geo_enabled: Optional[float] = None
    profile_use_background_image: Optional[float] = None
    profile_background_tile: Optional[float] = None
    protected: Optional[float] = None
    verified: Optional[float] = None
    utc_offset: Optional[float] = None
    # categorical columns
    lang: Optional[str] = None
    time_zone: Optional[str] = None
    profile_text_color: Optional[str] = None
    profile_sidebar_border_color: Optional[str] = None
    profile_sidebar_fill_color: Optional[str] = None
    profile_background_color: Optional[str] = None
    profile_link_color: Optional[str] = None
    # free-text columns
    name: Optional[str] = None
    screen_name: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "statuses_count": 1200,
                    "followers_count": 340,
                    "friends_count": 280,
                    "favourites_count": 50,
                    "listed_count": 2,
                    "verified": 0,
                    "lang": "en",
                    "time_zone": "UTC",
                    "name": "John Doe",
                    "screen_name": "johndoe",
                    "description": "Software developer. Coffee enthusiast.",
                    "location": "New York",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    prediction: Literal["Real Account", "Fake Account"]
    fake_probability: Optional[float] = Field(
        default=None,
        description="Model probability that the account is fake (0-1). "
        "Null when the trained model does not support probability estimates.",
    )
    real_probability: Optional[float] = Field(
        default=None,
        description="Model probability that the account is real (0-1). "
        "Null when the trained model does not support probability estimates.",
    )


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    model_loaded: bool
    preprocessor_loaded: bool

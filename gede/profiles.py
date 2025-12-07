# coding=utf-8
#
# profiles.py
#
# Profile management for gede
#

import os
import json
from typing import Optional, Dict, List
from pydantic import BaseModel

from .top import logger
from .config import get_config_dir


class Profile(BaseModel):
    """Profile configuration for gede"""

    model: Optional[str] = None
    instruction: Optional[str] = None
    private: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    web_search: Optional[str] = None
    tools: Optional[List[str]] = None
    trace: Optional[bool] = None
    log_level: Optional[str] = None
    mcp: Optional[List[str]] = None


def get_profiles_filepath():
    """Get the path to the profiles.json file"""
    config_dir = get_config_dir()
    return os.path.join(config_dir, "profiles.json")


def create_default_profiles():
    """Create default profiles.json file"""
    profiles_file = get_profiles_filepath()

    default_profiles = {
        "default": {
            "model": "voice_engine:doubao-seed-1-6",
            "instruction": "You are a helpful assistant.",
            "private": False,
            "reasoning_effort": "medium",
            "web_search": "auto",
            "tools": ["web_search", "now", "read_page"],
            "trace": False,
            "log_level": "INFO",
        }
    }

    with open(profiles_file, "w", encoding="utf-8") as f:
        json.dump(default_profiles, f, indent=2, ensure_ascii=False)

    logger.info(f"Default profiles.json created at {profiles_file}")


def load_profiles() -> Dict[str, Profile]:
    """Load all profiles from profiles.json"""
    profiles_file = get_profiles_filepath()

    if not os.path.exists(profiles_file):
        create_default_profiles()

    try:
        with open(profiles_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        profiles = {}
        for name, config in data.items():
            try:
                profiles[name] = Profile(**config)
            except Exception as e:
                logger.warning(f"Invalid profile '{name}': {e}")

        return profiles
    except Exception as e:
        logger.error(f"Failed to load profiles from {profiles_file}: {e}")
        create_default_profiles()
        return load_profiles()


def get_profile(profile_name: str = "default") -> Profile:
    """Get a specific profile by name"""
    profiles = load_profiles()

    if profile_name not in profiles:
        logger.warning(f"Profile '{profile_name}' not found, using 'default' profile")
        profile_name = "default"

    if profile_name not in profiles:
        logger.error("No 'default' profile found, creating default profiles")
        create_default_profiles()
        profiles = load_profiles()

    return profiles.get(profile_name, Profile())


def list_profiles() -> List[str]:
    """List all available profile names"""
    profiles = load_profiles()
    return list(profiles.keys())


def save_profiles(profiles: Dict[str, Profile]):
    """Save profiles to profiles.json"""
    profiles_file = get_profiles_filepath()

    data = {}
    for name, profile in profiles.items():
        # Convert to dict and remove None values
        profile_dict = profile.model_dump(exclude_none=True)
        data[name] = profile_dict

    try:
        with open(profiles_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Profiles saved to {profiles_file}")
    except Exception as e:
        logger.error(f"Failed to save profiles to {profiles_file}: {e}")


def add_profile(name: str, profile: Profile):
    """Add a new profile"""
    profiles = load_profiles()
    profiles[name] = profile
    save_profiles(profiles)


def delete_profile(name: str) -> bool:
    """Delete a profile"""
    if name == "default":
        logger.error("Cannot delete 'default' profile")
        return False

    profiles = load_profiles()
    if name in profiles:
        del profiles[name]
        save_profiles(profiles)
        return True
    return False

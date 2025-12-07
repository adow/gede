# coding=utf-8
#
# chatcore.py
#

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from uuid import uuid4
from dataclasses import dataclass, field, fields

from agents import (
    TResponseInputItem,
    ModelSettings,
)

from .top import logger, gede_dir
from .llm.providers import (
    GetLLMModelResult,
    get_llm_model,
    get_llm_model_settings,
    DEFAULT_MODEL_PATH,
)
from .llm.llm_provider import ReasoningEffortType, WebSearchType
from .encrypt import encrypt_aes, decrypt_aes
from .llm.generate_title import generate_title


def gede_instructions_dir():
    return os.path.join(gede_dir(), "instructions")


def create_model_settings_from_dict(data: dict) -> ModelSettings:
    """Create a ModelSettings instance from a dictionary."""

    # Get valid field names from ModelSettings dataclass
    valid_field_names = {field.name for field in fields(ModelSettings)}

    # Filter out None values and unknown fields
    filtered_data = {
        key: value
        for key, value in data.items()
        if key in valid_field_names and value is not None
    }

    # Handle special cases for BaseModel fields if needed
    if "reasoning" in filtered_data and isinstance(filtered_data["reasoning"], dict):
        try:
            from openai.types.shared import Reasoning

            filtered_data["reasoning"] = Reasoning(**filtered_data["reasoning"])
        except (ImportError, TypeError):
            # If we can't reconstruct the Reasoning object, skip it
            filtered_data.pop("reasoning", None)

    return ModelSettings(**filtered_data)


@dataclass
class ChatModel:
    chat_id: str
    filename: Optional[str]
    instruction: str
    model_path: str
    title: str
    is_private: bool = False
    private_password: Optional[str] = None
    message_num_in_context: int = 6

    # Model parameters set by user through /set-model-settings command
    user_model_settings: ModelSettings = field(default_factory=ModelSettings)

    # messages
    messages: list[TResponseInputItem] = field(default_factory=list)

    @property
    def model(self) -> GetLLMModelResult:
        """Get the LLM model used for current chat"""
        (provider_id, model_id) = self.model_path.split(":", maxsplit=1)
        return get_llm_model(provider_id, model_id)

    @property
    def model_settings(self) -> ModelSettings:
        (provider_id, model_id) = self.model_path.split(":", maxsplit=1)
        # Merge ModelSettings defined in provider with user-defined ones here
        base_settings = get_llm_model_settings(provider_id, model_id)
        return base_settings.resolve(self.user_model_settings)

    @property
    def info(self) -> str:
        output = f"""[bold]chat_id[/bold]: {self.chat_id}
[bold]title[/bold]: {self.title or ""}
[bold]filename[/bold]: {self.filename or ""}
[bold]private[/bold]: {self.is_private}
[bold]private_password[/bold]: {"Set" if self.private_password else "Not Set"}
[bold]model[/bold]: {self.model.provider_name}:{self.model.model.name}
[bold]instruction[/bold]: {self.instruction}
[bold]message_num_in_context[/bold]: {self.message_num_in_context}
[bold]message count[/bold]: {len(self.messages)}
[bold]model_supports[/bold]: {", ".join(self.model.model_supports) if self.model.model_supports else ""}
[bold]model_settings[/bold]:\n\t{json.dumps(self.model_settings.to_json_dict(), ensure_ascii=False)}
"""

        return output

    def __init__(self, is_private=False):
        self.chat_id = "cht-" + str(uuid4())
        # now = datetime.now().strftime("%Y%m%d%H%M%S")
        # self.filename = f"{now}.json"
        self.filename = None
        self.instruction = (
            loaded_instructions[0][1]
            if loaded_instructions
            else "You are a helpful assistant."
        )
        self.title = "New Chat"
        self.is_private = is_private
        self.model_path = DEFAULT_MODEL_PATH  # Default model path
        self.user_model_settings = ModelSettings()

        self.messages = [{"role": "system", "content": self.instruction}]

    def set_instruction(self, instruction: str):
        """Set chat instruction"""
        self.instruction = instruction
        system_message_pos = -1
        for pos, message in enumerate(self.messages):
            if message.get("role") == "system":
                system_message_pos = pos

        if system_message_pos >= 0:
            del self.messages[system_message_pos]
        self.messages.insert(0, {"role": "system", "content": self.instruction})

    def append_user_message(self, new_message: str):
        """Add user message"""
        self.messages.append({"role": "user", "content": new_message})
        self.save()

    def append_assistant_message(self, new_message: str):
        """Add assistant message"""
        self.messages.append({"role": "assistant", "content": new_message})
        self.save()

    def get_messages_to_talk(self):
        # Keep only the first message and the last few messages
        input_messages_copy: list[TResponseInputItem] = []
        if (
            self.message_num_in_context <= 0
            or len(self.messages) <= self.message_num_in_context
        ):
            input_messages_copy = self.messages.copy()
        else:
            input_messages_copy.append(self.messages[0])
            input_messages_copy.extend(self.messages[-self.message_num_in_context :])
        return input_messages_copy

    def generate_filename(self):
        if self.filename:
            return self.filename
        else:
            now = datetime.now().strftime("%Y%m%d%H%M%S")
            self.filename = f"{now}.json"

    async def geneate_title(self):
        if self.is_private:
            logger.warning("Private chat cannot generate title automatically.")
            return
        if self.title != "" and self.title != "New Chat" and self.title != "Untitled":
            logger.debug(
                "chat title is already set, skip generating title. %s", self.title
            )
            return
        try:
            title = await generate_title(self.messages)
            logger.debug("Generated chat title: %s", title)
            if title:
                self.title = title
            return title
        except Exception as e:
            logger.error("Failed to generate chat title: %s", str(e))
            return

    def save(self):
        if not self.filename:
            logger.debug("Chat filename is not set, cannot save.")
            return
        if self.is_private and not self.private_password:
            # logger.debug("Private chat requires a password to save.")
            return None
        # save chat file
        chat_dir = os.path.join(
            gede_dir(), "chats", "public" if not self.is_private else "private"
        )
        if not os.path.exists(chat_dir):
            os.makedirs(chat_dir)
        filepath = os.path.join(chat_dir, self.filename)
        output = {
            "chat_id": self.chat_id,
            "title": self.title,
            "model_path": self.model_path,
            "is_private": self.is_private,
            "model_settings": self.model_settings.to_json_dict(),
        }
        output_messages = []
        # messages
        for one in self.messages:
            if "role" in one and "content" in one:
                role = one["role"]
                content = one["content"]
                # logger.debug("role: %s, content: %s", role, content)

                if role and content:
                    if isinstance(content, str):
                        content = content
                    elif isinstance(content, list):
                        content = content[0]
                        content = content.get("text")
                    content = str(content)
                    if self.is_private and self.private_password:
                        content = encrypt_aes(content, self.private_password)
                    output_messages.append({"role": role, "content": content})
        output["messages"] = output_messages
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(output, indent=2, ensure_ascii=False))
            logger.debug("Saved chat to %s", filepath)
        return filepath

    @classmethod
    def load_from_file(
        CLS, filename: str, is_private=False, private_password: Optional[str] = None
    ):
        """Load chat record"""
        if not filename.endswith(".json"):
            filename += ".json"
        chat_dir = os.path.join(
            gede_dir(), "chats", "public" if not is_private else "private"
        )
        filepath = os.path.join(chat_dir, filename)
        if not os.path.exists(filepath):
            logger.error("Chat file %s does not exist.", filepath)
            return None
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
                is_load_private = bool(data.get("is_private", False))
                if is_private != is_load_private:
                    logger.error("Chat privacy mode does not match.")
                    return None
                if is_private and not private_password:
                    logger.error("Private chat requires a password to load.")
                    return None
                chat = ChatModel(is_private=is_load_private)
                chat.chat_id = data.get("chat_id")
                chat.filename = filename
                chat.title = data.get("title", "Chat")
                chat.private_password = private_password
                chat.model_path = data.get("model_path", DEFAULT_MODEL_PATH)
                model_settings_data = data.get("model_settings", {})
                chat.user_model_settings = create_model_settings_from_dict(
                    model_settings_data
                )
                messages = data.get("messages", [])
                output_messages = []
                for one in messages:
                    if "role" in one and "content" in one:
                        role = one["role"]
                        content = one["content"]
                        if role and content:
                            if isinstance(content, str):
                                content = content
                            elif isinstance(content, list):
                                content = content[0]
                                content = content.get("text")
                            content = str(content)
                            if chat.is_private and private_password:
                                try:
                                    content = decrypt_aes(content, private_password)
                                except Exception as e:
                                    logger.error(
                                        "Failed to decrypt message. Possibly wrong password."
                                    )
                                    return None
                            output_messages.append({"role": role, "content": content})
                            if role == "system":
                                chat.instruction = content
                chat.messages = output_messages
                logger.info("Loaded chat from %s", filepath)
                return chat
            except Exception as e:
                logger.error("Failed to load chat file %s: %s", filepath, str(e))
                return None

    # model settings

    def set_model_reasoning(self, effort: Optional[ReasoningEffortType] = None):
        (provider_id, model_id) = self.model_path.split(":", maxsplit=1)
        current_settings = self.user_model_settings
        self.user_model_settings = self.model.provider.set_model_reasoning(
            model_id, current_settings, effort
        )

    def set_model_web_search(self, search_type: WebSearchType = "on"):
        (provider_id, model_id) = self.model_path.split(":", maxsplit=1)
        current_settings = self.user_model_settings
        self.user_model_settings = self.model.provider.set_model_web_search(
            model_id, current_settings, search_type
        )


# instructions


def load_instructions():
    instructions_dir = gede_instructions_dir()

    default_instruction = """You are a helpful assistant. Please follow the user's instructions carefully. """
    if not os.path.exists(instructions_dir):
        os.makedirs(instructions_dir)
        file_path = os.path.join(instructions_dir, "default.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(default_instruction)

    instructions: list[tuple[str, str]] = []
    for filename in os.listdir(instructions_dir):
        if filename.endswith(".txt") or filename.endswith(".md"):
            (name, ext) = os.path.splitext(filename)
            filepath = os.path.join(instructions_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    if name == "default":
                        default_instruction = content
                    else:
                        content_lines = content.splitlines()
                        instructions.append((content_lines[0][:38], content))
    instructions.insert(0, (default_instruction, default_instruction))
    logger.debug(f"Loaded {len(instructions)} instructions from {instructions_dir}")
    return instructions


loaded_instructions = load_instructions()

# prompts


def gede_prompts_dir():
    return os.path.join(gede_dir(), "prompts")


def load_prompts():
    prompts_dir = gede_prompts_dir()
    if not os.path.exists(prompts_dir):
        os.makedirs(prompts_dir)
        file_path = os.path.join(prompts_dir, "hello.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Hello.")

    prompts: list[str] = []
    for filename in os.listdir(prompts_dir):
        if filename.endswith(".txt") or filename.endswith(".md"):
            filepath = os.path.join(prompts_dir, filename)
            with open(filepath, "r") as f:
                content = f.read().strip()
                if content:
                    prompts.append(content)
    logger.debug(f"Loaded {len(prompts)} prompts from {prompts_dir}")
    return prompts


loaded_prompts = load_prompts()

# chats


def load_chats_files(is_private=False):
    chat_dir = os.path.join(
        gede_dir(), "chats", "public" if not is_private else "private"
    )
    if not os.path.exists(chat_dir):
        os.makedirs(chat_dir)
    files: list[str] = []
    for filename in os.listdir(chat_dir):
        if filename.endswith(".json"):
            files.append(filename)
    logger.debug(f"Loaded {len(files)} chat files from {chat_dir}")
    return sorted(files, reverse=True)


def load_chats_files_2(is_private=False):
    chat_dir = os.path.join(
        gede_dir(), "chats", "public" if not is_private else "private"
    )
    if not os.path.exists(chat_dir):
        os.makedirs(chat_dir)
    # Each item contains (label, value)
    file_title_list: list[tuple[str, str]] = []
    for filename in os.listdir(chat_dir):
        if filename.endswith(".json"):
            title = "Untitled"

            # Read file content, get the first user message as title
            with open(os.path.join(chat_dir, filename), "r") as f:
                content = f.read()
                chat_data = json.loads(content)
                title = chat_data.get("title", "Untitled")
                if title == "" or title == "New Chat" or title == "Untitled":
                    messages = chat_data.get("messages", [])
                    for one_message in messages:
                        if one_message.get("role") == "user":
                            title = one_message.get("content", "Untitled")
                            break

            title = filename + ": " + title[:30]
            file_title_list.append((title, filename))
    logger.debug(f"Loaded {len(file_title_list)} chat files from {chat_dir}")
    return sorted(file_title_list, key=lambda x: x[1], reverse=True)


# export chat


class ExportChat:
    def __init__(self, chat: ChatModel):
        self.chat = chat

    def export_txt(self, filepath: Path) -> bool:
        """Export chat record as TXT file"""
        try:
            txt_dir = os.path.dirname(filepath)
            if txt_dir and not os.path.exists(txt_dir):
                os.makedirs(txt_dir)

            with open(filepath, "w", encoding="utf-8") as f:
                # Write title and metadata
                f.write(f"{self.chat.title}\n")
                f.write("=" * 80 + "\n\n")

                meta_text = f"Model: {self.chat.model.provider_name}:{self.chat.model.model.name}\n"
                meta_text += (
                    f"Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(meta_text)
                f.write("-" * 80 + "\n\n")

                # Write message content
                for message in self.chat.messages:
                    role = message.get("role", "unknown")
                    content = message.get("content", "")

                    # Skip over system messages
                    if role == "system":
                        continue

                    # Write role identifier
                    role_label = {"user": "USER", "assistant": "ASSISTANT"}.get(
                        role, role.upper()
                    )

                    f.write(f"【{role_label}】\n")

                    # Write message content (preserve original markdown format)
                    content_str = str(content)
                    f.write(content_str)
                    f.write("\n\n")
                    f.write("-" * 80 + "\n\n")

            logger.info(f"Successfully exported chat to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export chat to TXT: {str(e)}")
            return False

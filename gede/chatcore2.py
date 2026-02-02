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

from my_llmkit.chat.model_settings import ModelSettings
from my_llmkit.chat import UnifiedMessage, ContentBlock
from my_llmkit.models import ModelInfo, get_model_info


from .top import logger, gede_dir, DEFAULT_MODEL_PATH

from .encrypt import encrypt_aes, decrypt_aes
from .llm.generate_title import generate_title


def gede_instructions_dir():
    return os.path.join(gede_dir(), "instructions")


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
    messages: list[UnifiedMessage] = field(default_factory=list)

    def __init__(self, is_private=False):
        self.chat_id = "cht-" + str(uuid4())
        # now = datetime.now().strftime("%Y%m%d%H%M%S")
        # self.filename = f"{now}.json"
        self.filename = None
        self.instruction = "You are a helpful assistant."  # TODO: load instructions
        self.title = "New Chat"
        self.is_private = is_private
        self.model_path = DEFAULT_MODEL_PATH  # Default model path
        self.user_model_settings = ModelSettings()

        self.messages = [UnifiedMessage(role="system", content=self.instruction)]

    # properties

    @property
    async def model(self) -> ModelInfo:
        info = await get_model_info(self.model_path)
        if not info:
            raise ValueError(f"Cannot get model info for path: {self.model_path}")
        return info

    @property
    def model_settings(self) -> ModelSettings:
        # TODO: 从 provider 获取默认设置
        return ModelSettings()

    @property
    async def info(self) -> str:
        model = await self.model
        output = f"""[bold]chat_id[/bold]: {self.chat_id}
[bold]title[/bold]: {self.title or ""}
[bold]filename[/bold]: {self.filename or ""}
[bold]private[/bold]: {self.is_private}
[bold]private_password[/bold]: {"Set" if self.private_password else "Not Set"}
[bold]model[/bold]: {model.provider_name or ""}:{model.model_name}
[bold]instruction[/bold]: {self.instruction}
[bold]message_num_in_context[/bold]: {self.message_num_in_context}
[bold]message count[/bold]: {len(self.messages)}
[bold]model_supports[/bold]: {model.supports_description}
[bold]model_settings[/bold]:\n\t{json.dumps(self.model_settings.to_json_dict(), ensure_ascii=False)}
"""

        return output

    # messages

    def set_instruction(self, instruction: str):
        """Set chat instruction"""
        self.instruction = instruction
        system_message_pos = -1
        for pos, message in enumerate(self.messages):
            if message.role == "system":
                system_message_pos = pos

        if system_message_pos >= 0:
            del self.messages[system_message_pos]
        self.messages.insert(0, UnifiedMessage(role="system", content=instruction))

    def append_user_message(self, new_message: str):
        """Add user message"""
        self.messages.append(UnifiedMessage(role="user", content=new_message))
        self.save()

    def append_assistant_message(self, new_message: str):
        """Add assistant message"""
        self.messages.append(UnifiedMessage(role="assistant", content=new_message))
        self.save()

    def get_messages_to_talk(self):
        # Keep only the first message and the last few messages
        input_messages_copy: list[UnifiedMessage] = []
        if (
            self.message_num_in_context <= 0
            or len(self.messages) <= self.message_num_in_context
        ):
            input_messages_copy = self.messages.copy()
        else:
            input_messages_copy.append(self.messages[0])
            input_messages_copy.extend(self.messages[-self.message_num_in_context :])
        return input_messages_copy

    # save

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
            # TODO: implement generate_title
            # title = await generate_title(self.messages)
            title = "GENERATED TITLE"

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
            role = one.role
            content = one.content
            # logger.debug("role: %s, content: %s", role, content)

            if role and content:
                # Serialize content to JSON-compatible format
                if isinstance(content, str):
                    serialized_content = content
                elif isinstance(content, list):
                    # Convert list[ContentBlock] to list[dict]
                    serialized_content = [block.to_dict() for block in content]
                else:
                    serialized_content = str(content)

                # Handle encryption for private chats
                if self.is_private and self.private_password:
                    # Convert to JSON string for encryption
                    content_str = json.dumps(serialized_content, ensure_ascii=False)
                    encrypted_content = encrypt_aes(content_str, self.private_password)
                    output_messages.append({"role": role, "content": encrypted_content})
                else:
                    # Save structured content directly
                    output_messages.append(
                        {"role": role, "content": serialized_content}
                    )
        output["messages"] = output_messages
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(output, indent=2, ensure_ascii=False))
            logger.debug("Saved chat to %s", filepath)
        return filepath

    # read

    @classmethod
    def load_from_file(
        cls, filename: str, is_private=False, private_password: Optional[str] = None
    ):
        """Load chat record from file"""
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

                # Create new chat instance
                chat = cls(is_private=is_load_private)
                chat.chat_id = data.get("chat_id")
                chat.filename = filename
                chat.title = data.get("title", "Chat")
                chat.private_password = private_password
                chat.model_path = data.get("model_path", DEFAULT_MODEL_PATH)

                # Load model settings
                model_settings_data = data.get("model_settings", {})
                if model_settings_data:
                    # Filter only valid ModelSettings fields
                    from dataclasses import fields as dataclass_fields

                    valid_field_names = {
                        field.name for field in dataclass_fields(ModelSettings)
                    }
                    filtered_data = {
                        key: value
                        for key, value in model_settings_data.items()
                        if key in valid_field_names and value is not None
                    }
                    chat.user_model_settings = ModelSettings(**filtered_data)

                # Load messages
                messages = data.get("messages", [])
                output_messages = []
                for one in messages:
                    if "role" in one and "content" in one:
                        role = one["role"]
                        content = one["content"]

                        if role and content:
                            # Handle decryption for private chats
                            if chat.is_private and private_password:
                                try:
                                    content = decrypt_aes(content, private_password)
                                    # Try to parse as JSON (could be structured content)
                                    try:
                                        content = json.loads(content)
                                    except (json.JSONDecodeError, TypeError):
                                        # If not JSON, treat as plain string
                                        pass
                                except Exception as e:
                                    logger.error(
                                        "Failed to decrypt message. Possibly wrong password."
                                    )
                                    return None

                            # Reconstruct UnifiedMessage with proper content type
                            if isinstance(content, str):
                                # Plain text content
                                message = UnifiedMessage(role=role, content=content)
                            elif isinstance(content, list):
                                # Multimodal content - reconstruct ContentBlocks
                                try:
                                    content_blocks = [
                                        ContentBlock.from_dict(block)
                                        for block in content
                                    ]
                                    message = UnifiedMessage(
                                        role=role, content=content_blocks
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "Failed to reconstruct ContentBlocks, using string fallback: %s",
                                        str(e),
                                    )
                                    message = UnifiedMessage(
                                        role=role, content=str(content)
                                    )
                            else:
                                # Fallback to string
                                message = UnifiedMessage(
                                    role=role, content=str(content)
                                )

                            output_messages.append(message)

                            # Update instruction if system message
                            if role == "system" and isinstance(content, str):
                                chat.instruction = content

                chat.messages = output_messages
                logger.info("Loaded chat from %s", filepath)
                return chat
            except Exception as e:
                logger.error("Failed to load chat file %s: %s", filepath, str(e))
                return None


# TODO: 读取 instructions 列表
# TODO: 读取 prompts 列表

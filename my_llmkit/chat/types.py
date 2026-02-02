# coding=utf-8
#
# 统一类型定义
#
import base64
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union, cast


@dataclass
class UnifiedToolCall:
    """统一的工具调用格式"""

    id: str
    name: str
    arguments: str  # JSON string
    extra_content: Optional[str] = (
        None  # 用于存储额外内容（如 Gemini 的 tool_call extra_content）
    )


@dataclass
class UnifiedUsage:
    """统一的 token 使用量格式"""

    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    reasoning_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    model: Optional[str] = None


@dataclass
class UnifiedChunk:
    """统一的流式分片格式，与任何 SDK 解耦"""

    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[UnifiedToolCall]] = None  # 增量工具调用
    usage: Optional[UnifiedUsage] = None
    # 厂商特有字段，用于需要保留的元数据（如 Claude 的 signature）
    vendor_metadata: Optional[dict[str, Any]] = None


@dataclass
class UnifiedResponse:
    """统一的非流式完整响应格式，与任何 SDK 解耦"""

    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[UnifiedToolCall]] = None
    usage: Optional[UnifiedUsage] = None
    # 厂商特有字段，用于需要保留的元数据（如 Claude 的 signature）
    vendor_metadata: Optional[dict[str, Any]] = None


# contents


@dataclass
class ContentBlock(ABC):
    """内容块基类 - 只存储数据，不包含转换逻辑"""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """转换为字典表示"""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContentBlock":
        """
        工厂方法：从字典创建对应类型的内容块对象

        Args:
            data: 包含内容块数据的字典，必须包含 content_block_type 字段

        Returns:
            对应类型的 ContentBlock 实例（TextContent/ImageContent/DocumentContent）

        Raises:
            ValueError: 不支持的 content_block_type 或缺少必需字段

        Example:
            >>> block = ContentBlock.from_dict({"content_block_type": "text", "text": "Hello"})
            >>> isinstance(block, TextContent)
            True
            >>> img = ContentBlock.from_dict({"content_block_type": "image", "image_url": "https://..."})
            >>> isinstance(img, ImageContent)
            True
        """
        content_type = data.get("content_block_type")
        if not content_type:
            raise ValueError("字典中缺少 content_block_type 字段")

        if content_type == "text":
            return TextContent.from_dict(data)
        elif content_type == "image":
            return ImageContent.from_dict(data)
        elif content_type == "document":
            return DocumentContent.from_dict(data)
        else:
            raise ValueError(f"不支持的 content_block_type: {content_type}")


@dataclass
class TextContent(ContentBlock):
    """文本内容块"""

    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"content_block_type": "text", "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextContent":
        """
        从字典创建文本内容块

        Args:
            data: 必须包含 text 字段

        Returns:
            TextContent 对象

        Raises:
            ValueError: 缺少必需字段
        """
        if "text" not in data:
            raise ValueError("TextContent 需要 text 字段")
        return cls(text=data["text"])


@dataclass
class ImageContent(ContentBlock):
    """
    图片内容块

    统一使用 image_url 存储：
    1. HTTP/HTTPS URL: "https://example.com/image.jpg"
    2. Base64 Data: "/9j/4AAQ..." (raw base64 string)
    """

    # 图片 URL（如果是 base64，则为原始 base64 字符串）
    image_url: str

    # 媒体类型（当 image_url 为 base64 时必填）
    media_type: Optional[str] = None

    # 详细程度（仅部分厂商支持，如 OpenAI）
    detail: Optional[Literal["auto", "low", "high"]] = None

    @classmethod
    def from_url(
        cls, url: str, detail: Optional[Literal["auto", "low", "high"]] = None
    ) -> "ImageContent":
        """从 URL 创建图片内容块"""
        return cls(image_url=url, detail=detail)

    @classmethod
    def from_base64(
        cls,
        base64_data: str,
        media_type: Literal[
            "image/jpeg", "image/png", "image/gif", "image/webp"
        ] = "image/jpeg",
        detail: Optional[Literal["auto", "low", "high"]] = None,
    ) -> "ImageContent":
        """从 base64 数据创建图片内容块"""
        return cls(image_url=base64_data, media_type=media_type, detail=detail)

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        detail: Optional[Literal["auto", "low", "high"]] = None,
    ) -> "ImageContent":
        """
        从本地文件创建图片内容块，自动检测媒体类型并转换为 base64

        Args:
            file_path: 图片文件路径
            detail: 详细程度（仅 OpenAI 支持）

        Returns:
            ImageContent 对象

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的图片格式

        Example:
            >>> img = ImageContent.from_file("path/to/image.jpg")
            >>> img = ImageContent.from_file("path/to/image.png", detail="high")
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在: {file_path}")

        # 根据文件扩展名确定媒体类型
        suffix = path.suffix.lower()
        media_type_mapping = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }

        media_type = media_type_mapping.get(suffix)
        if not media_type:
            raise ValueError(
                f"不支持的图片格式: {suffix}. "
                f"支持的格式: {', '.join(media_type_mapping.keys())}"
            )

        # 读取文件并编码为 base64
        with open(path, "rb") as f:
            image_data = f.read()
            base64_data = base64.b64encode(image_data).decode("utf-8")

        return cls.from_base64(
            base64_data=base64_data,
            media_type=cast(
                Literal["image/jpeg", "image/png", "image/gif", "image/webp"],
                media_type,
            ),
            detail=detail,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "content_block_type": "image",
            "image_url": self.image_url,
        }

        if self.media_type:
            result["media_type"] = self.media_type

        if self.detail:
            result["detail"] = self.detail

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageContent":
        """
        从字典创建图片内容块

        Args:
            data: 必须包含 image_url 字段，可选 media_type 和 detail 字段

        Returns:
            ImageContent 对象

        Raises:
            ValueError: 缺少必需字段
        """
        if "image_url" not in data:
            raise ValueError("ImageContent 需要 image_url 字段")
        return cls(
            image_url=data["image_url"],
            media_type=data.get("media_type"),
            detail=data.get("detail"),
        )


@dataclass
class DocumentContent(ContentBlock):
    """
    文档内容块

    统一使用 document_url 存储：
    1. HTTP/HTTPS URL: "https://example.com/document.pdf"
    2. Base64 Data: "JVBERi0xLj..." (raw base64 string)
    """

    # 文档 URL（如果是 base64，则为原始 base64 字符串）
    document_url: str

    # 媒体类型（当 document_url 为 base64 时必填）
    media_type: Optional[str] = None

    # 文件名（OpenAI 必需，Claude 可选）
    filename: Optional[str] = None

    @classmethod
    def from_url(cls, url: str, filename: Optional[str] = None) -> "DocumentContent":
        """从 URL 创建文档内容块"""
        # 如果没有提供 filename，尝试从 URL 提取
        if not filename:
            filename = url.split("/")[-1].split("?")[0] or "document.pdf"
        return cls(document_url=url, filename=filename)

    @classmethod
    def from_base64(
        cls,
        base64_data: str,
        media_type: Literal["application/pdf"] = "application/pdf",
        filename: Optional[str] = None,
    ) -> "DocumentContent":
        """从 base64 数据创建文档内容块"""
        if not filename:
            # 根据媒体类型生成默认文件名
            if media_type == "application/pdf":
                filename = "document.pdf"
            else:
                filename = "document.bin"
        return cls(document_url=base64_data, media_type=media_type, filename=filename)

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        filename: Optional[str] = None,
    ) -> "DocumentContent":
        """
        从本地文件创建文档内容块，自动检测媒体类型并转换为 base64

        Args:
            file_path: 文档文件路径
            filename: 文件名（可选，默认使用实际文件名）

        Returns:
            DocumentContent 对象

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文档格式

        Example:
            >>> doc = DocumentContent.from_file("path/to/document.pdf")
            >>> doc = DocumentContent.from_file("path/to/report.pdf", filename="report.pdf")
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"文档文件不存在: {file_path}")

        # 根据文件扩展名确定媒体类型
        suffix = path.suffix.lower()
        media_type_mapping = {
            ".pdf": "application/pdf",
        }

        media_type = media_type_mapping.get(suffix)
        if not media_type:
            raise ValueError(
                f"不支持的文档格式: {suffix}. "
                f"支持的格式: {', '.join(media_type_mapping.keys())}"
            )

        # 使用实际文件名（如果未提供）
        if not filename:
            filename = path.name

        # 读取文件并编码为 base64
        with open(path, "rb") as f:
            document_data = f.read()
            base64_data = base64.b64encode(document_data).decode("utf-8")

        return cls.from_base64(
            base64_data=base64_data,
            media_type=cast(Literal["application/pdf"], media_type),
            filename=filename,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "content_block_type": "document",
            "document_url": self.document_url,
        }

        if self.media_type:
            result["media_type"] = self.media_type

        if self.filename:
            result["filename"] = self.filename

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentContent":
        """
        从字典创建文档内容块

        Args:
            data: 必须包含 document_url 字段，可选 media_type 和 filename 字段

        Returns:
            DocumentContent 对象

        Raises:
            ValueError: 缺少必需字段
        """
        if "document_url" not in data:
            raise ValueError("DocumentContent 需要 document_url 字段")
        return cls(
            document_url=data["document_url"],
            media_type=data.get("media_type"),
            filename=data.get("filename"),
        )


@dataclass
class UnifiedMessage:
    """统一的消息格式"""

    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, list[ContentBlock]]] = None
    tool_calls: Optional[list[UnifiedToolCall]] = None
    tool_call_id: Optional[str] = None  # for tool role
    reasoning_content: Optional[str] = None
    # 厂商特有字段
    vendor_metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """
        转换为 dict，用于序列化和日志
        DEPRECATED: 这里的方法将移到 OpenAICompatibleChatCompletionStream 中去
        """
        result: dict[str, Any] = {"role": self.role}

        # 处理 content - 支持字符串或内容块数组
        if self.content is not None:
            if isinstance(self.content, str):
                result["content"] = self.content
            else:
                # 内容块数组，转换为通用格式（用于日志）
                content_list = []
                for block in self.content:
                    if isinstance(block, TextContent):
                        content_list.append({"type": "text", "text": block.text})
                    elif isinstance(block, ImageContent):
                        # 简化的格式用于日志
                        image_url = block.image_url
                        if not image_url.startswith("http"):
                            image_url = f"data:{block.media_type};base64,{image_url}"
                        img_dict: dict[str, Any] = {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        }
                        content_list.append(img_dict)
                    elif isinstance(block, DocumentContent):
                        # 文档内容块
                        doc_dict: dict[str, Any] = {"type": "document"}
                        if block.document_url.startswith("http"):
                            doc_dict["url"] = block.document_url
                        else:
                            doc_dict["data"] = (
                                f"data:{block.media_type};base64,{block.document_url[:50]}..."
                            )
                        if block.filename:
                            doc_dict["filename"] = block.filename
                        content_list.append(doc_dict)

                result["content"] = content_list

        if self.tool_calls:
            tool_calls_list = []
            for tc in self.tool_calls:
                tc_dict = {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                # gemini 的 thought_signature
                if tc.extra_content:
                    if isinstance(tc.extra_content, str):
                        tc_dict["extra_content"] = json.loads(tc.extra_content)
                    else:
                        tc_dict["extra_content"] = tc.extra_content
                tool_calls_list.append(tc_dict)
            result["tool_calls"] = tool_calls_list

        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.reasoning_content:
            result["reasoning_content"] = self.reasoning_content
        if self.vendor_metadata:
            result.update(self.vendor_metadata)

        return result

    @classmethod
    def create_user_message(
        cls,
        text: Optional[str] = None,
        images: Optional[list[Union[str, ImageContent]]] = None,
    ) -> "UnifiedMessage":
        """
        便捷方法：创建用户消息

        Args:
            text: 文本内容
            images: 图片列表，可以是 URL 字符串或 ImageContent 对象

        Examples:
            # 纯文本
            UnifiedMessage.create_user_message(text="你好")

            # 纯图片
            UnifiedMessage.create_user_message(images=["https://example.com/img.jpg"])

            # 文本 + 图片
            UnifiedMessage.create_user_message(
                text="这是什么？",
                images=["https://example.com/img.jpg"]
            )
        """
        if not text and not images:
            raise ValueError("至少需要提供 text 或 images 之一")

        # 如果只有文本，使用简单格式
        if text and not images:
            return cls(role="user", content=text)

        # 构建内容块数组
        content_blocks: list[ContentBlock] = []

        if text:
            content_blocks.append(TextContent(text=text))

        if images:
            for img in images:
                if isinstance(img, str):
                    content_blocks.append(ImageContent.from_url(img))
                else:
                    content_blocks.append(img)

        return cls(role="user", content=content_blocks)


@dataclass
class UnifiedToolParam:
    """统一的工具参数格式，与任何 SDK 解耦"""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema 格式

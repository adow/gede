# coding=utf-8
#
# test tools
#
#

from datetime import datetime, timezone


def now_tool() -> str:
    """
    Get the current date and time, use this tool when you need to determine the current time
    Returns:
        Output format includes current timezone, local time (YYYY-MM-DD HH:MM:SS Day of week) and UTC time
    """

    # Get local time
    local_now = datetime.now()
    # Get UTC time
    utc_now = datetime.now(timezone.utc)
    # Get timezone information
    timezone_name = local_now.astimezone().tzname()

    # Day of week mapping
    weekdays = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekday_en = weekdays[local_now.weekday()]

    # Format output
    local_time_str = local_now.strftime("%Y-%m-%d %H:%M:%S")
    utc_time_str = utc_now.strftime("%Y-%m-%d %H:%M:%S")

    result = f"Current timezone: {timezone_name}. Local time: {local_time_str} {weekday_en}. UTC time: {utc_time_str}"

    # context.print_tool_info(f"now_tool: \n{result}")
    return result


def get_weather_tool(day: str):
    """
    获取指定日期的天气信息

    Args:
    - day: 指定的日期，格式为 "YYYY-MM-DD"

    Returns: 返回天气信息字符串
    """
    return f"{day} 温度 6°C, 天气晴朗，适合外出。"

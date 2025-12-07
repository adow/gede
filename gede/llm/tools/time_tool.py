# coding=utf-8
#
#
from datetime import datetime, timezone
import locale
from agents import function_tool, RunContextWrapper
from ...commands import CommandConext


@function_tool(name_override="now_tool")
def now_tool(wrapper: RunContextWrapper[CommandConext]) -> str:
    """
    Get the current date and time, use this tool when you need to determine the current time
    Returns:
        Output format includes current timezone, local time (YYYY-MM-DD HH:MM:SS Day of week) and UTC time
    """
    context = wrapper.context

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

    result = f"Current timezone: {timezone_name}\nLocal time: {local_time_str} {weekday_en}\nUTC time: {utc_time_str}"

    # context.print_tool_info(f"now_tool: \n{result}")
    return result

import pyrcareworld.attributes as attr
from pyrcareworld.side_channel.side_channel import (
    IncomingMessage,
    OutgoingMessage,
)
import pyrcareworld.utils.utility as utility


def parse_message(msg: IncomingMessage) -> dict:
    this_object_data = {}
    return this_object_data


def MakeCloud(kwargs: dict) -> OutgoingMessage:
    compulsory_params = ["id", "positions"]
    utility.CheckKwargs(kwargs, compulsory_params)
    msg = OutgoingMessage()
    msg.write_int32(kwargs["id"])
    msg.write_string("MakeCloud")
    print(kwargs["positions"])
    msg.write_float32_list(kwargs["positions"])
    return msg


def SetRadius(kwargs: dict) -> OutgoingMessage:
    compulsory_params = ["id", "radius"]
    utility.CheckKwargs(kwargs, compulsory_params)
    msg = OutgoingMessage()
    msg.write_int32(kwargs["id"])
    msg.write_string("SetRadius")
    msg.write_float32(kwargs["radius"])
    return msg


def SetCloudPos(kwargs: dict) -> OutgoingMessage:
    compulsory_params = ["id", "position"]
    utility.CheckKwargs(kwargs, compulsory_params)
    msg = OutgoingMessage()
    msg.write_int32(kwargs["id"])
    msg.write_string("SetCloudPos")
    msg.write_float32_list(kwargs["position"])
    return msg

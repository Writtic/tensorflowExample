import sys
import httplib
import json
import datetime


def eventHandler(event, context, callback):

    callback["result"] = "node2"

    strall = "{"

    for element in context["previousData"]:
        l_last_result_str = element["callback"]
        l_last_result = json.loads(l_last_result_str)
        strall = strall + l_last_result["result"] + ","

    strtmp = strall[:len(strall) - 1]
    strtmp = strtmp + "}"
    conn = httplib.HTTPSConnection("hooks.slack.com")
    l_dumps_content = "previous node = " + strtmp + " / current node = node2"
    l_payload_webhook = {"text": l_dumps_content}
    params = json.dumps(l_payload_webhook)
    print(params)

    conn.request(
        "POST", "/services/T1P5CV091/B1PV8CWHX/NEB7M8Y0A5OO7SctSxntHdZt", params)

    response2 = conn.getresponse()
    print(str(response2))

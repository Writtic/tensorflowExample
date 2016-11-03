import sys
import httplib
import json
import datetime


def eventHandler(event, context, callback):
    callback["result"] = "node1"
    conn = httplib.HTTPSConnection("hooks.slack.com")
    l_dumps_content = "init node = node1"
    l_payload_webhook = {"text": l_dumps_content}
    params = json.dumps(l_payload_webhook)
    print(params)

    conn.request(
        "POST", "/services/T1P5CV091/B1PV8CWHX/NEB7M8Y0A5OO7SctSxntHdZt", params)

    response2 = conn.getresponse()
    print(str(response2))

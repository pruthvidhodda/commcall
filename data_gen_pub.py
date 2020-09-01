# Visualize our data
import numpy as np
import json
import time


from dapr.clients import DaprClient

with DaprClient() as d:
    while True:
        x = np.random.rand(10)
        y = np.sin(x) * np.power(x,3) + 3*x + np.random.rand(10)*0.8
        req_data = {
            'X': x.tolist(),
            'Y': y.tolist()
        }

        # Create a typed message with content type and body
        resp = d.publish_event(
            pubsub_name='pubsub',
            topic='DATA',
            data=json.dumps(req_data),
        )

        # Print the request
        print(req_data, flush=True)
        time.sleep(2)
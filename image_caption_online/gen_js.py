

import json
data = []
for i in range(1, 5):
    for line in open('server_test/ensemble.{}.txt'.format(i)):
        inputs = line.strip().split('\t')
        data.append({'image_id': int(inputs[0]), 'caption': inputs[1]})
json.dump(data, open('server_test/captions_test2014_reviewnet_results.json', 'w'))

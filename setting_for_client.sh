docker build -t client:0.0 -f ./client/Dockerfile .
docker run -it --entrypoint sh client:0.0

# python
# from sioClient import *

# FL token Host: token = None
# flAggregation('http://fl-aggregator.kro.kr', capacity = 1, model_bytes = 20)
# print(token) : ex) 'CnIwJYCjrwObS7Re'

# FL token Client: token = 'CnIwJYCjrwObS7Re'
# flAggregation('http://fl-aggregator.kro.kr', token = 'CnIwJYCjrwObS7Re', model_bytes = 20)
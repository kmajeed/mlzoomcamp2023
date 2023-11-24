import requests

# where to send request to test our image and return results:
url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

# We are only sending URL to the lambda function
data = {
  "url": "https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg"
}

# return results as json, lambda function should pass results that are json serializable
result = requests.post(url, json=data).json()
print(result)
---
trigger: always_on
---

Code should be written in a way to ensure modularity. Many object detection model training logic can be reused across architectures and script files.

However, make sure each function and block of code is fully customizable by the user, avoid hardcoding important configuration parameters especially for controling the process of model training and model deployment (inference). If appropriate include kwargs support so that the user may pass some optional imoirtant parameters, if the number of parameters is so large it would make the function or code unclear and hard to maintain.

Code whould be written to maximize readability and maintainability.
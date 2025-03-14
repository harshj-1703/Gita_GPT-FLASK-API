---
title: Flask API on Hugging Face
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "latest"
app_file: app.py
pinned: false
---

Check out the configuration reference at [Hugging Face Spaces Config Docs](https://huggingface.co/docs/hub/spaces-config-reference).

This Space hosts a **Flask API** that serves predictions from a **PyTorch model**. The API is deployed using **Docker**.

### 🔧 **How to Use**

- Send a `POST` request to `/predict` with JSON input.
- Example request:
  ```json
  {
    "input": [1.0, 2.0, 3.0]
  }
  ```

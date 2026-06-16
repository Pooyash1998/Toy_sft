#!/bin/bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

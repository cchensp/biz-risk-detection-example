#!/bin/bash

pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
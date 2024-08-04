#!/bin/bash

if [ -f .env ]; then
    export $(cat .env | sed 's/#.*//g' | xargs)
fi

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

streamlit run app.py

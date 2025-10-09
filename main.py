import asyncio
import sys

# Import modules theo tầng (presentation gọi application, application gọi domain/infra)
from core import config
from core import logging_setup
from models import models
from utils import utils
from tools import tools
from memory import cache
from retrieval import search
from retrieval import fetch
from services import prompt
from services import render
from agents import intent
from agents import llm
from api import ui

if __name__ == "__main__":
    demo = ui.build_ui()
    demo.queue()
    demo.launch(show_error=True, share=True)
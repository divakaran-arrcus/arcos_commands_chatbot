.PHONY: help install test-index test-rag web clean

help:
	@echo "Router Config RAG Assistant - Available Commands"
	@echo ""
	@echo "  make install      - Install Python dependencies"
	@echo "  make test-index   - Test document processor"
	@echo "  make test-rag     - Test RAG engine with sample query"
	@echo "  make web          - Start Chainlit web UI"
	@echo "  make clean        - Clean up generated files"

install:
	python3 -m venv venv
	./venv/bin/pip install -r requirements.txt

test-index:
	./venv/bin/python -m src.document_processor

test-rag:
	./venv/bin/python -m src.rag_engine "show bgp summary"

web:
	./venv/bin/chainlit run src/web_ui.py -w

clean:
	rm -rf data/chromadb/*
	rm -rf venv
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

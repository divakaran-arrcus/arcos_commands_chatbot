.PHONY: help install test-index test-rag web clean build start stop restart logs rebuild-index

# ──── Development Commands ────

help:
	@echo "Router Config RAG Assistant - Available Commands"
	@echo ""
	@echo "  Development:"
	@echo "    make install      - Install Python dependencies"
	@echo "    make test-index   - Test document processor"
	@echo "    make test-rag     - Test RAG engine with sample query"
	@echo "    make web          - Start Chainlit web UI (dev mode)"
	@echo "    make slack        - Start Slack bot (dev mode)"
	@echo ""
	@echo "  Docker Operations:"
	@echo "    make build        - Build Docker images"
	@echo "    make start        - Start all services (detached)"
	@echo "    make stop         - Stop all services"
	@echo "    make restart      - Restart all services"
	@echo "    make logs         - Tail all logs"
	@echo "    make logs-slack   - Tail Slack bot logs"
	@echo "    make logs-web     - Tail web UI logs"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make rebuild-index- Rebuild ChromaDB index"
	@echo "    make backup       - Backup ChromaDB index"
	@echo "    make clean        - Clean up generated files"
	@echo "    make ollama-status- Check Ollama status"

install:
	python3 -m venv venv
	./venv/bin/pip install -r requirements.txt

test-index:
	./venv/bin/python -m src.document_processor

test-rag:
	./venv/bin/python -m src.rag_engine "show bgp summary"

web:
	./venv/bin/chainlit run src/web_ui.py -w --port 8000

slack:
	./venv/bin/python src/slack_bot.py

# ──── Docker Commands ────

build:
	docker compose build

start:
	docker compose up -d
	@echo "✅ Services started. Web UI at http://localhost:8080"

stop:
	docker compose down

restart:
	docker compose restart

restart-slack:
	docker compose restart slack-bot

restart-web:
	docker compose restart web-ui

logs:
	docker compose logs -f

logs-slack:
	docker compose logs -f slack-bot

logs-web:
	docker compose logs -f web-ui

# ──── Maintenance Commands ────

rebuild-index:
	@echo "⏳ Rebuilding index from data/adoc_files/..."
	./venv/bin/python scripts/rebuild_index.py
	@echo "✅ Index rebuilt"

rebuild-index-dryrun:
	./venv/bin/python scripts/rebuild_index.py --dry-run

backup:
	@mkdir -p backups
	tar -czf backups/chromadb_$$(date +%Y%m%d_%H%M%S).tar.gz data/chromadb/
	@echo "✅ Backup saved to backups/"

ollama-status:
	@echo "=== Ollama Models ==="
	@curl -s http://localhost:11434/api/tags | python3 -m json.tool 2>/dev/null || echo "Ollama not responding"
	@echo ""
	@echo "=== Running Models ==="
	@curl -s http://localhost:11434/api/ps | python3 -m json.tool 2>/dev/null || echo ""

clean:
	rm -rf data/chromadb/*
	rm -rf venv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
